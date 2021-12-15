import torch
import os
from colorize_data import *
from torch.utils.data import Dataset, DataLoader
from basic_model import *
import torch.optim as optim
from tqdm import tqdm
import logging as log
import wandb
os.environ['CUDA_VISIBLE_DEVICES']='0'
wandb.init(project="colorization", entity="hangzheng")

class Trainer:
    def __init__(self,args):
        # Define hparams here or load them from a config file
        # Set device to use

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        device_name = torch.cuda.get_device_name(device=self.device)
        log.info(f'Using {device_name} with CUDA v{torch.version.cuda}')
        self.log_dict = {}

        self.args = args
        self.set_dataset()
        self.set_dataloader()
        self.set_net()
        self.set_optimizer()

        
        wandb.config = {
        "learning_rate": self.args.lr,
        "epochs": self.args.epochs,
        "batch_size": self.args.batch_size
        }

    def set_dataset(self):
        self.train_dataset = ColorizeData(self.args.train_path)    
        self.val_dataset = ColorizeData(self.args.test_path)

        
    def set_dataloader(self):
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, 
                                            shuffle=True, pin_memory=True, num_workers=0)
        
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.args.batch_size, 
                                            shuffle=True, pin_memory=True, num_workers=0)
    
    
    def set_net(self):
        self.net = Net().to(self.device)
        wandb.watch(self.net)


    def set_optimizer(self):
        if self.args.optimizer == 'adam':
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(self.net.parameters(), lr=self.args.lr, momentum=0.8)
        else:
            raise ValueError('Invalid optimizer.')


    def train(self):
        # train loop
        for epoch in range(self.args.epochs):
            self.pre_epoch()
            self.iterate()    
            self.post_epoch(epoch)
            if epoch % self.args.valid_every == 0:
                self.validate(epoch)
            

    def pre_epoch(self):
        self.log_dict['total_loss'] = 0
        self.log_dict['total_iter_count'] = 0
    
    def iterate(self):
        self.net.train()
        for n_iter, data in enumerate(tqdm(self.train_dataloader)):
            x = data[0].to(self.device)
            y = data[1].to(self.device)

            # Prepare for inference
            batch_size = x.shape[0]
            self.net.zero_grad()

            # Calculate loss
            loss = 0
            preds = self.net(x)
            theta = 1
            l1_regularization=0
            for i in range(batch_size):
                z = preds[i] - y[i]
                L1_idx = torch.where(torch.abs(z)>=theta)
                L2_idx = torch.where(torch.abs(z)<theta)
                loss +=  (1/2*z[L2_idx]**2).sum()+(theta*(torch.abs(z[L1_idx])-1/2*theta)).sum()
            
            # add L1 norm
            # for param in self.net.parameters():
            #     l1_regularization += torch.norm(param, 1)
            
            loss /= batch_size
            
            # loss += 0.00005*l1_regularization

            


            self.log_dict['total_loss'] += loss.item()
            self.log_dict['total_iter_count'] += batch_size


            # Backpropagate
            loss.backward()
            self.optimizer.step()




    def validate(self,epoch):
        pass
        # Validation loop begin
        # ------
        self.net.eval()
        for n_iter, data in enumerate(tqdm(self.val_dataloader)):
            x = data[0].to(self.device)
            y = data[1].to(self.device)

            # Prepare for inference
            batch_size = x.shape[0]
            self.net.zero_grad()

            # Calculate loss
            loss = 0
            preds = self.net(x)
            for i in range(batch_size):
                loss += ((preds[i]-y[i])**2).sum()
            
        loss /= len(self.val_dataloader)

        log_text = 'EPOCH {}/{}'.format(epoch+1, self.args.epochs)
        log_text += ' | validation loss: {:>.3E}'.format(loss)
        log.info(log_text)
        wandb.log({"validation loss": loss})
        # Validation loop end
        # ------
        # Determine your evaluation metrics on the validation dataset.


    def post_epoch(self, epoch):
        self.net.eval()
        self.log_tb(epoch)
        if epoch % self.args.save_every == 0:
            self.save_model(epoch)


    def save_model(self, epoch):
        if not os.path.exists(self.args.model_path):
            os.makedirs(self.args.model_path)
        model_fname = os.path.join(self.args.model_path,'net_0001')
        torch.save(self.net.state_dict(), model_fname)


    def log_tb(self,epoch):
        log_text = 'EPOCH {}/{}'.format(epoch+1, self.args.epochs)
        self.log_dict['total_loss'] /= self.log_dict['total_iter_count'] + 1e-6
        log_text += ' | total loss: {:>.3E}'.format(self.log_dict['total_loss'])
        log.info(log_text)
        wandb.log({"training loss": self.log_dict['total_loss']})

        
    