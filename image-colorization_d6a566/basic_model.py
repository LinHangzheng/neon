import torch.nn as nn
import torchvision.models as models
import torch

class Net(nn.Module):
  def __init__(self, input_size=256):
    super(Net, self).__init__()
    

    # ResNet - First layer accepts grayscale images, 
    # and we take only the first few layers of ResNet for this task
    resnet = models.resnet18(pretrained=True)

    resnet.conv1.weight = nn.Parameter(resnet.conv1.weight.sum(dim=1).unsqueeze(1)) 
    self.midlevel_resnet = nn.Sequential(*list(resnet.children())[0:-2])
    RESNET_FEATURE_SIZE = 512
    ## Upsampling Network
    self.upsample = nn.Sequential(     
      nn.Conv2d(RESNET_FEATURE_SIZE, 256, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU(),
      nn.Upsample(scale_factor=2),

      nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.Upsample(scale_factor=2),

      nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Upsample(scale_factor=2),

      nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.Upsample(scale_factor=2),

      nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(3),
      nn.ReLU(),
      nn.Upsample(scale_factor=2)
    )
    self.discriminator = nn.Sequential(
      nn.Conv2d(4,4, kernel_size = 1, stride = 1, padding=0),
      nn.ReLU(),
      nn.Conv2d(4,3, kernel_size = 1, stride = 1, padding=0)
    )

  def forward(self, input):
    with torch.no_grad():
      midlevel_features = self.midlevel_resnet(input)
    output = self.upsample(midlevel_features)
    output = self.discriminator(torch.cat((output,input),dim=1))
    output = torch.tanh(output)
    return output
