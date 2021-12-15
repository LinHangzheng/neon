import argparse
from  options import parse_options
from train import *
import logging as log
log.basicConfig(format='[%(asctime)s] [INFO] %(message)s', 
                datefmt='%d/%m %H:%M:%S',
                level=log.INFO)

if __name__ == '__main__':
    
    args, args_str = parse_options()
    trainer = Trainer(args)
    trainer.train()
