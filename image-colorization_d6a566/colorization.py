from options import *
from basic_model import Net
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms as T

if __name__ == '__main__':
    parser = parse_options(return_parser=True)
    colorization_group = parser.add_argument_group('colorization')
    colorization_group.add_argument('--img-dir', type=str, default='_results/',
                           help='Directory to output the images')
    colorization_group.add_argument('--input-path', type=str,
                           help='path to the input images')
    args = parser.parse_args()

    name = args.input_path.split('/')[-1].split('.')[0]

    # Pick device
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # Make output directory
    if not os.path.exists(args.img_dir):
        os.makedirs(args.img_dir)
    
    # Get model
    net = Net()
    net.load_state_dict(torch.load(args.model_path+'net_0001'))
    net.to(device)
    net.eval()

    # Use the input transform to convert images to grayscale
    input_transform = T.Compose([T.ToTensor(),
                        T.Resize(size=(256,256))
                        ])

    img = plt.imread(args.input_path)
    img=Image.fromarray(img)
    img=input_transform(img).to(device)

    def tensor_to_PIL(tensor):
        unloader = T.ToPILImage()
        image = tensor.cpu().clone()
        image = image.squeeze(0)
        image = unloader(image)
        return image


    # gray_img = input_transform(img).to(device)
    # gray_img = gray_img.detach()  
    color_img = net(img.unsqueeze(0))

    # gray_img = tensor_to_PIL(gray_img)
    # gray_img.save(args.img_dir+name+'_gray.jpg')

    
    color_img = tensor_to_PIL(color_img)
    color_img.save(args.img_dir+name+'_color.jpg')
 
    
