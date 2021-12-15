# Image Colorization Starter Code
The objective is to produce color images given grayscale input image. 

## Setup Instructions
Create a conda environment with pytorch, cuda. 

`$ conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia`

For systems without a dedicated gpu, you may use a CPU version of pytorch.
`$ conda install pytorch torchvision torchaudio cpuonly -c pytorch`

## Dataset
Use the zipfile provided as your dataset. You are expected to split your dataset to create a validation set for initial testing. Your final model can use the entire dataset for training. Note that this model will be evaluated on a test dataset not visible to you.

## Code Guide
Baseline Model: A baseline model is available in `basic_model.py` You may use this model to kickstart this assignment. We use 256 x 256 size images for this problem.
-	Fill in the dataloader, (colorize_data.py)
-	Fill in the loss function and optimizer. (train.py)
-	Complete the training loop, validation loop (train.py)
-	Determine model performance using appropriate metric. Describe your metric and why the metric works for this model? 
- Prepare an inference script that takes as input grayscale image, model path and produces a color image. 

## Additional Tasks 
- The network available in model.py is a very simple network. How would you improve the overall image quality for the above system? (Implement)
- You may also explore different loss functions here.

## Bonus
You are tasked to control the average color/mood of the image that you are colorizing. What are some ideas that come to your mind? (Bonus: Implement)

## Solution
- Document the things you tried, what worked and did not.   
I tried different model structures and finally chose the pretrained resnet18 with a decoder  
I tried to use the dropout layer and the L1 loss for weights, but they didn't show too much difference.  
I tried to use the MSE loss and the loss that combine both the L1 and L2 loss. The last one performed better.  
I tried to use the data augmentation with image crop, rotation, flip to increase the number of data.  
I tried different learning rates and finally set the lr=0.0001  

- Update this README.md file to add instructions on how to run your code. (train, inference).   
Note. All following commands should be ran within the `neon` directory.
To train the model, you can frist try to use data augmentation by using  

```
python image-colorization_d6a566/preprocess.py
```
which will generate splited dataset in `data/train` and `data/test`  
Then, run the training by using   
```
python image-colorization_d6a566/main.py
```
Other inputs are described in `image-colorization-d6a566/options.py`. Feel free to change them.  

For inference, you can use 
```
python image-colorization_d6a566/colorization.py --input-path [The path to .jpg file]
```
The result will in the `_result` folder  

- Once you are done, zip the code, upload your solution.  
