import torch
# torchvision.transforms: common image transformations.
# They accept tensor images/batch of tensor images.
# A Tensor Image is a tensor with (C, H, W) shape, where C is a number of channels, H and W are image height and width.
import torchvision.transforms as transforms
# Torchvision provides many built-in datasets in the torchvision.datasets module, 
# as well as utility classes for building your own datasets.
from torchvision import datasets
 
# loading training data
train_dataset = datasets.MNIST(root='./data', 
                               train=True, 
                               # Convert a PIL Image or ndarray to tensor and scale the values accordingly.
                               transform=transforms.ToTensor(),
                               download=True)
#loading test data
test_dataset = datasets.MNIST(root='./data', 
                              train=False, 
                              transform=transforms.ToTensor())

print("number of training samples: " + str(len(train_dataset)) + "\n" +
      "number of testing samples: " + str(len(test_dataset)))