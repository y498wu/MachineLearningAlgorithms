import torch
# torchvision.transforms: common image transformations.
# They accept tensor images/batch of tensor images.
# A Tensor Image is a tensor with (C, H, W) shape, where C is a number of channels, H and W are image height and width.
import torchvision.transforms as transforms
# Torchvision provides many built-in datasets in the torchvision.datasets module, 
# as well as utility classes for building your own datasets.
from torchvision import datasets
import matplotlib.pyplot as plt
 
# loading training data
train_dataset = datasets.MNIST(root='./data', 
                               train=True, 
                               # Convert a PIL Image or ndarray to tensor and scale the values accordingly.
                               transform=transforms.ToTensor(),
                               download=True)
# loading test data
test_dataset = datasets.MNIST(root='./data', 
                              train=False, 
                              transform=transforms.ToTensor())

# verify number of training and testing samples in the dataset
print("number of training samples: " + str(len(train_dataset)) + "\n" +
      "number of testing samples: " + str(len(test_dataset)))

# inspect the data type and size of the first element in the training data
# the first index [0] represents that this is the first element
# the second index [0] represents that this is the image
# Each sample in the dataset is a pair of image and label.
print("datatype of the 1st training sample: ", train_dataset[0][0].type())
print("size of the 1st training sample: ", train_dataset[0][0].size())

# print result: 
# datatype of the 1st training sample:  torch.FloatTensor
# size of the 1st training sample:  torch.Size([1, 28, 28])

# The first sample in the dataset is a FloatTensor and it is a 
# 28x28-pixel image in grayscale, hence the size [1, 28, 28].

# print the labels of the first two samples in the training set.
# the first index [0] and [1] represent that these are the first and second elements
# the second index [1] represents that these are labels
print("label of the first taining sample: ", train_dataset[0][1])
print("label of the second taining sample: ", train_dataset[1][1])

# show images of the first two elements in training dataset
img_5 = train_dataset[0][0].numpy().reshape(28, 28)
plt.imshow(img_5, cmap='gray')
plt.show()
img_0 = train_dataset[1][0].numpy().reshape(28, 28)
plt.imshow(img_0, cmap='gray')
plt.show()