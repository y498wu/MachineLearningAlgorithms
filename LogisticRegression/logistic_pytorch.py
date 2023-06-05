import torch
# torchvision.transforms: common image transformations.
# They accept tensor images/batch of tensor images.
# A Tensor Image is a tensor with (C, H, W) shape, 
# where C is a number of channels, H and W are image height and width.
import torchvision.transforms as transforms
# Torchvision provides many built-in datasets in the torchvision.datasets module, 
# as well as utility classes for building your own datasets.
from torchvision import datasets
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
 
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
# Yeshu tip: cmap='PuRd' is also great! Feels like a purple-red gem -w-
plt.imshow(img_5, cmap='GnBu')
plt.show()
img_0 = train_dataset[1][0].numpy().reshape(28, 28)
plt.imshow(img_0, cmap='GnBu')
plt.show()

# load train and test data samples into dataloader
# DataLoader allows you to read data in batches, not samples.
batach_size = 32
train_loader = DataLoader(dataset=train_dataset, batch_size=batach_size, shuffle=True) 
test_loader = DataLoader(dataset=test_dataset, batch_size=batach_size, shuffle=False)

# build custom module for logistic regression
# torch.nn.Module: Base class for all neural network modules. 
# Your models should also subclass this class.
class LogisticRegression(torch.nn.Module):    
    # build the constructor
    def __init__(self, n_inputs, n_outputs):
        super(LogisticRegression, self).__init__()  
        # Applies a linear transformation to the incoming data: y = x * A^T + b
        self.linear = torch.nn.Linear(n_inputs, n_outputs)
    # make predictions
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
    
# instantiate the model
n_inputs = 28*28 # makes a 1D vector of 784
n_outputs = 10
log_regr = LogisticRegression(n_inputs, n_outputs)

# defining the optimizer
optimizer = torch.optim.SGD(log_regr.parameters(), lr=0.001)
# defining Cross-Entropy loss
criterion = torch.nn.CrossEntropyLoss()

epochs = 50
Loss = []
acc = []
