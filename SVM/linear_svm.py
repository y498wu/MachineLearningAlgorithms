import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

# linear data
X = np.array([1, 5, 1.5, 8, 1, 9, 7, 8.7, 2.3, 5.5, 7.7, 6.1])
y = np.array([2, 8, 1.8, 8, 0.6, 11, 10, 9.4, 4, 3, 8.8, 7.5])

# shaping data for training the model
training_X = np.vstack((X, y)).T
# The value 0 indicates that a data point belongs to the first class
# and the value 1 indicates that a data point belongs to the second class.
training_y = [0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1]

# define the model
# C: regularization parameter
clf = svm.SVC(kernel='linear', C=1.0)

# train the model
clf.fit(training_X, training_y)

# get the weight values for the linear equation from the trained SVM model
w = clf.coef_[0]
# w: [0.1332713  0.42667492]

# Decision boundary equation: 
# w[0] * x0 + w[1] * x1 + b = 0
# To plot decision boundary, we need to express it as a function like x1 = f(x0):
# x1 = (-w[0]/w[1]) * x0 - (b/w[1])

# get the y-offset for the linear equation
a = -w[0] / w[1]

# make the x-axis space for the data points
# the default number of points is 50
XX = np.linspace(0, 13)

# get the y-values to plot the decision boundary
# clf.intercept_[0] is b
yy = a * XX - clf.intercept_[0] / w[1]

# plot the decision boundary
plt.plot(XX, yy, 'k-')

# show the plot visually
plt.scatter(training_X[:, 0], training_X[:, 1], c=training_y, marker='*')
plt.legend()
plt.show()