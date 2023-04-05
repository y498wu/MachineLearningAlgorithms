import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn import svm

# non-linear data
circle_X, circle_y = datasets.make_circles(n_samples=300, noise=0.05)

# make non-linear algorithm for model
nonlinear_clf = svm.SVC(kernel='rbf', C=1.0)

# training non-linear model
nonlinear_clf.fit(circle_X, circle_y)

# Plot the decision boundary for a non-linear SVM problem
def plot_decision_boundary(model, ax=None):
    # ax: axes object, a rectangular area in a figure  
    # where plots, images, and other graphical elements can be added.
    if ax is None:
        # get current axes
        ax = plt.gca()
        
    # the range of values that are displayed on the x-axis and y-axis
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    # 30 represents the number of nodes
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    # create a grid of x and y coordinate pairs over the range of the data
    Y, X = np.meshgrid(y, x)

	# shape data
    # ravel() is a NumPy function
    # used to flatten the arrays X and Y obtained from np.meshgrid into 1D arrays
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    
	# get the decision boundary based on the model
    # Reshaping the output using reshape(X.shape) will ensure that 
    # we have a 2D array of shape (30, 30) that we can pass to ax.contour(). 
    # This way, we can plot the decision boundary on the same grid as the scatter plot.
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary
    # levels=[0] specifies that only the contour line where P=0 is drawn
    # The value of alpha ranges from 0 (fully transparent) to 1 (fully opaque). 
    # In this code, alpha=0.5 specifies that the decision boundary should be semi-transparent
    ax.contour(X, Y, P,
               levels=[0], alpha=0.5,
               linestyles=['dashed'])

# plot data and decision boundary
plt.scatter(circle_X[:, 0], circle_X[:, 1], c=circle_y, marker='*', s=50)
plot_decision_boundary(nonlinear_clf)
# nonlinear_clf.support_vectors holds the coordinates of the support vectors

# [:, 0] returns an array of the x-coordinates of the support vectors
# The colon : before the comma indicates that we want to select all rows, 
# and the 0 after the comma indicates that we want to select the first column.
# [:, 1] returns an array of the y-coordinates of the support vectors

# s=50 sets the size of the markers for the support vectors to 50
# lw=1 sets the linewidth of the marker edge to 1
# facecolors='none' sets the color of the marker face to none, meaning the markers are transparent.
plt.scatter(nonlinear_clf.support_vectors_[:, 0], nonlinear_clf.support_vectors_[:, 1], s=50, lw=1, facecolors='none')
plt.show()