import pandas as pd # for data manipulation
import numpy as np # for data manipulation

from sklearn.model_selection import train_test_split # for splitting the data into train and test samples
from sklearn.metrics import classification_report # for model evaluation metrics
from sklearn.svm import SVC # for Support Vector Classification model

import plotly.express as px  # for data visualization
import plotly.graph_objects as go # for data visualization

# Read in the csv
df=pd.read_csv('datasets/games.csv', encoding='utf-8')

# Difference between white rating and black rating - independent variable
df['rating_difference']=df['white_rating']-df['black_rating']

# White wins flag (1=win vs. 0=not-win) - dependent (target) variable
df['white_win']=df['winner'].apply(lambda x: 1 if x=='white' else 0)

# Print a snapshot of a few columns
df.iloc[:,[0,1,5,6,8,9,10,11,13,16,17]]

def fitting(X, y, C, gamma):
    # Create training and testing samples
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Fit the model
    # Note, available kernels: {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’

    # probability: False by default. This must be enabled prior to calling fit

    # C: regularization parameter
    # A smaller value of C creates a larger margin at the cost of more margin violations, 
    # while a larger value of C leads to a smaller margin, but fewer margin violations. 

    # gamma: a hyperparameter that determines the shape of the decision boundary.
    # A high gamma will result in a more complex decision boundary, 
    # and a low gamma will result in a smoother boundary.
    model = SVC(kernel='rbf', probability=True, C=C, gamma=gamma)
    clf = model.fit(X_train, y_train)

    # Predict class labels on training data
    pred_labels_tr = model.predict(X_train)
    # Predict class labels on a test data
    pred_labels_te = model.predict(X_test)

    # Use score method to get accuracy of the model
    print('----- Evaluation on Test Data -----')
    # score_te: The score is calculated as the ratio of the number of correctly classified samples 
    # to the total number of samples in the test set.
    score_te = model.score(X_test, y_test)
    print('Accuracy Score: ', score_te)
    # Look at classification report to evaluate the model
    # Precision – Accuracy of positive predictions.
    # Precision = TP/(TP + FP)
    # Recall: Fraction of positives that were correctly identified.
    # Recall = TP/(TP+FN)
    # F1 Score = 2*(Recall * Precision) / (Recall + Precision)
    # The F1 score is a weighted harmonic mean of precision and recall 
    # such that the best score is 1.0 and the worst is 0.0
    print(classification_report(y_test, pred_labels_te))
    print('--------------------------------------------------------')

    print('----- Evaluation on Training Data -----')
    score_tr = model.score(X_train, y_train)
    print('Accuracy Score: ', score_tr)
    # Look at classification report to evaluate the model
    print(classification_report(y_train, pred_labels_tr))
    print('--------------------------------------------------------')
    
    # Evaluate a ML model on both training and test data separately to check for overfitting
    # If the model performs well on training data but poorly on test data, it may be overfitting. 
    # If the model performs poorly on both training and test data, it may be underfitting
    
    # Return relevant data for chart plotting
    return X_train, X_test, y_train, y_test, clf


def Plot_3D(X, X_test, y_test, clf):
            
    # Specify a size of the mesh to be used
    mesh_size = 5
    margin = 1

    # Create a mesh grid on which we will run our model

    # Get the minimum and maximum values for the x and y axes of the mesh grid.
    # The fillna method is used to replace any missing values in the data with the mean of the column.
    x_min, x_max = X.iloc[:, 0].fillna(X.mean()).min() - margin, X.iloc[:, 0].fillna(X.mean()).max() + margin
    y_min, y_max = X.iloc[:, 1].fillna(X.mean()).min() - margin, X.iloc[:, 1].fillna(X.mean()).max() + margin
    xrange = np.arange(x_min, x_max, mesh_size)
    yrange = np.arange(y_min, y_max, mesh_size)
    xx, yy = np.meshgrid(xrange, yrange)
            
    # Calculate predictions on grid

    # Predict will give either 0 or 1 as output. 
    # Whereas Predict_proba will give the probability of both 0 and 1.

    # np.c_ : add along second axis
    # e.g.: np.c_[np.array([[1,2,3]]), 0, 0, np.array([[4,5,6]])]
    # shapes:
    # np.array([[1,2,3]]) = 1,3
    # np.array([[4,5,6]]) = 1,3
    # we can think of it as [[0]] = 1,1
    # result 1,3+1+1+3 = 1,8
    # which is the shape of result : array([[1, 2, 3, 0, 0, 4, 5, 6]])

    # Another e.g.: # both are 2 dimensional array
    # a = array([[1, 2, 3], [4, 5, 6]])
    # b = array([[7, 8, 9], [10, 11, 12]])
    # The new shape should be (2, 3 + 3) = (2, 6)
    # so the result is: 
    # [[1,2,3,7,8,9],
    #  [4,5,6,10,11,12]]

    # The numpy.ravel() functions returns contiguous flattened array
    # (1D array with all the input-array elements and with the same type as it).

    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # numpy.reshape: gives a new shape to an array without changing its data.
    # e.g.: >>> a = np.arange(6).reshape((3, 2))
    # >>> a  
    # array([[0, 1],
    #        [2, 3],
    #        [4, 5]])
    Z = Z.reshape(xx.shape)

    # Create a 3D scatter plot with predictions
    fig = px.scatter_3d(x=X_test['rating_difference'], y=X_test['turns'], z=y_test, 
                     opacity=0.8, color_discrete_sequence=['black'])

    # Set figure title and colors
    fig.update_layout(#title_text="Scatter 3D Plot with SVM Prediction Surface",
                      paper_bgcolor = 'white',
                      scene = dict(xaxis=dict(backgroundcolor='white',
                                              color='black',
                                              gridcolor='#f0f0f0'),
                                   yaxis=dict(backgroundcolor='white',
                                              color='black',
                                              gridcolor='#f0f0f0'
                                              ),
                                   zaxis=dict(backgroundcolor='lightgrey',
                                              color='black', 
                                              gridcolor='#f0f0f0', 
                                              )))
    # Update marker size
    fig.update_traces(marker=dict(size=1))

    # Add prediction plane
    fig.add_traces(go.Surface(x=xrange, y=yrange, z=Z, name='SVM Prediction',
                              colorscale='RdBu', showscale=False, 
                              contours = {"z": {"show": True, "start": 0.2, "end": 0.8, "size": 0.05}}))
    fig.show()

# Select data for modeling
X=df[['rating_difference', 'turns']]
y=df['white_win'].values

# Fit the model and display results
X_train, X_test, y_train, y_test, clf = fitting(X, y, 1, 'scale')

Plot_3D(X, X_test, y_test, clf)