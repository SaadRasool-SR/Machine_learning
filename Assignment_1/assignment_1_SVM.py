
# Import necessary libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
import plotly.express as px
import plotly.graph_objects as go


# setting random seed
random_seed = 132456

# to print full data set
pd.reset_option('display.max_rows', None)
pd.reset_option('display.max_columns', None)

# importing data
Breast_Cancer_dataset = pd.read_csv('') # breast cancer
heart_dataset_test = pd.read_csv('') # heart test
heart_dataset_train = pd.read_csv('') # heart train

# EDA 
Breast_Cancer_dataset['diagnosis'] = (Breast_Cancer_dataset['diagnosis'] =='M').astype(int)


# correlation matrix
correlation = Breast_Cancer_dataset.iloc[:,1:].corr() # indexing from 2 so it doesnt take account for the id and diagnosis



# removing features with less than 0.20 correlation
features_selected = correlation[abs(correlation['diagnosis']) >=0.20]['diagnosis']

# dataset removing not required cols, based on correlation with the diagnosis variable
Breast_Cancer_dataset_x = Breast_Cancer_dataset[['radius_mean',
                          'texture_mean',
                          'perimeter_mean',
                          'area_mean',
                          'smoothness_mean',
                          'compactness_mean',
                          'concavity_mean',
                          'concave points_mean',
                          'symmetry_mean',
                          'radius_se',
                          'perimeter_se',
                          'area_se',
                          'compactness_se',
                          'concavity_se',
                          'concave points_se',
                          'radius_worst',
                          'texture_worst',
                          'perimeter_worst',
                          'area_worst',
                          'smoothness_worst',
                          'compactness_worst',
                          'concavity_worst',
                          'concave points_worst',
                          'symmetry_worst',
                          'fractal_dimension_worst'
                          ]]


Breast_Cancer_dataset_y = Breast_Cancer_dataset[['diagnosis']]

Breast_Cancer_dataset_normalized = (Breast_Cancer_dataset_x - Breast_Cancer_dataset_x.mean()) / Breast_Cancer_dataset_x.std()

train_sizes = [0.05, .10, 0.15, .20, 0.25, .30, 0.35, .40, 0.45, .50, 0.55, .60, 0.65, .70, 0.75, .80, 0.85, .90 ,0.95]
train_acc_linear = []
test_acc_linear = []
train_acc_ploy = []
test_acc_ploy = []
f1_score_linear = []
f1_score_poly = []

for i in train_sizes:

    X_train, X_test, y_train, y_test = train_test_split(Breast_Cancer_dataset_x, Breast_Cancer_dataset_y, train_size=i, random_state=random_seed, shuffle=True)

    # Initialize the SVM classifier
    svm_classifier = SVC(kernel='linear')  # You can choose different kernels like 'rbf', 'poly', etc.

    # Train the classifier
    svm_classifier.fit(X_train, y_train)


    predictions = svm_classifier.predict(Breast_Cancer_dataset_x)

    # Calculate the accuracy of the classifier
    accuracy = accuracy_score(Breast_Cancer_dataset_y, predictions)
    print("Accuracy Breast Dataset - linear kernel:", accuracy)
    train_acc_linear.append(1-accuracy)

    # Make predictions on the testing set
    predictions = svm_classifier.predict(X_test)

    # Calculate the accuracy of the classifier
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy Breast Dataset - linear kernel:", accuracy)
    test_acc_linear.append(1-accuracy)

    f1_ada = f1_score(y_test, predictions)
    f1_score_linear.append(f1_ada)


    # Initialize the SVM classifier
    svm_classifier = SVC(kernel='poly')  # You can choose different kernels like 'rbf', 'poly', etc.

    # Train the classifier
    svm_classifier.fit(X_train, y_train)
    predictions = svm_classifier.predict(Breast_Cancer_dataset_x)

    # Calculate the accuracy of the classifier
    accuracy = accuracy_score(Breast_Cancer_dataset_y, predictions)
    print("Accuracy Breast Dataset - linear kernel:", accuracy)
    train_acc_ploy.append(1-accuracy)

    # Make predictions on the testing set
    predictions = svm_classifier.predict(X_test)

    # Calculate the accuracy of the classifier
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy Breast Dataset - poly kernel:", accuracy)
    test_acc_ploy.append(1-accuracy)

    f1_ada = f1_score(y_test, predictions)
    f1_score_poly.append(f1_ada)


fig = go.Figure()
# Add traces
fig.add_trace(go.Scatter(x=train_sizes, y=train_acc_linear,
                    mode='lines',
                    name='Train-error-linear'))

fig.add_trace(go.Scatter(x=train_sizes, y=test_acc_linear,
                    mode='lines',
                    name='Test-error-linear'))

fig.add_trace(go.Scatter(x=train_sizes, y=train_acc_ploy,
                    mode='lines',
                    name='Train-error-poly'))

fig.add_trace(go.Scatter(x=train_sizes, y=test_acc_ploy,
                    mode='lines',
                    name='Test-error-poly'))


fig.update_layout(title='Test and Train Accuracy - SVM (Breast_Cancer_Dataset)')
fig.update_xaxes(title = 'Train Sizes')
fig.update_yaxes(title = 'Error')
fig.show()

fig = go.Figure()
# Add traces
fig.add_trace(go.Scatter(x=train_sizes, y=f1_score_linear,
                    mode='lines',
                    name='f1 score linear'))

fig.add_trace(go.Scatter(x=train_sizes, y=f1_score_poly,
                    mode='lines',
                    name='f1 score poly'))

fig.update_layout(title='F1 score - SVM (Breast_Cancer_Dataset)')
fig.update_xaxes(title = 'Train Sizes')
fig.update_yaxes(title = 'F1 score')
fig.show()


# dataset number 2

heart_dataset = pd.concat([heart_dataset_train, heart_dataset_test], axis=0)

# correlation matrix
correlation = heart_dataset.corr() # indexing from 2 so it doesnt take account for the id and diagnosis

# removing features with less than 0.20 correlation
features_selected = correlation[abs(correlation['target']) >=0.10]['target']

# dataset removing not required cols, based on correlation with the diagnosis variable
heart_dataset_x = heart_dataset[['age',
                                       'sex',
                                       'cp',
                                       'trestbps',
                                       'restecg',
                                       'thalach',
                                       'exang',
                                       'oldpeak',
                                       'slope',
                                       'ca',
                                       'thal'
                          ]]


heart_dataset_y = heart_dataset[['target']]


train_sizes = [0.05, .10, 0.15, .20, 0.25, .30, 0.35, .40, 0.45, .50, 0.55, .60, 0.65, .70, 0.75, .80, 0.85, .90 ,0.95]
train_acc_linear = []
test_acc_linear = []
train_acc_ploy = []
test_acc_ploy = []
f1_score_linear = []
f1_score_poly = []

for i in train_sizes:


# grid search params

    X_train, X_test, y_train, y_test = train_test_split(heart_dataset_x, heart_dataset_y, train_size=i, random_state=random_seed)


    # Initialize the SVM classifier
    svm_classifier = SVC(kernel='linear')  # You can choose different kernels like 'rbf', 'poly', etc.

    # Train the classifier
    svm_classifier.fit(X_train, y_train)


    predictions = svm_classifier.predict(heart_dataset_x)
    # Calculate the accuracy of the classifier
    accuracy = accuracy_score(heart_dataset_y, predictions)
    print("Accuracy Breast Dataset - linear kernel:", accuracy)
    train_acc_linear.append(1-accuracy)


    # Make predictions on the testing set
    predictions = svm_classifier.predict(X_test)
    # Calculate the accuracy of the classifier
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy Heart Dataset - linear kernel:", accuracy)
    test_acc_linear.append(1-accuracy)
    f1_ada = f1_score(y_test, predictions)
    f1_score_linear.append(f1_ada)


    # Initialize the SVM classifier
    svm_classifier = SVC(kernel='poly')  # You can choose different kernels like 'rbf', 'poly', etc.

    # Train the classifier
    svm_classifier.fit(X_train, y_train)

    predictions = svm_classifier.predict(heart_dataset_x)
    # Calculate the accuracy of the classifier
    accuracy = accuracy_score(heart_dataset_y, predictions)
    print("Accuracy Breast Dataset - linear kernel:", accuracy)
    train_acc_ploy.append(1-accuracy)

    # Make predictions on the testing set
    predictions = svm_classifier.predict(X_test)
    # Calculate the accuracy of the classifier
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy Heart Dataset - linear poly:", accuracy)
    test_acc_ploy.append(1-accuracy)

    f1_ada = f1_score(y_test, predictions)
    f1_score_poly.append(f1_ada)



fig = go.Figure()
# Add traces
fig.add_trace(go.Scatter(x=train_sizes, y=train_acc_linear,
                    mode='lines',
                    name='Train-error-linear'))

fig.add_trace(go.Scatter(x=train_sizes, y=test_acc_linear,
                    mode='lines',
                    name='Test-error-linear'))

fig.add_trace(go.Scatter(x=train_sizes, y=train_acc_ploy,
                    mode='lines',
                    name='Train-error-poly'))

fig.add_trace(go.Scatter(x=train_sizes, y=test_acc_ploy,
                    mode='lines',
                    name='Test-error-poly'))


fig.update_layout(title='Test and Train Accuracy - SVM (Heart Disease Detection Dataset)')
fig.update_xaxes(title = 'Train Sizes')
fig.update_yaxes(title = 'Error')
fig.show()

fig = go.Figure()
# Add traces
fig.add_trace(go.Scatter(x=train_sizes, y=f1_score_linear,
                    mode='lines',
                    name='f1 score linear'))

fig.add_trace(go.Scatter(x=train_sizes, y=f1_score_poly,
                    mode='lines',
                    name='f1 score poly'))

fig.update_layout(title='F1 score - SVM (Heart Disease Detection Dataset)')
fig.update_xaxes(title = 'Train Sizes')
fig.update_yaxes(title = 'F1 score')
fig.show()

print('completed')