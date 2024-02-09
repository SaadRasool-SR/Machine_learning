import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

# setting random seed
random_seed = 132456

# to print full data set
pd.reset_option('display.max_rows', None)
pd.reset_option('display.max_columns', None)

# importing data
Breast_Cancer_dataset = pd.read_csv('') #breast cancer
heart_dataset_test = pd.read_csv('') #heart  test
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

Breast_Cancer_dataset_x_normalized = (Breast_Cancer_dataset_x - Breast_Cancer_dataset_x.mean()) / Breast_Cancer_dataset_x.std()

Breast_Cancer_dataset_y = Breast_Cancer_dataset[['diagnosis']]

X_train, X_test, y_train, y_test = train_test_split(Breast_Cancer_dataset_x_normalized, Breast_Cancer_dataset_y, train_size=0.8, random_state=random_seed)

K = [] 
training = [] 
test = [] 
scores = {}
f1_score_ls = []
  
for k in range(2,10): 
    clf = KNeighborsClassifier(n_neighbors = k) 
    clf.fit(X_train.values, np.ravel(y_train))
    training_score=clf.score(X_train.values,np.ravel(y_train))
    training.append(training_score) 
    test_score=clf.score(X_test.values,np.ravel(y_test))
    predictions = clf.predict(X_test.values)
    K.append(k) 
    training.append(training_score) 
    test.append(test_score) 
    scores[k] = [training_score, test_score]
    f1_ada = f1_score(y_test, predictions)
    f1_score_ls.append(f1_ada)



# ploting  
fig = go.Figure()
# Add traces
fig.add_trace(go.Scatter(x=K, y=training,
                    mode='lines',
                    name='Train-error'))

fig.add_trace(go.Scatter(x=K, y=test,
                    mode='lines',
                    name='Test-error'))

fig.add_trace(go.Scatter(x=K, y=f1_score_ls,
                    mode='lines',
                    name='F1 Score'))


fig.update_layout(title='Test and Train Accuracy - KNN - Breast Cancer Dataset')
fig.update_xaxes(title = 'Number of Clusters')
fig.update_yaxes(title = 'Accuracy')
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

heart_dataset_normalized = (heart_dataset_x - heart_dataset_x.mean()) / heart_dataset_x.std()

# grid search params

X_train, X_test, y_train, y_test = train_test_split(heart_dataset_normalized, heart_dataset_y, train_size=.5, random_state=random_seed)
 
K = [] 
training = [] 
test = [] 
scores = {}
f1_score_ls = []
  
for k in range(2,25): 
    clf = KNeighborsClassifier(n_neighbors = k,) 
    clf.fit(X_train.values, np.ravel(y_train))
    training_score=clf.score(X_train.values,np.ravel(y_train))
    training.append(training_score) 
    test_score=clf.score(X_test.values,np.ravel(y_test))
    predictions = clf.predict(X_test.values)
    K.append(k) 
    training.append(training_score) 
    test.append(test_score) 
    scores[k] = [training_score, test_score]
    f1_ada = f1_score(y_test, predictions)
    f1_score_ls.append(f1_ada)

# ploting  
fig = go.Figure()
# Add traces
fig.add_trace(go.Scatter(x=K, y=training,
                    mode='lines',
                    name='Train-error'))

fig.add_trace(go.Scatter(x=K, y=test,
                    mode='lines',
                    name='Test-error'))

fig.add_trace(go.Scatter(x=K, y=f1_score_ls,
                    mode='lines',
                    name='F1 Score'))


fig.update_layout(title='Test and Train Accuracy - KNN - Heart Dataset')
fig.update_xaxes(title = 'Number of Clusters')
fig.update_yaxes(title = 'Accuracy')
fig.show()
