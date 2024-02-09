import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (GridSearchCV, cross_val_score,
                                     train_test_split)
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

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

# balance / sample points:
fig = go.Figure(data=[go.Histogram(x=Breast_Cancer_dataset['diagnosis'])])

# Customize layout
fig.update_layout(
    title="Breast Cancer Dataset Target",
    xaxis_title="Target",
    yaxis_title="Frequency",
    bargap=0.1,  # gap between bars
    bargroupgap=0.05,  # gap between groups of bars
)

# Show plot
fig.show()

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

X_train, X_test, y_train, y_test = train_test_split(Breast_Cancer_dataset_x, Breast_Cancer_dataset_y, train_size=0.7, random_state=random_seed, shuffle=True)

acc_score_test_adaB = []
acc_score_train_adaB = []
n_estimators = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
for n in n_estimators:
    ############ adaboost Training
    clf_AdaBoost = AdaBoostClassifier(n_estimators=n, algorithm="SAMME.R", learning_rate=1,random_state=random_seed)
    clf_AdaBoost.fit(X_train,y_train)
    y_pred_train_ada = clf_AdaBoost.predict(Breast_Cancer_dataset_x)
    acc_train_adaB = accuracy_score(Breast_Cancer_dataset_y, y_pred_train_ada)
    acc_score_train_adaB.append(1-acc_train_adaB)
    
    ############ adaboost Testing
    y_pred_adaB = clf_AdaBoost.predict(X_test)
    acc_score_adaB = accuracy_score(y_test, y_pred_adaB)
    acc_score_test_adaB.append(1-acc_score_adaB)

# ploting  

fig = go.Figure()
# Add traces
fig.add_trace(go.Scatter(x=n_estimators, y=acc_score_train_adaB,
                    mode='lines',
                    name='Train-error-AdaBoost'))

fig.add_trace(go.Scatter(x=n_estimators, y=acc_score_test_adaB,
                    mode='lines',
                    name='Test-error-AdaBoost'))


fig.update_layout(title='Test and Train Accuracy - AdaBoost N_Estimator (Breast_Cancer_dataset)')
fig.update_xaxes(title = 'Number of Trees')
fig.update_yaxes(title = 'Error')
fig.show()


# grid search
param = {'ccp_alpha': [0.1, .01, .001],
            'max_depth' : [5, 6, 7, 8, 9],
            }
clf = DecisionTreeClassifier(ccp_alpha = 0.01, criterion = 'entropy', max_depth= None, random_state=random_seed)
grid_search = GridSearchCV(estimator=clf,param_grid=param,cv=5, verbose=True)
grid_search.fit(X_train,y_train)


#best model
final_model = grid_search.best_estimator_
print(final_model)


train_sizes = [0.05, .10, 0.15, .20, 0.25, .30, 0.35, .40, 0.45, .50, 0.55, .60, 0.65, .70, 0.75, .80, 0.85, .90 ,0.95]
acc_score_test = []
acc_score_train = []
acc_score_test_adaB = []
acc_score_train_adaB = []
f1_score_dt = []
f1_score_ada = []

for i in train_sizes:

    # spiliting data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(Breast_Cancer_dataset_x, Breast_Cancer_dataset_y, train_size=i, random_state=random_seed, shuffle=True)


    # Training 
    final_model.fit(X_train,y_train)

    y_pred_train = final_model.predict(Breast_Cancer_dataset_x)
    acc_train = accuracy_score(Breast_Cancer_dataset_y, y_pred_train)
    acc_score_train.append(1-acc_train)

    # Prediction 
    y_pred = final_model.predict(X_test)
    acc_test = accuracy_score(y_test, y_pred)
    acc_score_test.append(1-acc_test)

    f1 = f1_score(y_test, y_pred)
    f1_score_dt.append(f1)


    ############ adaboost Training
    clf_AdaBoost = AdaBoostClassifier(n_estimators=55, algorithm="SAMME.R",learning_rate=1,random_state=random_seed)
    clf_AdaBoost.fit(X_train,y_train)
    y_pred_train_ada = clf_AdaBoost.predict(Breast_Cancer_dataset_x)
    acc_train_adaB = accuracy_score(Breast_Cancer_dataset_y, y_pred_train_ada)
    acc_score_train_adaB.append(1-acc_train_adaB)

    ############ adaboost Testing
    y_pred_adaB = clf_AdaBoost.predict(X_test)
    acc_score_adaB = accuracy_score(y_test, y_pred_adaB)
    acc_score_test_adaB.append(1-acc_score_adaB)
    f1_ada = f1_score(y_test, y_pred_adaB)
    f1_score_ada.append(f1_ada)


fig = go.Figure()
# Add traces
fig.add_trace(go.Scatter(x=train_sizes, y=f1_score_dt,
                    mode='lines',
                    name='F1 Score'))

fig.add_trace(go.Scatter(x=train_sizes, y=f1_score_ada,
                    mode='lines',
                    name='F1 Score AdaBoost'))


fig.update_layout(title='F1 Score - Decision Tree (Breast_Cancer_Dataset)')
fig.update_xaxes(title = 'Train Sizes')
fig.update_yaxes(title = 'F1 Score')
fig.show()




# ploting  

fig = go.Figure()
# Add traces
fig.add_trace(go.Scatter(x=train_sizes, y=acc_score_train,
                    mode='lines',
                    name='Train-error'))

fig.add_trace(go.Scatter(x=train_sizes, y=acc_score_test,
                    mode='lines',
                    name='Test-error'))

fig.add_trace(go.Scatter(x=train_sizes, y=acc_score_train_adaB,
                    mode='lines',
                    name='Train-error-AdaBoost'))

fig.add_trace(go.Scatter(x=train_sizes, y=acc_score_test_adaB,
                    mode='lines',
                    name='Test-error-AdaBoost'))


fig.update_layout(title='Test and Train Accuracy - Decision Tree (Breast_Cancer_Dataset)')
fig.update_xaxes(title = 'Train Sizes')
fig.update_yaxes(title = 'Error')
fig.show()





# dataset 2
#heart_dataset_test = pd.read_csv('/home/srasool/Documents/Machine_learning/Assignment_1/heart_test.csv')
#heart_dataset_train = pd.read_csv('/home/srasool/Documents/Machine_learning/Assignment_1/heart_train.csv')
heart_dataset = pd.concat([heart_dataset_train, heart_dataset_test], axis=0)


# correlation matrix
correlation = heart_dataset.corr() # indexing from 2 so it doesnt take account for the id and diagnosis



# removing features with less than 0.20 correlation
features_selected = correlation[abs(correlation['target']) >=0.20]['target']


# balance / sample points:
fig = go.Figure(data=[go.Histogram(x=heart_dataset['target'])])

# Customize layout
fig.update_layout(
    title="Heart Disease Detection Dataset Target",
    xaxis_title="Target",
    yaxis_title="Frequency",
    bargap=0.1,  # gap between bars
    bargroupgap=0.05,  # gap between groups of bars
)

# Show plot
fig.show()
fig.write_image("/home/srasool/Documents/Machine_learning/Assignment_1/Images/Heart Disease Detection_dataset_target.png")

# dataset removing not required cols, based on correlation with the diagnosis variable
heart_dataset_x = heart_dataset[['age',
                                       'sex',
                                       'cp',
                                       'thalach',
                                       'exang',
                                       'oldpeak',
                                       'slope',
                                       'ca',
                                       'thal'
                          ]]


heart_dataset_y = heart_dataset[['target']]


# grid search params

X_train, X_test, y_train, y_test = train_test_split(heart_dataset_x, heart_dataset_y, train_size=.70, random_state=random_seed)

acc_score_test_adaB = []
acc_score_train_adaB = []
n_estimators = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140]
for n in n_estimators:
    ############ adaboost Training
    clf_AdaBoost = AdaBoostClassifier(n_estimators=n, algorithm="SAMME.R", learning_rate=.9,random_state=random_seed)
    clf_AdaBoost.fit(X_train,y_train)
    y_pred_train_ada = clf_AdaBoost.predict(heart_dataset_x)
    acc_train_adaB = accuracy_score(heart_dataset_y, y_pred_train_ada)
    acc_score_train_adaB.append(1-acc_train_adaB)
    
    ############ adaboost Testing
    y_pred_adaB = clf_AdaBoost.predict(X_test)
    acc_score_adaB = accuracy_score(y_test, y_pred_adaB)
    acc_score_test_adaB.append(1-acc_score_adaB)

# ploting  

fig = go.Figure()
# Add traces
fig.add_trace(go.Scatter(x=n_estimators, y=acc_score_train_adaB,
                    mode='lines',
                    name='Train-error-AdaBoost'))

fig.add_trace(go.Scatter(x=n_estimators, y=acc_score_test_adaB,
                    mode='lines',
                    name='Test-error-AdaBoost'))


fig.update_layout(title='Test and Train Accuracy - AdaBoost N_Estimator (heart_dataset)')
fig.update_xaxes(title = 'Number of Trees')
fig.update_yaxes(title = 'Error')
fig.show()

fig.write_image("/home/srasool/Documents/Machine_learning/Assignment_1/Images/Error_AdaBoost_N_estimator_heart_dataset.png")


# grid search
param = {'ccp_alpha': [0.1, .01, .001],
            'max_depth' : [5, 6, 7, 8, 9],
            }
clf_2 = DecisionTreeClassifier(ccp_alpha = 0.01, criterion = 'entropy', max_depth=5, random_state=random_seed)
grid_search = GridSearchCV(estimator=clf_2,param_grid=param,cv=5, verbose=True)
grid_search.fit(X_train,y_train)

#best model
final_model = grid_search.best_estimator_
print(final_model)




train_sizes = [0.05, .10, 0.15, .20, 0.25, .30, 0.35, .40, 0.45, .50, 0.55, .60, 0.65, .70, 0.75, .80, 0.85, .90 ,0.95]
acc_score_test = []
acc_score_train = []
acc_score_train_adaB=[]
acc_score_test_adaB=[]
f1_score_dt = []
f1_score_ada = []

for i in train_sizes:

    # spiliting data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(heart_dataset_x, heart_dataset_y, train_size=i, random_state=random_seed)

    # Training 
    final_model.fit(X_train,y_train)
    y_pred_train = final_model.predict(heart_dataset_x)
    acc_train = accuracy_score(heart_dataset_y, y_pred_train)
    acc_score_train.append(1-acc_train)


    #Testing
    y_pred_test = final_model.predict(X_test)
    acc_test = accuracy_score(y_test, y_pred_test)
    acc_score_test.append(1-acc_test)
    f1 = f1_score(y_test, y_pred_test)
    f1_score_dt.append(f1)

    ############ adaboost Training
    clf_AdaBoost = AdaBoostClassifier(n_estimators=140, algorithm="SAMME.R",learning_rate=0.9,random_state=random_seed)
    clf_AdaBoost.fit(X_train,y_train)
    y_pred_train_ada = clf_AdaBoost.predict(heart_dataset_x)
    acc_train_adaB = accuracy_score(heart_dataset_y, y_pred_train_ada)
    acc_score_train_adaB.append(1-acc_train_adaB)

    ############ adaboost Testing
    y_pred_adaB = clf_AdaBoost.predict(X_test)
    acc_score_adaB = accuracy_score(y_test, y_pred_adaB)
    acc_score_test_adaB.append(1-acc_score_adaB)
    f1_ada = f1_score(y_test, y_pred_adaB)
    f1_score_ada.append(f1_ada)

fig = go.Figure()
# Add traces
fig.add_trace(go.Scatter(x=train_sizes, y=f1_score_dt,
                    mode='lines',
                    name='F1 Score'))

fig.add_trace(go.Scatter(x=train_sizes, y=f1_score_ada,
                    mode='lines',
                    name='F1 Score AdaBoost'))


fig.update_layout(title='F1 Score - Decision Tree (Heart_Dataset)')
fig.update_xaxes(title = 'Train Sizes')
fig.update_yaxes(title = 'F1 Score')
fig.show()





# ploting  
fig = go.Figure()
# Add traces
fig.add_trace(go.Scatter(x=train_sizes, y=acc_score_train,
                    mode='lines',
                    name='Train-error'))

fig.add_trace(go.Scatter(x=train_sizes, y=acc_score_test,
                    mode='lines',
                    name='Test-error'))

fig.add_trace(go.Scatter(x=train_sizes, y=acc_score_train_adaB,
                    mode='lines',
                    name='Train-error-AdaBoost'))

fig.add_trace(go.Scatter(x=train_sizes, y=acc_score_test_adaB,
                    mode='lines',
                    name='Test-error-AdaBoost'))

fig.update_layout(title='Test and Train Accuracy - Decision Tree (heart_dataset)')
fig.update_xaxes(title = 'Train Sizes')
fig.update_yaxes(title = 'Error')
fig.show()


print('completed')
