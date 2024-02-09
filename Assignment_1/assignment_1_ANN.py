import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import torch
#from mlflow import MlflowClient
#import mlflow.pytorch
from torch import nn
from torch.utils.data import DataLoader
#from torchvision import datasets, transforms
import plotly.graph_objects as go
from sklearn.metrics import f1_score




# setting random seed
random_seed = 132456

torch.manual_seed(random_seed)

# to print full data set
pd.reset_option('display.max_rows', None)
pd.reset_option('display.max_columns', None)

# importing data
Breast_Cancer_dataset = pd.read_csv('/home/srasool/Documents/Machine_learning/Assignment_1/breast-cancer.csv')

# EDA 
Breast_Cancer_dataset['diagnosis'] = (Breast_Cancer_dataset['diagnosis'] =='M').astype(int)


# correlation matrix
correlation = Breast_Cancer_dataset.iloc[:,1:].corr() # indexing from 2 so it doesnt take account for the id and diagnosis



# removing features with less than 0.20 correlation
features_selected = correlation[abs(correlation['diagnosis']) >=0.20]['diagnosis']

# testing and trianing dataset

BCD_numpy_y = Breast_Cancer_dataset['diagnosis']
BCD_numpy_X = Breast_Cancer_dataset[['radius_mean',
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


X_train, X_test, y_train, y_test = train_test_split(BCD_numpy_X, BCD_numpy_y, test_size=0.3, random_state=random_seed)
X_train = torch.tensor(X_train.to_numpy(),dtype=torch.float32)
X_test = torch.tensor(X_test.to_numpy(),dtype=torch.float32)
y_train = torch.tensor(y_train.to_numpy(),dtype=torch.float32).reshape(-1,1)
y_test = torch.tensor(y_test.to_numpy(),dtype=torch.float32).reshape(-1,1)


class nn_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(in_features=25, out_features=8)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(8, 64)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(64, 1)
        self.act_output = nn.Sigmoid()
    
    def forward(self, x):
        #print(x)
        x = self.act1(self.input_layer(x))
        x = self.act2(self.hidden2(x))
        x = self.act_output(self.output(x))
        return x

learning_rate = 0.0005 
lr_dict = {}
n_epochs = 100
batch_size = [5,7,10]
#epoch_train = []
#loss_train = []
accuracy_train = []
accuracy_test = []

# training 
for bs in batch_size:
    epoch_train = []
    loss_train = []
    ann_model = nn_model()
    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(ann_model.parameters(),lr=0.0005)
    for epoch in range(n_epochs):
        for i in range(0, len(X_train), bs):
            Xbatch = X_train[i:i+bs]
            Ybatch = y_train[i:i+bs]
            y_pred = ann_model(Xbatch)
            loss = loss_function(y_pred, Ybatch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_train.append(epoch)
        loss_train.append(loss.data.item())
    lr_dict[bs] = loss_train
    
    # compute accuracy (no_grad is optional)
    with torch.no_grad():
        y_pred = ann_model(X_train)
    accuracy = (y_pred.round() == y_train).float().mean()
    accuracy_train.append(accuracy)


    # testing
    with torch.no_grad():
        pred_test = (ann_model(X_test) > 0.5).int()
    accuracy_tes = (pred_test == y_test).float().mean()
    accuracy_test.append(accuracy_tes)

# ploting training loss
fig = go.Figure()
# Add traces
for l in batch_size:
    fig.add_trace(go.Scatter(x=epoch_train, y=lr_dict[l],
                        mode='lines',
                        name='Batch Size = ' + str(l)))

fig.update_layout(title='Training Loss vs Batch Size - Batch Size Optimization')
fig.update_xaxes(title = 'Epochs')
fig.update_yaxes(title = 'Training Loss - Breast_Cancer_dataset')
fig.show()


learning_rate = [0.01, 0.005, 0.001, 0.0005, 0.0001]
lr_dict = {}
n_epochs = 100
batch_size = 5
#epoch_train = []
#loss_train = []
accuracy_train = []
accuracy_test = []


# training 
for Lr in learning_rate:
    epoch_train = []
    loss_train = []
    ann_model = nn_model()
    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(ann_model.parameters(),lr=Lr)
    for epoch in range(n_epochs):
        for i in range(0, len(X_train), batch_size):
            Xbatch = X_train[i:i+batch_size]
            Ybatch = y_train[i:i+batch_size]
            y_pred = ann_model(Xbatch)
            loss = loss_function(y_pred, Ybatch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_train.append(epoch)
        loss_train.append(loss.data.item())
    lr_dict[Lr] = loss_train
    
    # compute accuracy (no_grad is optional)
    with torch.no_grad():
        y_pred = ann_model(X_train)
    accuracy = (y_pred.round() == y_train).float().mean()
    accuracy_train.append(accuracy)


    # testing
    with torch.no_grad():
        pred_test = (ann_model(X_test) > 0.5).int()
    accuracy_tes = (pred_test == y_test).float().mean()
    accuracy_test.append(accuracy_tes)

# ploting training loss
fig = go.Figure()
# Add traces
for l in learning_rate:
    fig.add_trace(go.Scatter(x=epoch_train, y=lr_dict[l],
                        mode='lines',
                        name='lr = ' + str(l)))

fig.update_layout(title='Training Loss vs Epochs - Learning Rate Optimization')
fig.update_xaxes(title = 'Epochs')
fig.update_yaxes(title = 'Training Loss - Breast_Cancer_dataset')
fig.show()


learning_rate = 0.0001
lr_dict = {}
n_epochs = 100
batch_size = 7
accuracy_train = []
accuracy_test = []
f1_score_ada =[]

# # training 

epoch_train = []
loss_train = []
ann_model = nn_model()
loss_function = nn.BCELoss()
optimizer = torch.optim.Adam(ann_model.parameters(),lr=learning_rate)
for epoch in range(n_epochs):
    for i in range(0, len(X_train), batch_size):
        Xbatch = X_train[i:i+batch_size]
        Ybatch = y_train[i:i+batch_size]
        y_pred = ann_model(Xbatch)
        loss = loss_function(y_pred, Ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    epoch_train.append(epoch)
    loss_train.append(loss.data.item())



    # compute accuracy (no_grad is optional)
    with torch.no_grad():
        y_pred = ann_model(X_train)
    accuracy = (y_pred.round() == y_train).float().mean()
    accuracy_train.append(accuracy)


    # testing
    with torch.no_grad():
        pred_test = (ann_model(X_test) > 0.5).int()
    accuracy_tes = (pred_test == y_test).float().mean()
    accuracy_test.append(accuracy_tes)
    f1_ada = f1_score(y_test, pred_test)
    f1_score_ada.append(f1_ada)

fig = go.Figure()
# Add traces
fig.add_trace(go.Scatter(x=epoch_train, y=accuracy_train,
                    mode='lines',
                    name='Train'))

fig.add_trace(go.Scatter(x=epoch_train, y=accuracy_test,
                    mode='lines',
                    name='Test'))
fig.add_trace(go.Scatter(x=epoch_train, y=accuracy_test,
                    mode='lines',
                    name='F1 Score'))

fig.update_layout(title='Test and Train Accuracy - Learning Rate Optimization')
fig.update_xaxes(title = 'Epochs')
fig.update_yaxes(title = 'Accuracy')
fig.show()



print('done')

# ###### Data Set 2


heart_dataset_test = pd.read_csv('/home/srasool/Documents/Machine_learning/Assignment_1/heart_test.csv')
heart_dataset_train = pd.read_csv('/home/srasool/Documents/Machine_learning/Assignment_1/heart_train.csv')
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



X_train, X_test, y_train, y_test = train_test_split(heart_dataset_x, heart_dataset_y, test_size=0.3, random_state=random_seed)
X_train = torch.tensor(X_train.to_numpy(),dtype=torch.float32)
X_test = torch.tensor(X_test.to_numpy(),dtype=torch.float32)
y_train = torch.tensor(y_train.to_numpy(),dtype=torch.float32).reshape(-1,1)
y_test = torch.tensor(y_test.to_numpy(),dtype=torch.float32).reshape(-1,1)


class nn_model_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(in_features=11, out_features=8)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(8, 64)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(64, 1)
        self.act_output = nn.Sigmoid()
    
    def forward(self, x):
        #print(x)
        x = self.act1(self.input_layer(x))
        x = self.act2(self.hidden2(x))
        x = self.act_output(self.output(x))
        return x


learning_rate= [0.001, 0.0001, 0.0005, 0.00001]
n_epochs = 250
batch_size = 5
lr_dict = {}
accuracy_train = []
accuracy_test = []
f1_score_ada = []


# training 
for lr in learning_rate:
    epoch_train = []
    loss_train = []
    ann_model_2 = nn_model_2()
    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(ann_model_2.parameters(),lr=lr)
    for epoch in range(n_epochs):
        for i in range(0, len(X_train), batch_size):
            Xbatch = X_train[i:i+batch_size]
            Ybatch = y_train[i:i+batch_size]
            y_pred = ann_model_2(Xbatch)
            loss = loss_function(y_pred, Ybatch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_train.append(epoch)
        loss_train.append(loss.data.item())
        print(epoch,loss.data.item())
    lr_dict[lr] = loss_train


    # compute accuracy (no_grad is optional)
    with torch.no_grad():
        y_pred = ann_model_2(X_train)
    accuracy = (y_pred.round() == y_train).float().mean()
    accuracy_train.append(accuracy)


    # testing
    with torch.no_grad():
        pred_test = (ann_model_2(X_test) > 0.5).int()
    accuracy_tes = (pred_test == y_test).float().mean()
    accuracy_test.append(accuracy_tes)
    f1_ada = f1_score(y_test, pred_test)
    f1_score_ada.append(f1_ada)

# ploting training loss
fig = go.Figure()
# Add traces
for l in learning_rate:
    fig.add_trace(go.Scatter(x=epoch_train, y=lr_dict[l],
                        mode='lines',
                        name='lr = ' + str(l)))

fig.update_layout(title='Training Loss vs Epochs - Learning Rate Optimization (Heart Dataset)')
fig.update_xaxes(title = 'Epochs')
fig.update_yaxes(title = 'Training Loss')
fig.show()


learning_rate= 0.001
n_epochs = 250
batch_size = [2,3,5,7]
lr_dict = {}
accuracy_train = []
accuracy_test = []


# training 
for bs in batch_size:
    epoch_train = []
    loss_train = []
    ann_model_2 = nn_model_2()
    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(ann_model_2.parameters(),lr=learning_rate)
    for epoch in range(n_epochs):
        for i in range(0, len(X_train), bs):
            Xbatch = X_train[i:i+bs]
            Ybatch = y_train[i:i+bs]
            y_pred = ann_model_2(Xbatch)
            loss = loss_function(y_pred, Ybatch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_train.append(epoch)
        loss_train.append(loss.data.item())
        print(epoch,loss.data.item())
    lr_dict[bs] = loss_train


    # # compute accuracy (no_grad is optional)
    # with torch.no_grad():
    #     y_pred = ann_model_2(X_train)
    # accuracy = (y_pred.round() == y_train).float().mean()
    # accuracy_train.append(accuracy)


    # # testing
    # with torch.no_grad():
    #     pred_test = (ann_model_2(X_test) > 0.5).int()
    # accuracy_tes = (pred_test == y_test).float().mean()
    # accuracy_test.append(accuracy_tes)

fig = go.Figure()
# Add traces
for l in batch_size:
    fig.add_trace(go.Scatter(x=epoch_train, y=lr_dict[l],
                        mode='lines',
                        name='batch size = ' + str(l)))

fig.update_layout(title='Training Loss vs Epochs - Batch Size Optimization (Heart Dataset)')
fig.update_xaxes(title = 'Epochs')
fig.update_yaxes(title = 'Training Loss')
fig.show()


learning_rate= 0.0001
n_epochs = 250
batch_size = 2
lr_dict = {}
accuracy_train = []
accuracy_test = []


# training 

epoch_train = []
loss_train = []
ann_model_2 = nn_model_2()
loss_function = nn.BCELoss()
optimizer = torch.optim.Adam(ann_model_2.parameters(),lr=learning_rate)
for epoch in range(n_epochs):
    for i in range(0, len(X_train), bs):
        Xbatch = X_train[i:i+bs]
        Ybatch = y_train[i:i+bs]
        y_pred = ann_model_2(Xbatch)
        loss = loss_function(y_pred, Ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    epoch_train.append(epoch)
    loss_train.append(loss.data.item())
    print(epoch,loss.data.item())

    # compute accuracy (no_grad is optional)
    with torch.no_grad():
        y_pred = ann_model_2(X_train)
    accuracy = (y_pred.round() == y_train).float().mean()
    accuracy_train.append(accuracy)


    # testing
    with torch.no_grad():
        pred_test = (ann_model_2(X_test) > 0.5).int()
    accuracy_tes = (pred_test == y_test).float().mean()
    accuracy_test.append(accuracy_tes)

fig = go.Figure()
# Add traces
fig.add_trace(go.Scatter(x=epoch_train, y=accuracy_train,
                    mode='lines',
                    name='Train'))

fig.add_trace(go.Scatter(x=epoch_train, y=accuracy_test,
                    mode='lines',
                    name='Test'))

fig.add_trace(go.Scatter(x=epoch_train, y=accuracy_test,
                    mode='lines',
                    name='F1 Score'))

fig.update_layout(title='Test and Train Accuracy - (Heart Dataset)')
fig.update_xaxes(title = 'Epochs')
fig.update_yaxes(title = 'Accuracy')
fig.show()

print('completed')


