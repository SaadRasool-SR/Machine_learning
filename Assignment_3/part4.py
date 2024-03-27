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
from sklearn.preprocessing import StandardScaler
import time


# importing data
################### breast cancer dataset

number_of_iterations = 100

breast_cancer_dataset = pd.read_csv('Assignment_3/Data/breast-cancer.csv') # breast cancer
breast_cancer_dataset['diagnosis'] = (breast_cancer_dataset['diagnosis'] =='M').astype(int)
breast_cancer_dataset_y = breast_cancer_dataset['diagnosis']
breast_cancer_dataset_x = breast_cancer_dataset.drop('diagnosis', axis=1)
breast_cancer_dataset_x = breast_cancer_dataset_x.drop('id', axis=1)
scaler = StandardScaler()
scaled_cancer_dataset_x = scaler.fit_transform(breast_cancer_dataset_x)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(scaled_cancer_dataset_x, breast_cancer_dataset_y, test_size=0.3, random_state=42, shuffle = True)
X_train = torch.tensor(X_train,dtype=torch.float32)
X_test = torch.tensor(X_test,dtype=torch.float32)
y_train = torch.tensor(y_train.to_numpy(),dtype=torch.float32).reshape(-1,1)
y_test = torch.tensor(y_test.to_numpy(),dtype=torch.float32).reshape(-1,1)

class nn_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(in_features=30, out_features=8)
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
    
avg_accuracy_train = np.zeros((1, 100))
avg_accuracy_test = np.zeros((1, 100))
avg_f1_score_ada =np.zeros((1, 100))
avg_run_time_without =[]
learning_rate = 0.0001
lr_dict = {}
n_epochs = 100


for i in range(number_of_iterations):
    start_time_without_ica = time.time()
    learning_rate = 0.0001
    lr_dict = {}
    n_epochs = 100
    batch_size = 7
    accuracy_train = np.empty((0,)) #[]
    accuracy_test = np.empty((0,)) #[]
    f1_score_ada = np.empty((0,)) #[]
    # # training 
    start_time_without_ica = time.time()
    epoch_train = np.empty((0,))#[]
    loss_train = np.empty((0,))#[]
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
        epoch_train =np.append(epoch_train,epoch)
        loss_train = np.append(loss_train,loss.data.item())

        
        #epoch_train.append(epoch)
        #loss_train.append(loss.data.item())



        # compute accuracy (no_grad is optional)
        with torch.no_grad():
            y_pred = ann_model(X_train)
        accuracy = (y_pred.round() == y_train).float().mean()
        #accuracy_train.append(accuracy)
        accuracy_train =np.append(accuracy_train,accuracy)


        # testing
        with torch.no_grad():
            pred_test = (ann_model(X_test) > 0.5).int()
        accuracy_tes = (pred_test == y_test).float().mean()
        accuracy_test =np.append(accuracy_test,accuracy_tes)
        f1_ada = f1_score(y_test, pred_test)
        #f1_score_ada.append(f1_ada)
        f1_score_ada = np.append(f1_score_ada,f1_ada)

    #avg_accuracy_train.append(epoch_train/len())
    
    

    end_time_without_ica = time.time()
    total_time_without_ica = end_time_without_ica - start_time_without_ica
    avg_run_time_without.append(total_time_without_ica)
    avg_accuracy_train = avg_accuracy_train + accuracy_train
    avg_accuracy_test = avg_accuracy_test + accuracy_test
    avg_f1_score_ada = avg_f1_score_ada + f1_score_ada

avg_accuracy_train = avg_accuracy_train/number_of_iterations
avg_accuracy_test = avg_accuracy_test/number_of_iterations
avg_f1_score_ada = avg_f1_score_ada/number_of_iterations
avg_run_time_without = sum(avg_run_time_without)/number_of_iterations


#### ICA Implementation
breast_cancer_dataset = pd.read_csv('Assignment_3/Data/breast-cancer.csv') # breast cancer
breast_cancer_dataset['diagnosis'] = (breast_cancer_dataset['diagnosis'] =='M').astype(int)
breast_cancer_dataset_y = breast_cancer_dataset['diagnosis']
breast_cancer_dataset_x = breast_cancer_dataset.drop('diagnosis', axis=1)
breast_cancer_dataset_x_ica = breast_cancer_dataset_x.drop(['perimeter_mean',
                                                        'smoothness_mean',
                                                        'symmetry_mean',
                                                        'radius_se',
                                                        'texture_se',
                                                        'perimeter_se',
                                                        'area_se',
                                                        'smoothness_se',
                                                        'concavity_se',
                                                        'concave points_se',
                                                        'symmetry_se',
                                                        'fractal_dimension_se',
                                                        'smoothness_worst',
                                                        'symmetry_worst',
                                                        'fractal_dimension_worst'], axis=1)

scaler = StandardScaler()
scaled_cancer_dataset_x_ica = scaler.fit_transform(breast_cancer_dataset_x_ica)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(scaled_cancer_dataset_x_ica, breast_cancer_dataset_y, test_size=0.3, random_state=42, shuffle = True)
X_train = torch.tensor(X_train,dtype=torch.float32)
X_test = torch.tensor(X_test,dtype=torch.float32)
y_train = torch.tensor(y_train.to_numpy(),dtype=torch.float32).reshape(-1,1)
y_test = torch.tensor(y_test.to_numpy(),dtype=torch.float32).reshape(-1,1)


learning_rate_ica = 0.0001
lr_dict_ica = {}
n_epochs_ica = 100
batch_size_ica = 7
#accuracy_train_ica = []
#accuracy_test_ica = []
#f1_score_ada_ica =[]

avg_accuracy_train_ica = np.zeros((1, 100))
avg_accuracy_test_ica = np.zeros((1, 100))
avg_f1_score_ada_ica =np.zeros((1, 100))
avg_run_time_ica =[]



class nn_model_ica(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(in_features=16, out_features=8)
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

for i in range(number_of_iterations):
# # training 
    start_time_with_ica = time.time()
    epoch_train_ica = np.empty((0,))
    loss_train_ica = np.empty((0,))
    ann_model = nn_model_ica()
    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(ann_model.parameters(),lr=learning_rate_ica)
    accuracy_train_ica = np.empty((0,)) #[]
    accuracy_test_ica = np.empty((0,)) #[]
    f1_score_ada_ica = np.empty((0,)) #[]
    for epoch in range(n_epochs_ica):
        for i in range(0, len(X_train), batch_size_ica):
            Xbatch = X_train[i:i+batch_size_ica]
            Ybatch = y_train[i:i+batch_size_ica]
            y_pred = ann_model(Xbatch)
            loss = loss_function(y_pred, Ybatch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_train_ica = np.append(epoch_train_ica,epoch)
        loss_train_ica = np.append(loss_train_ica,loss.data.item())

        # compute accuracy (no_grad is optional)
        with torch.no_grad():
            y_pred = ann_model(X_train)
        accuracy = (y_pred.round() == y_train).float().mean()
        accuracy_train_ica =np.append(accuracy_train_ica,accuracy)
        # testing
        with torch.no_grad():
            pred_test = (ann_model(X_test) > 0.5).int()
        accuracy_tes = (pred_test == y_test).float().mean()
        accuracy_test_ica=np.append(accuracy_test_ica,accuracy_tes)
        f1_ada = f1_score(y_test, pred_test)
        f1_score_ada_ica=np.append(f1_score_ada_ica,f1_ada)
    end_time_with_ica = time.time()
    total_time_with_ica = end_time_with_ica - start_time_with_ica

    avg_run_time_ica.append(total_time_with_ica)
    avg_accuracy_train_ica = avg_accuracy_train_ica + accuracy_train
    avg_accuracy_test_ica = avg_accuracy_test_ica + accuracy_test
    avg_f1_score_ada_ica = avg_f1_score_ada_ica + f1_score_ada

avg_accuracy_train_ica = avg_accuracy_train_ica/number_of_iterations
avg_accuracy_test_ica = avg_accuracy_test_ica/number_of_iterations
avg_f1_score_ada_ica = avg_f1_score_ada_ica/number_of_iterations
avg_run_time_ica = sum(avg_run_time_ica)/number_of_iterations

avg_accuracy_train_std = np.std(avg_accuracy_train[0],axis=0)
avg_accuracy_test_std = np.std(avg_accuracy_test[0],axis=0)
avg_f1_score_ada_std = np.std(avg_f1_score_ada[0],axis=0)
avg_accuracy_train_ica_std = np.std(avg_accuracy_train_ica[0],axis=0)
avg_accuracy_test_ica_std = np.std(avg_accuracy_test_ica[0],axis=0)
avg_f1_score_ada_ica_std = np.std(avg_f1_score_ada_ica[0],axis=0)

fig = go.Figure()
# Add traces
fig.add_trace(go.Scatter(x=epoch_train, y=avg_accuracy_train[0],
                    mode='lines',
                    name='Train Without ICA'))

fig.add_trace(go.Scatter(x=epoch_train, y=avg_accuracy_test[0],
                    mode='lines',
                    name='Test without ICA'))
fig.add_trace(go.Scatter(x=epoch_train, y=avg_f1_score_ada[0],
                    mode='lines',
                    name='F1 Score Without ICA'))

fig.add_trace(go.Scatter(x=epoch_train, y=avg_accuracy_train_ica[0],
                    mode='lines',
                    name='Train With ICA'))

fig.add_trace(go.Scatter(x=epoch_train, y=avg_accuracy_test_ica[0],
                    mode='lines',
                    name='Test With ICA'))
fig.add_trace(go.Scatter(x=epoch_train, y=avg_f1_score_ada_ica[0],
                    mode='lines',
                    name='F1 Score With ICA'))

fig.update_layout(title='Test, Train Accuracy with F1 Score')
fig.update_xaxes(title = 'Epochs')
fig.update_yaxes(title = 'Accuracy')
fig.show()

print('completed')
# Sample data

total_times= [round(avg_run_time_ica,3), round(avg_run_time_without,3)]
experiments = ['With ICA', 'Without ICA']
colors = ['blue', 'green']

      # Create bar plot
fig = go.Figure()

fig.add_trace(go.Bar(x=experiments, y=total_times, text=total_times, textposition='outside',marker=dict(color=colors)))

# Update layout
fig.update_layout(title='Execution Time',
                  xaxis_title='Experiment',
                  yaxis_title='Seconds',
                  barmode='group')  # Use 'group' for grouped bars

# Show the plot
fig.show()