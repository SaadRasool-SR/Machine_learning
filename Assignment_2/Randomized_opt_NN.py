
import mlrose_hiive as mlrose
import numpy as np
import plotly.graph_objects as go
import time
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler



# setting random seed
random_seed = 132456

torch.manual_seed(random_seed)

# to print full data set
pd.reset_option('display.max_rows', None)
pd.reset_option('display.max_columns', None)


heart_dataset_test = pd.read_csv('/home/srasool/Documents/Machine_learning/Assignment_2/Data/heart_test.csv')
heart_dataset_train = pd.read_csv('/home/srasool/Documents/Machine_learning/Assignment_2/Data/heart_train.csv')

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

scaler = StandardScaler()
scaled_heart_dataset_x = scaler.fit_transform(heart_dataset_x)

X_train, X_test, y_train, y_test = train_test_split(scaled_heart_dataset_x, heart_dataset_y, test_size=0.3, random_state=random_seed)
X_train = torch.tensor(X_train,dtype=torch.float32)
X_test = torch.tensor(X_test,dtype=torch.float32)
y_train = torch.tensor(y_train.to_numpy(),dtype=torch.float32).reshape(-1,1)
y_test = torch.tensor(y_test.to_numpy(),dtype=torch.float32).reshape(-1,1)

# max_iteration = list(range(250))

# train_accuracy = []
# test_accuracy = []

# #for i in max_iteration:

#     # Initialize neural network object and fit object


# og_model = mlrose.NeuralNetwork(hidden_nodes = [11,8,64,1], activation = 'relu', \
#                                 algorithm = 'gradient_descent', max_iters = 10000, \
#                                 bias = True, is_classifier = True, learning_rate = 0.00001, \
#                                 early_stopping = True,curve=True, max_attempts = 10000, \
#                                 random_state = random_seed)



# og_model  = og_model.fit(X_train,y_train)

# print('gradient_descent')

# rhc_model = mlrose.NeuralNetwork(hidden_nodes = [11,8,64,1], activation = 'relu', \
#                                 algorithm = 'random_hill_climb', max_iters = 10000, \
#                                 bias = True, is_classifier = True, learning_rate = 0.00001, \
#                                 early_stopping = True,curve=True, max_attempts = 100, \
#                                 random_state = random_seed)

# rhc_model  = rhc_model.fit(X_train,y_train)
# fig = go.Figure()

# fig.add_trace(go.Scatter(x=list(range(len(rhc_model.fitness_curve))), y=rhc_model.fitness_curve,
#                     mode='lines',
#                     name='Random_Hill_Climb'))

# fig.update_layout(title='Fitness Curves')
# fig.update_xaxes(title = 'Number of Training Iterations')
# fig.update_yaxes(title = 'Fitness')
# fig.show()
# fig.write_image('/home/srasool/Documents/Machine_learning/Assignment_2/images/Fitness_nn_ro.png')

# print('done-random_hill_climb')

# sa_model = mlrose.NeuralNetwork(hidden_nodes = [11,8,64,1], activation = 'relu', \
#                                 algorithm = 'simulated_annealing', max_iters = 10000, \
#                                 bias = True, is_classifier = True, learning_rate = 0.00001, \
#                                 early_stopping = True,curve=True, max_attempts = 10000, \
#                                 random_state = random_seed)

# sa_model  = sa_model.fit(X_train,y_train)
# print('done-simulated_annealing')

# gen_model = mlrose.NeuralNetwork(hidden_nodes = [11,8,64,1], activation = 'relu', \
#                                 algorithm = 'genetic_alg', max_iters = 100, \
#                                 bias = True, is_classifier = True, learning_rate = 0.00001, \
#                                 early_stopping = True,curve=True, max_attempts = 100, \
#                                 random_state = random_seed,pop_size=500, mutation_prob=0.2)

# gen_model  = gen_model.fit(X_train,y_train)
# print('done-genetic_alg')



# fig = go.Figure()
# # Add traces
# # fig.add_trace(go.Scatter(x=list(range(len(og_model.fitness_curve))), y=og_model.fitness_curve,
# #                     mode='lines',
# #                     name='Gradient_Descent'))

# fig.add_trace(go.Scatter(x=list(range(len(rhc_model.fitness_curve))), y=rhc_model.fitness_curve,
#                     mode='lines',
#                     name='Random_Hill_Climb'))

# fig.add_trace(go.Scatter(x=list(range(len(sa_model.fitness_curve))), y=sa_model.fitness_curve,
#                     mode='lines',
#                     name='Simulated_Annealing'))

# fig.add_trace(go.Scatter(x=list(range(len(gen_model.fitness_curve))), y=gen_model.fitness_curve,
#                     mode='lines',
#                     name='Genetic_Alg'))

# fig.update_layout(title='Fitness Curves')
# fig.update_xaxes(title = 'Number of Training Iterations')
# fig.update_yaxes(title = 'Fitness')
# fig.show()
# fig.write_image('/home/srasool/Documents/Machine_learning/Assignment_2/images/Fitness_nn_ro.png')

########################################################## F1 Score

iter= 500

max_iteration = list(range(iter))

test_accuracy_gd = []
f1_score_test_gd = []

test_accuracy_rhc = []
f1_score_test_rhc = []

test_accuracy_sa = []
f1_score_test_sa = []

test_accuracy_gen_a = []
f1_score_test_gen_a = []

for i in max_iteration:

    # Initialize neural network object and fit object


    og_model = mlrose.NeuralNetwork(hidden_nodes = [11,8,64,1], activation = 'relu', \
                                    algorithm = 'gradient_descent', max_iters = i, \
                                    bias = False, is_classifier = True, learning_rate = 0.0001, \
                                    early_stopping = True,curve=True, max_attempts = 1000, \
                                    random_state = random_seed)



    og_model  = og_model.fit(X_train,y_train)
    y_test_predict = og_model.predict(X_test)
    f1 = f1_score(y_test, y_test_predict)
    test_accuracy_gd.append(accuracy_score(y_test,y_test_predict))
    f1_score_test_gd.append(f1)


    print('gradient_descent')

    rhc_model = mlrose.NeuralNetwork(hidden_nodes = [11,8,64,1], activation = 'relu', \
                                    algorithm = 'random_hill_climb', max_iters = i, \
                                    bias = False, is_classifier = True, learning_rate = 0.1, \
                                    early_stopping = True,curve=True, max_attempts = 1000, \
                                    random_state = random_seed)

    rhc_model  = rhc_model.fit(X_train,y_train)
    y_test_predict = rhc_model.predict(X_test)
    f1 = f1_score(y_test, y_test_predict)
    test_accuracy_rhc.append(accuracy_score(y_test,y_test_predict))
    f1_score_test_rhc.append(f1)

    print('done-random_hill_climb')

    schedule = mlrose.GeomDecay(init_temp=5, decay=0.75, min_temp=0.001)

    sa_model = mlrose.NeuralNetwork(hidden_nodes = [11,8,64,1], activation = 'relu', \
                                    algorithm = 'simulated_annealing',schedule=schedule, max_iters = i, \
                                    bias = False, is_classifier = True, learning_rate = 0.1, \
                                    early_stopping = True,curve=True, max_attempts = 1000, \
                                    restarts=2 ,random_state = random_seed)

    sa_model  = sa_model.fit(X_train,y_train)
    y_test_predict = sa_model.predict(X_test)
    f1 = f1_score(y_test, y_test_predict)
    test_accuracy_sa.append(accuracy_score(y_test,y_test_predict))
    f1_score_test_sa.append(f1)
    print('done-simulated_annealing')

    gen_model = mlrose.NeuralNetwork(hidden_nodes = [11,8,64,1], activation = 'relu', \
                                    algorithm = 'genetic_alg', max_iters = i, \
                                    bias = False, is_classifier = True, learning_rate = 0.1, \
                                    early_stopping = True,curve=True, max_attempts = 1000, \
                                    random_state = random_seed,pop_size=150, mutation_prob=0.4)

    gen_model  = gen_model.fit(X_train,y_train)
    y_test_predict = gen_model.predict(X_test)
    f1 = f1_score(y_test, y_test_predict)
    test_accuracy_gen_a.append(accuracy_score(y_test,y_test_predict))
    f1_score_test_gen_a.append(f1)
    print('done-genetic_alg')


    print(i)

# plot

fig = go.Figure()
# Add traces
fig.add_trace(go.Scatter(x=list(range(iter)), y=test_accuracy_gd,
                    mode='lines',
                    name='gradient_descent'))

fig.add_trace(go.Scatter(x=list(range(iter)), y=test_accuracy_rhc,
                    mode='lines',
                    name='random_hill_climb'))

fig.add_trace(go.Scatter(x=list(range(iter)), y=test_accuracy_sa,
                    mode='lines',
                    name='simulated_annealing'))

fig.add_trace(go.Scatter(x=list(range(iter)), y=test_accuracy_gen_a,
                    mode='lines',
                    name='genetic_alg'))


fig.update_layout(title='Test Accuracy')
fig.update_xaxes(title = 'Number of Iterations')
fig.update_yaxes(title = 'Accuracy')
fig.show()

fig.write_image('/home/srasool/Documents/Machine_learning/Assignment_2/images/accuracy_nn_ro.png')



fig = go.Figure()
# Add traces
fig.add_trace(go.Scatter(x=list(range(iter)), y=f1_score_test_gd,
                    mode='lines',
                    name='gradient_descent'))

fig.add_trace(go.Scatter(x=list(range(iter)), y=f1_score_test_rhc,
                    mode='lines',
                    name='random_hill_climb'))

fig.add_trace(go.Scatter(x=list(range(iter)), y=f1_score_test_sa,
                    mode='lines',
                    name='simulated_annealing'))

fig.add_trace(go.Scatter(x=list(range(iter)), y=f1_score_test_gen_a,
                    mode='lines',
                    name='genetic_alg'))


fig.update_layout(title='F1 Score - Testing')
fig.update_xaxes(title = 'Number of Iterations')
fig.update_yaxes(title = 'F1 Score')
fig.show()

fig.write_image('/home/srasool/Documents/Machine_learning/Assignment_2/images/F1_nn_ro.png')




print('completed')




