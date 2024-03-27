import pandas as pd
from Algorithms import *
from sklearn.model_selection import train_test_split
import math
from plot_code import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
from sklearn.model_selection import ParameterGrid
import numpy as np
from sklearn.metrics import calinski_harabasz_score
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV

# Getting data

# dataset 1
breast_cancer_dataset = pd.read_csv('Assignment_3/Data/breast-cancer.csv') # breast cancer
breast_cancer_dataset['diagnosis'] = (breast_cancer_dataset['diagnosis'] =='M').astype(int)
breast_cancer_dataset_y = breast_cancer_dataset['diagnosis']
breast_cancer_dataset_x = breast_cancer_dataset.drop('diagnosis', axis=1)
breast_cancer_dataset_x = breast_cancer_dataset_x.drop('id', axis=1)
scaler = StandardScaler()
scaled_cancer_dataset_x = scaler.fit_transform(breast_cancer_dataset_x)

# Split the dataset into train and test sets
#X_train_Bc, X_test_Bc, y_train_Bc, y_test_Bc = train_test_split(breast_cancer_dataset_x, breast_cancer_dataset_y, test_size=0.3, random_state=42, shuffle = True)

#dataset 2
heart_dataset_test = pd.read_csv('Assignment_3/Data/heart_test.csv') # heart test
heart_dataset_train = pd.read_csv('Assignment_3/Data/heart_train.csv') # heart train
heart_dataset = pd.concat([heart_dataset_train, heart_dataset_test], axis=0)
heart_dataset_y = heart_dataset[['target']]
heart_dataset_x = heart_dataset.drop('target', axis=1)
scaler = StandardScaler()
scaled_heart_dataset_x = scaler.fit_transform(heart_dataset_x)
#scaled_heart_dataset_x = scaler.fit_transform(heart_dataset)



# Split the dataset into train and test sets
X_train_Hd, X_test_Hd, y_train_Hd, y_test_Hd = train_test_split(scaled_heart_dataset_x, heart_dataset_y, test_size=0.3, random_state=42)

# clusting EM Guassian

number_cluster = list(range(2,30))

#num_gaussian = 2
#awayfrommean = {}
score_list= []
aic_list = []
bic_list = []
silhouette_list = []

score_list_h= []
aic_list_h = []
bic_list_h = []
silhouette_list_h = []

score_list_t= []
aic_list_t = []
bic_list_t = []
silhouette_list_t = []

# from sklearn.mixture import GaussianMixture
# from sklearn.model_selection import GridSearchCV

# # Define parameter grid
# param_grid = {
#     'n_components': list(range(2,30)),  # Try different numbers of components
#     'covariance_type': ['full', 'tied', 'diag', 'spherical'],  # Try different covariance types
#     'init_params': ['kmeans', 'random'],  # Try different initialization methods
#     'tol': [1e-2, 1e-3, 1e-4, 1e-5],  # Try different convergence thresholds
#     'reg_covar': [0.001,0.01, 0.1, 1.0]  # Try different regularization strengths
# }

# # Create Gaussian Mixture Model
# gmm = GaussianMixture()

# # Grid search with cross-validation
# grid_search = GridSearchCV(gmm, param_grid, cv=5)
# grid_search.fit(scaled_heart_dataset_x)  # X is your data

# # Best parameters and best score
# print("Best Parameters:", grid_search.best_params_)
# print("Best Score:", grid_search.best_score_)

def SelBest(arr, n):
    # Sort the array in descending order
    arr_sorted = sorted(arr, reverse=True)
    # Select the top n elements
    return arr_sorted[:n]


iterations = 30
tot=1e-4

for clust in number_cluster:
    temp_sil_score = []
    temp_h_score = []
    temp_aic_h_score = []
    temp_bic_h_score = []
    for i in range(iterations):
        print(i)
        #bc_em, bc_em_labels, probs, score, aic, bic= em(data=scaled_cancer_dataset_x, n_components=clust)
        #silhouette_avg_bc = silhouette_score(scaled_cancer_dataset_x, bc_em_labels)
        h_em, h_em_labels, probs_h, score_h, aic_h, bic_h= em(X_train_Hd,tot, n_components=clust)
        silhouette_avg_H = silhouette_score(X_train_Hd, h_em_labels)

        temp_h_score.append(score_h)
        temp_aic_h_score.append(aic_h)
        temp_bic_h_score.append(bic_h)
        temp_sil_score.append(silhouette_avg_H)
    val_sel_h = np.mean(SelBest(np.array(temp_sil_score), int(iterations/6)))
    val_score_h = np.mean(SelBest(np.array(temp_h_score), int(iterations/6)))
    val_aic_h = np.mean(SelBest(np.array(temp_aic_h_score), int(iterations/6)))
    val_bic_h = np.mean(SelBest(np.array(temp_bic_h_score), int(iterations/6)))

    score_list_h.append(val_score_h)
    aic_list_h.append(val_aic_h)
    bic_list_h.append(val_bic_h)
    silhouette_list_h.append(val_sel_h)


score_plot(data_list=[score_list, score_list_h],
           y_data_list=[number_cluster, number_cluster],
           title= 'EM Gaussian - log_likelihood vs Number of Clusters',
           x_title= 'Number of Clusters',
           y_title= 'log_likelihood' ,
           labels_list=['Breast Cancer Dataset', 'Heart Dataset'])

score_plot(data_list=[aic_list, aic_list_h],
           y_data_list=[number_cluster, number_cluster],
           title= 'EM Gaussian - AIC vs Number of Clusters',
           x_title= 'Number of Clusters',
           y_title= 'AIC Values' ,
           labels_list=['Breast Cancer Dataset', 'Heart_Dataset'])

score_plot(data_list=[bic_list, bic_list_h],
           y_data_list=[number_cluster,number_cluster],
           title= 'EM Gaussian - BIC vs Number of Clusters',
           x_title= 'Number of Clusters',
           y_title= 'BIC Values' ,
           labels_list=['Breast Cancer Dataset', 'Heart_Dataset'])

score_plot(data_list=[silhouette_list, silhouette_list_h],
           y_data_list=[number_cluster,number_cluster],
           title= 'EM Gaussian - Silhouette core vs Number of Clusters',
           x_title= 'Number of Clusters',
           y_title= 'silhouette score' ,
           labels_list=['Breast Cancer Dataset', 'Heart_Dataset'])

score_plot(data_list=[np.gradient(score_list_h)],
           y_data_list=[number_cluster],
           title= 'EM Gaussian - log_likelihood Gradient vs Number of Clusters',
           x_title= 'Number of Clusters',
           y_title= 'Gradient' ,
           labels_list=['Heart_Dataset'])

score_plot(data_list=[np.gradient(aic_list_h)],
           y_data_list=[number_cluster],
           title= 'EM Gaussian - Aic Gradient vs Number of Clusters',
           x_title= 'Number of Clusters',
           y_title= 'Gradient' ,
           labels_list=['Heart_Dataset'])

score_plot(data_list=[np.gradient(bic_list_h)],
           y_data_list=[number_cluster],
           title= 'EM Gaussian - Bic Gradient vs Number of Clusters',
           x_title= 'Number of Clusters',
           y_title= 'Gradient' ,
           labels_list=['Heart_Dataset'])

score_plot(data_list=[np.gradient(silhouette_list_h)],
           y_data_list=[number_cluster],
           title= 'EM Gaussian - Silhouette Gradient vs Number of Clusters',
           x_title= 'Number of Clusters',
           y_title= 'Gradient' ,
           labels_list=['Heart_Dataset'])


print('completed')

#K-means - Heart Dataset
#OPTIMIZATION

data=X_train_Hd


# Define parameters grid
# param_grid = {
#     'n_clusters': list(range(2,30)),  # Range of K values
#     'init': ['k-means++', 'random'],  # Initialization methods
#     'max_iter':[100,1000,10000],  # Maximum number of iterations
#     'tol': [1e-3,1e-4, 1e-5, 1e-6],  # Tolerance for convergence
#     'algorithm': ['lloyd', 'full', 'elkan']  # Algorithm
# }

# # Define a function to compute silhouette score
# def compute_silhouette_score(estimator,data):
#     labels = estimator.fit_predict(data)
#     return silhouette_score(data, labels)

# # Perform grid search using silhouette score as the evaluation metric
# grid_search = GridSearchCV(estimator=KMeans(), param_grid=param_grid, cv=5)
# grid_search.fit(data)

# # Print best parameters and best silhouette score
# print("Best Parameters:", grid_search.best_params_)
# print("Best Silhouette Score:", grid_search.best_score_)


# print('completed')
#data = heart_dataset
k= list(range(2,30))
init= 'k-means++'
max_iter = 100000
tol = 0.0001
algo= 'full'
iterations = 20
wcss_values_avg = []
silhout_value_avg = []
for clusters in k:
    temp_wcss = []
    shil_temp_h = []
    for i in range(iterations):
        kmeans_cluster, centroids, labels = kmeans_algo(data, clusters, init, max_iter, tol, algo)
        temp_wcss.append(kmeans_cluster.inertia_)
        shil_val = silhouette_score(data, labels=labels)
        shil_temp_h.append(shil_val)
    val_wcss_h = np.mean(SelBest(np.array(temp_wcss), int(iterations/5)))
    val_silhout_h = np.mean(SelBest(np.array(shil_temp_h), int(iterations/5)))
    wcss_values_avg.append(val_wcss_h)
    silhout_value_avg.append(val_silhout_h)

score_plot(data_list=[wcss_values_avg],
           y_data_list=[k],
           title= 'K-Mean- Inertia vs Number of Clusters',
           x_title= 'Number of Clusters',
           y_title= 'Inertia' ,
           labels_list=['Heart_Dataset'])

score_plot(data_list=[np.gradient(wcss_values_avg)],
           y_data_list=[k],
           title= 'K-Mean- Inertia Gradient vs Number of Clusters',
           x_title= 'Number of Clusters',
           y_title= 'Gradient' ,
           labels_list=['Heart_Dataset'])

score_plot(data_list=[silhout_value_avg],
           y_data_list=[k],
           title= 'K-Mean- Silhouette Score vs Number of Clusters',
           x_title= 'Number of Clusters',
           y_title= 'Silhouette Score' ,
           labels_list=['Heart_Dataset'])

score_plot(data_list=[np.gradient(silhout_value_avg)],
           y_data_list=[k],
           title= 'K-Mean- Silhouette Score Gradient vs Number of Clusters',
           x_title= 'Number of Clusters',
           y_title= 'Gradient' ,
           labels_list=['Heart_Dataset']) 

print('completed')

import matplotlib.pyplot as plt

X = X_train_Hd

for i in range(len(heart_dataset.columns)-1):
    kmeans = KMeans(n_clusters = 2, init = 'k-means++', n_init=30, max_iter=100000, tol=0.0001, algorithm='full')
    y_kmeans = kmeans.fit_predict(X)
    print(y_kmeans)


    # Create trace for data points with cluster assignments
    trace_data = go.Scatter(
        x=np.array(X)[:, i],
        y=np.array(X)[:, 12],
        mode='markers',
        marker=dict(
            color=kmeans.labels_,
            colorscale='viridis',
            size=10,
            line=dict(color='black', width=1)
        ),
        name='Data Points'
    )

    # Create trace for centroids
    trace_centroids = go.Scatter(
        x=kmeans.cluster_centers_[:, i],
        y=kmeans.cluster_centers_[:, 13],
        mode='markers',
        marker=dict(
            symbol='x',
            size=14,
            color='red'
        ),
        name='Centroids'
    )

    # Create layout
    layout = go.Layout(
        xaxis=dict(title=heart_dataset.columns[i]),
        yaxis=dict(title='Target'),
        title='K-means Clustering with Centroids',
        showlegend=True,
        legend=dict(x=0.85, y=1)
    )

    # Create figure
    fig = go.Figure(data=[trace_data, trace_centroids], layout=layout)

    fig.show()