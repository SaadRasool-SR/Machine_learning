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
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr



# # Getting data

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


#PCA
n = list(range(1,14))

pca_transform, pca_fit = PCA_algo(X_train_Hd, 13)
eigenvalues = pca_fit.explained_variance_
prct_variation = pca_fit.explained_variance_ratio_ * 100

mse_error_ls = []
for i in n:
    pca_transform, pca_fit = PCA_algo(X_train_Hd, i)
    x_reconstructed = pca_fit.inverse_transform(pca_transform)
    reconstructed_error = mean_squared_error(X_train_Hd,x_reconstructed)
    mse_error_ls.append(reconstructed_error)

cumvalues = [sum(prct_variation[:i+1]) for i in range(len(prct_variation))]

new_score_plot(x_data_list=[n], 
               y_data_list=[eigenvalues], 
               title='PCA - Eigenvalues vs PCA Components (Heart Disease Dataset)', 
               x_title='PCA Components', 
               y_title='Eigenvalues', 
               labels_list=['Heart Dataset'])

# new_score_plot(x_data_list=[n], 
#                y_data_list=[mse_error_ls], 
#                title='MSE vs PCA Components - Reconstruction Heart Dataset', 
#                x_title='PCA Components', 
#                y_title='Mean Squared Error', 
#                labels_list=['Heart Dataset'])

bar_score_plot(x=n,
               y=prct_variation,
               title='Percentages of Variation For Each PCA (Heart Disease Dataset)',
               x_title='PCA - Components',
               y_title='% Variation',
               y_cum=cumvalues)

print('completed')

X_train_Hd, X_test_Hd, y_train_Hd, y_test_Hd = train_test_split(scaled_heart_dataset_x, heart_dataset_y, test_size=0.3, random_state=42)
components = list(range(1,14))
#ICA
#ica, ica_algo_results, kurtosis_results = ICA_algo(data=X_train_Hd,n=14)
ica_mse_error = []
kurtosis_results_values = []
for i in components:
    ica, ica_algo_results, kurtosis_results = ICA_algo(data=X_train_Hd,n=i)
    avg_kurtosis = np.mean(kurtosis_results)
    kurtosis_results_values.append(avg_kurtosis)
    mixing_matrix = np.linalg.pinv(ica.components_)
    X_reconstruct = np.dot(ica_algo_results, mixing_matrix.T)
    reconstruction_error = mean_squared_error(X_train_Hd, X_reconstruct)
    ica_mse_error.append(reconstruction_error)

print('completed')

# new_score_plot(x_data_list=[n,n], 
#                y_data_list=[mse_error_ls, ica_mse_error], 
#                title='PCA and ICA MSE vs Number of Componenets- Reconstruction Heart Dataset', 
#                x_title='Number of Components', 
#                y_title='Mean Squared Error', 
#                labels_list=['PCA MSE', 'ICA MSE'])


new_score_plot(x_data_list=[n], 
               y_data_list=[kurtosis_results_values], 
               title='ICA - Kurtosis vs PCA Components (Heart Disease Dataset)', 
               x_title='ICA Components', 
               y_title=' Avg Kurtosis', 
               labels_list=['Heart Dataset'])


print('completed')

components = 14
ica, ica_algo_results, kurtosis_results = ICA_algo(data=X_train_Hd,n=components)
mixing_matrix = ica.components_
kurtosis_val_abs_sort = np.argsort(abs(kurtosis_results))
abs_kurtosis = abs(kurtosis_results)


bar_plot(x=kurtosis_val_abs_sort,
         y=abs_kurtosis,
         title='ICA - Abs Kurtosis vs Features (Heart Disease Dataset)',
         x_title='Features_Index',
         y_title='Abs Kurtosis')

print('completed')

#Random Projection
n=list(range(1,14))
mse_vals = []
explained_variance_ratio_ls =[]
var_original = np.var(X_train_Hd)
for i in n:
    rp, mse, X_rp = RP(data=X_train_Hd,n_components=i)
    mse_vals.append(mse)
    var_projected = np.var(X_rp)
    explained_variance_ratio = var_projected / var_original
    explained_variance_ratio_ls.append(explained_variance_ratio)

cumvalues = [sum(explained_variance_ratio_ls[:i+1]) for i in range(len(explained_variance_ratio_ls))]
print('done')

new_score_plot(x_data_list=[n,n,n], 
               y_data_list=[mse_error_ls, ica_mse_error,mse_vals ], 
               title='PCA, ICA and RP MSE vs Number of Componenets- Reconstruction (Heart Disease Dataset)', 
               x_title='Number of Components', 
               y_title='Mean Squared Error', 
               labels_list=['PCA MSE', 'ICA MSE', 'RP MSE'])


bar_score_plot(x=n,
               y=explained_variance_ratio_ls,
               title='Percentages of Variation For Each RP (Heart Disease Dataset)',
               x_title='RP - Components',
               y_title='% Variation',
               y_cum=cumvalues)

# Manifold Algorithm
n = 14
#for i in range(1,15):

tsne_results = tsne_algo(data=X_train_Hd, n=2, perplexity=15)

new_score_plot_2(tsne_results, np.ravel(y_train_Hd))

print('done')



##################### Dataset 1
# Getting data

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
X_train_Bc, X_test_Bc, y_train_Bc, y_test_Bc = train_test_split(scaled_cancer_dataset_x, breast_cancer_dataset_y, test_size=0.3, random_state=42, shuffle = True)



#PCA
n = list(range(1,30))

pca_transform, pca_fit = PCA_algo(X_train_Bc, 30)
eigenvalues = pca_fit.explained_variance_
prct_variation = pca_fit.explained_variance_ratio_ * 100

mse_error_ls = []
for i in n:
    pca_transform, pca_fit = PCA_algo(X_train_Bc, i)
    x_reconstructed = pca_fit.inverse_transform(pca_transform)
    reconstructed_error = mean_squared_error(X_train_Bc,x_reconstructed)
    mse_error_ls.append(reconstructed_error)

cumvalues = [sum(prct_variation[:i+1]) for i in range(len(prct_variation))]

new_score_plot(x_data_list=[n], 
               y_data_list=[eigenvalues], 
               title='PCA - Eigenvalues vs PCA Components (Breast Cancer Dataset)', 
               x_title='PCA Components', 
               y_title='Eigenvalues', 
               labels_list=['Heart Dataset'])

# new_score_plot(x_data_list=[n], 
#                y_data_list=[mse_error_ls], 
#                title='MSE vs PCA Components - Reconstruction Heart Dataset', 
#                x_title='PCA Components', 
#                y_title='Mean Squared Error', 
#                labels_list=['Heart Dataset'])

bar_score_plot(x=n,
               y=prct_variation,
               title='Percentages of Variation For Each PCA (Breast Cancer Dataset)',
               x_title='PCA - Components',
               y_title='% Variation',
               y_cum=cumvalues)

print('completed')

X_train_Bc, X_test_Bc, y_train_Bc, y_test_Bc = train_test_split(scaled_cancer_dataset_x, breast_cancer_dataset_y, test_size=0.3, random_state=42, shuffle = True)
components = list(range(1,30))
#ICA
#ica, ica_algo_results, kurtosis_results = ICA_algo(data=X_train_Hd,n=14)
ica_mse_error = []
kurtosis_results_values = []
for i in components:
    ica, ica_algo_results, kurtosis_results = ICA_algo(data=X_train_Bc,n=i)
    avg_kurtosis = np.mean(kurtosis_results)
    kurtosis_results_values.append(avg_kurtosis)
    mixing_matrix = np.linalg.pinv(ica.components_)
    X_reconstruct = np.dot(ica_algo_results, mixing_matrix.T)
    reconstruction_error = mean_squared_error(X_train_Bc, X_reconstruct)
    ica_mse_error.append(reconstruction_error)

print('completed')

# new_score_plot(x_data_list=[n,n], 
#                y_data_list=[mse_error_ls, ica_mse_error], 
#                title='PCA and ICA MSE vs Number of Componenets- Reconstruction Heart Dataset', 
#                x_title='Number of Components', 
#                y_title='Mean Squared Error', 
#                labels_list=['PCA MSE', 'ICA MSE'])


new_score_plot(x_data_list=[n], 
               y_data_list=[kurtosis_results_values], 
               title='ICA - Kurtosis vs PCA Components (Breast Cancer Dataset)', 
               x_title='ICA Components', 
               y_title=' Avg Kurtosis', 
               labels_list=['Heart Dataset'])


print('completed')

components = 30
ica, ica_algo_results, kurtosis_results = ICA_algo(data=X_train_Bc,n=components)
mixing_matrix = ica.components_
kurtosis_val_abs_sort = np.argsort(abs(kurtosis_results))
abs_kurtosis = abs(kurtosis_results)


bar_plot(x=kurtosis_val_abs_sort,
         y=abs_kurtosis,
         title='ICA - Abs Kurtosis vs Features (Breast Cancer Dataset)',
         x_title='Features_Index',
         y_title='Abs Kurtosis')

print('completed')

#Random Projection
n=list(range(1,30))
mse_vals = []
explained_variance_ratio_ls =[]
var_original = np.var(X_train_Bc)
for i in n:
    rp, mse, X_rp = RP(data=X_train_Bc,n_components=i)
    mse_vals.append(mse)
    var_projected = np.var(X_rp)
    explained_variance_ratio = var_projected / var_original
    explained_variance_ratio_ls.append(explained_variance_ratio)

cumvalues = [sum(explained_variance_ratio_ls[:i+1]) for i in range(len(explained_variance_ratio_ls))]
print('done')

new_score_plot(x_data_list=[n,n,n], 
               y_data_list=[mse_error_ls, ica_mse_error,mse_vals ], 
               title='PCA, ICA and RP MSE vs Number of Componenets- Reconstruction Breast Cancer Dataset', 
               x_title='Number of Components', 
               y_title='Mean Squared Error', 
               labels_list=['PCA MSE', 'ICA MSE', 'RP MSE'])


bar_score_plot(x=n,
               y=explained_variance_ratio_ls,
               title='Percentages of Variation For Each RP (Breast Cancer Dataset)',
               x_title='RP - Components',
               y_title='% Variation',
               y_cum=cumvalues)

# Manifold Algorithm
n = 14
#for i in range(1,15):

tsne_results = tsne_algo(data=X_train_Bc, n=2, perplexity=15)

new_score_plot_3(tsne_results, np.ravel(y_train_Bc))

print('done')
