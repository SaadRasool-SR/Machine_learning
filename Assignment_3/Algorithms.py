from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
from sklearn import random_projection
from sklearn.metrics import mean_squared_error
from sklearn.manifold import TSNE



# clustering Algorithms, 1) Expectation Maximization, 2)

def em(data, tot, n_components,covariance_type='full', max_iter=10000, init_params='kmeans', verbose_interval=10):
    em_algo = GaussianMixture(n_components=n_components,
                            covariance_type=covariance_type,
                            max_iter=max_iter,
                            tol= tot,
                            n_init=15,
                            reg_covar=0.001,
                            init_params=init_params,
                            verbose_interval=verbose_interval).fit(data)
    
    em_labels = em_algo.predict(data)
    probs = em_algo.predict_proba(data)
    score = em_algo.score(data)
    aic = em_algo.aic(data)
    bic = em_algo.bic(data)

    return em_algo, em_labels, probs, score, aic, bic


def kmeans_algo(data, k, init, max_iter, tol, algo):
    kmeans_cluster = KMeans(n_clusters=k, init=init, n_init=30, max_iter=max_iter, tol=tol, algorithm=algo).fit(data)
    centroids = kmeans_cluster.cluster_centers_
    labels = kmeans_cluster.labels_
    return kmeans_cluster, centroids, labels



def PCA_algo(data,n):
    pca_fit = PCA(n_components=n).fit(data)
    X_pca = pca_fit.transform(data)
    return X_pca, pca_fit



def ICA_algo(data, n):
    ica = FastICA(n_components=n,random_state=42)
    ica_fit_transform = ica.fit_transform(data)
    component_kurtosis = kurtosis(ica_fit_transform, axis=0)
    return ica,ica_fit_transform, component_kurtosis


def RP(data,n_components):
    rp = random_projection.SparseRandomProjection(n_components=n_components)
    X_rp = rp.fit_transform(data)
    X_approx = rp.inverse_transform(X_rp)
    mse = mean_squared_error(data, X_approx)

    return rp, mse, X_rp

def tsne_algo(data, n, perplexity):
    tsne = TSNE(n_components=n, perplexity=perplexity, init='random', learning_rate='auto')
    x_tsne = tsne.fit_transform(data)
    return x_tsne

