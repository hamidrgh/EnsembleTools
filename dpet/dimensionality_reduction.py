from abc import ABC, abstractmethod
from typing import List, Tuple
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
import numpy as np
from sklearn.manifold import MDS
from neo_force_scheme import NeoForceScheme
from sklearn.metrics import silhouette_score
from umap import UMAP

class DimensionalityReduction(ABC):
    @abstractmethod
    def fit(self, data:np.ndarray):
        """
        Fit the dimensionality reduction model to the data.

        Parameters
        ----------
        data : np.ndarray
            The input data array of shape (n_samples, n_features).

        Notes
        -----
        This method fits the dimensionality reduction model to the input data.
        """
        raise NotImplementedError("Method 'fit' must be implemented in subclasses.")

    @abstractmethod
    def transform(self, data:np.ndarray) -> np.ndarray:
        """
        Transform the input data using the fitted dimensionality reduction model.

        Parameters
        ----------
        data : np.ndarray
            The input data array of shape (n_samples, n_features) to be transformed.

        Returns
        -------
        np.ndarray
            The transformed data array of shape (n_samples, n_components).

        Notes
        -----
        This method transforms the input data using the fitted dimensionality reduction model.
        """
        raise NotImplementedError("Method 'transform' must be implemented in subclasses.")
    
    @abstractmethod
    def fit_transform(self, data:np.ndarray) -> np.ndarray:
        """
        Fit the dimensionality reduction model to the data and then transform it.

        Parameters
        ----------
        data : np.ndarray
            The input data array of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            The transformed data array of shape (n_samples, n_components).

        Notes
        -----
        This method fits the dimensionality reduction model to the input data and then transforms it.
        """
        raise NotImplementedError("Method 'fit_transform' must be implemented in subclasses.")

class PCAReduction(DimensionalityReduction):
    """
    Principal Component Analysis (PCA) for dimensionality reduction.

    Parameters
    ----------
    num_dim : int, optional
        Number of components to keep. Default is 10.
    """

    def __init__(self, num_dim: int = 10):
        self.num_dim = num_dim

    def fit(self, data:np.ndarray):
        self.pca = PCA(n_components=self.num_dim)
        self.pca.fit(data)
        return self.pca
    
    def transform(self, data:np.ndarray) -> np.ndarray:
        reduce_dim_data = self.pca.transform(data)
        return reduce_dim_data
    
    def fit_transform(self, data:np.ndarray) -> np.ndarray:
        self.pca = PCA(n_components=self.num_dim)
        transformed = self.pca.fit_transform(data)
        return transformed

class TSNEReduction(DimensionalityReduction):
    """
    Class for performing dimensionality reduction using t-SNE algorithm.

    Parameters
    ----------
    perplexity_vals : List[float], optional
        List of perplexity values. Default is [10, 20, 50, 100].
    metric : str, optional
        Metric to use. Default is "euclidean".
    circular : bool, optional
        Whether to use circular metrics. Default is False.
    n_components : int, optional
        Number of dimensions of the embedded space. Default is 2.
    learning_rate : float, optional
        Learning rate. Default is 100.0.
    range_n_clusters : List[int], optional
        Range of cluster values. Default is range(2, 10, 1).
    """

    def __init__(
            self, perplexity_vals:List[float]=[10, 20, 50, 100], 
            metric:str="euclidean", 
            circular:bool=False, 
            n_components:int=2, 
            learning_rate:float=100.0, 
            range_n_clusters:List[int] = range(2,10,1)):
        self.perplexity_vals = perplexity_vals
        self.metric = unit_vector_distance if circular else metric
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.results = []
        self.range_n_clusters = range_n_clusters

    def fit(self, data:np.ndarray):
        return super().fit(data)
    
    def transform(self, data:np.ndarray) -> np.ndarray:
        return super().transform(data)

    def fit_transform(self, data:np.ndarray) -> np.ndarray:
        self.data = data
        print("tsne is running...")
        for perplexity in self.perplexity_vals:
            tsneObject = TSNE(
                n_components=self.n_components,
                perplexity=perplexity,
                early_exaggeration=10.0,
                learning_rate=self.learning_rate,
                n_iter=3500,
                metric=self.metric,
                n_iter_without_progress=300,
                min_grad_norm=1e-7,
                init="random",
                method="barnes_hut",
                angle=0.5,
            )
            tsne = tsneObject.fit_transform(data)
            self._cluster(tsne, perplexity)
        best_result = max(self.results, key=lambda x: x['silhouette_product'])
        self.bestP = best_result['perplexity']
        self.bestK = best_result['n_clusters']
        self.best_tsne = best_result['tsne_features']
        self.best_kmeans = best_result['kmeans_model']
        print("Best Perplexity:", self.bestP)
        print("Best Number of Clusters:", self.bestK)
        print("Silhouette Score Low Dimensional:", best_result['silhouette_ld'])
        print("Silhouette Score High Dimensional:", best_result['silhouette_hd'])
        print("Silhouette Score Product", best_result['silhouette_product'])
        return self.best_tsne

    def _cluster(self, tsne:np.ndarray, perplexity:float):
        for n_clusters in self.range_n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, n_init='auto').fit(tsne)
            silhouette_ld = silhouette_score(tsne, kmeans.labels_)
            silhouette_hd = silhouette_score(self.data, kmeans.labels_)
            result = {
                'perplexity': perplexity,
                'n_clusters': n_clusters,
                'silhouette_ld': silhouette_ld,
                'silhouette_hd': silhouette_hd,
                'silhouette_product': silhouette_ld * silhouette_hd,
                'tsne_features': tsne,
                'kmeans_model': kmeans
            }
            self.results.append(result)

class UMAPReduction(DimensionalityReduction):
    """
    Class for performing dimensionality reduction using Uniform Manifold Approximation and Projection (UMAP) algorithm.

    Parameters
    ----------
    num_dim : int, optional
        Number of dimensions for the reduced space. Default is 2.
    n_neighbors : List[int], optional
        Number of neighbors to consider for each point in the input data. Default is [20, 30, 40].
    min_dist : float, optional
        The minimum distance between embedded points. Default is 0.1.
    metric : str, optional
        The metric to use for distance calculation. Default is 'euclidean'.
    range_n_clusters : range or List, optional
        Range of cluster values to consider for silhouette scoring. Default is range(2, 10, 1).
    """

    def __init__(self, num_dim:int=2, n_neighbors:List[int]=[20, 30, 40], circular= False,min_dist:float=0.1 , metric:str='euclidean', range_n_clusters:List[int] = range(2,10,1)):
        self.num_dim = num_dim
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = unit_vector_distance if circular else metric
        self.range_n_clusters = range_n_clusters
        self.sil_scores = []
        self.best_embedding = None
        self.best_score = -1
        self.best_n_clusters = None
        self.best_n_neighbors = None
    
    def fit(self, data):
        return super().fit(data)
    
    def transform(self, data) -> np.ndarray:
        return super().transform(data)

    def fit_transform(self, data) -> np.ndarray:
        for n_neighbor in self.n_neighbors:
            umap_model = UMAP(n_neighbors=n_neighbor, min_dist=self.min_dist, n_components=self.num_dim, metric=self.metric)
            embedding = umap_model.fit_transform(data)
            scores = self.cluster(embedding, n_neighbor)
            best_score_for_n_neighbor = max(scores, key=lambda x: x[2])

            if best_score_for_n_neighbor[2] > self.best_score:
                self.best_score = best_score_for_n_neighbor[2]
                self.best_embedding = embedding
                self.best_n_clusters = best_score_for_n_neighbor[1]
                self.best_n_neighbors = n_neighbor
        print('UMAP is running...')
        print(f'Best number of neighbors: {self.best_n_neighbors}')
        print(f'Best number of clusters : {self.best_n_clusters}')
        
        return self.best_embedding
    
    def cluster(self, embedding, n_neighbor) -> List[Tuple]:
        """
        Perform clustering using KMeans algorithm for each number of clusters in the specified range.

        Returns
        -------
        List[Tuple]
            A list of tuples containing the number of clusters and the corresponding silhouette score
            for each clustering result.
        """
        
        for n_clusters in self.range_n_clusters:
            clusterer = KMeans(n_clusters=n_clusters, n_init="auto", random_state=10)
            cluster_labels = clusterer.fit_predict(embedding)
            silhouette_avg = silhouette_score(embedding, cluster_labels)
            self.sil_scores.append((n_neighbor, n_clusters,silhouette_avg))
            # print(
            #      "For n_neighbors =",
            #     n_neighbor,
            #     "For n_clusters =",
            #     n_clusters,
            #     "The average silhouette_score is :",
            #     silhouette_avg,
            # )
        return self.sil_scores
    
class KPCAReduction(DimensionalityReduction):
    """
    Class for performing dimensionality reduction using Kernel PCA (KPCA) algorithm.

    Parameters
    ----------
    circular : bool, optional
        Whether to use circular metrics for angular features. Default is False.
    num_dim : int, optional
        Number of dimensions for the reduced space. Default is 10.
    gamma : float, optional
        Kernel coefficient. Default is None.
    """

    def __init__(self, circular:bool=False, num_dim:int=10, gamma:float=None) -> None:
        self.circular = circular
        self.num_dim = num_dim
        self.gamma = gamma

    def fit(self, data):
        # Use angular features and a custom similarity function.
        if self.circular:
            self.gamma = 1/data.shape[1] if self.gamma is None else self.gamma
            kernel = lambda a1, a2: unit_vector_kernel(a1, a2, gamma=self.gamma)
            pca_in = data
        # Use raw features.
        else:
            pca_in = data

        self.pca = KernelPCA(
            n_components=self.num_dim,
            kernel=kernel,
            gamma=self.gamma  # Ignored if using circular.
        )
        self.pca.fit(pca_in)
        return self.pca
    
    def transform(self, data) -> np.ndarray:
        reduce_dim_data = self.pca.transform(data)
        return reduce_dim_data
    
    def fit_transform(self, data) -> np.ndarray:
        return super().fit_transform(data)

class DimensionalityReductionFactory:
    """
    Factory class for creating instances of various dimensionality reduction algorithms.

    Methods
    -------
    get_reducer(method, \*args, \*\*kwargs)
        Get an instance of the specified dimensionality reduction algorithm.
    """

    @staticmethod
    def get_reducer(method, *args, **kwargs) -> DimensionalityReduction:
        """
        Get an instance of the specified dimensionality reduction algorithm.

        Parameters
        ----------
        method : str
            Name of the dimensionality reduction method.
        \*args
            Positional arguments to pass to the constructor of the selected method.
        \*\*kwargs
            Keyword arguments to pass to the constructor of the selected method.

        Returns
        -------
        DimensionalityReduction
            Instance of the specified dimensionality reduction algorithm.
        """
        if method == "pca":
            return PCAReduction(*args, **kwargs)
        elif method == "tsne":
            return TSNEReduction(*args, **kwargs)
        elif method == "kpca":
            return KPCAReduction(*args, **kwargs)
        elif method == "umap":
            return UMAPReduction(*args, **kwargs)
        else:
            raise NotImplementedError("Unsupported dimensionality reduction method.")

#----------------------------------------------------------------------
# Functions for performing dimensionality reduction on circular data. -
#----------------------------------------------------------------------

def unit_vectorize(a: np.ndarray) -> np.ndarray:
    """
    Convert an array with (\*, N) angles in an array with (\*, N, 2) sine and
    cosine values for the N angles.
    """
    v = np.concatenate([np.cos(a)[...,None], np.sin(a)[...,None]], axis=-1)
    return v

def unit_vector_distance(a0: np.ndarray, a1: np.ndarray, sqrt: bool = True):
    """
    Compute the sum of distances between two (\*, N) arrays storing the
    values of N angles.
    """
    v0 = unit_vectorize(a0)
    v1 = unit_vectorize(a1)
    # Distance between N pairs of angles.
    if sqrt:
        dist = np.sqrt(np.square(v0 - v1).sum(axis=-1))
    else:
        dist = np.square(v0 - v1).sum(axis=-1)
    # We sum over the N angles.
    dist = dist.sum(axis=-1)
    return dist

def unit_vector_kernel(a1, a2, gamma):
    """
    Compute unit vector kernel.
    """
    dist = unit_vector_distance(a1, a2, sqrt=False)
    sim = np.exp(-gamma*dist)
    return sim