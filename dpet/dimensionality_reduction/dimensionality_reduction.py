from abc import ABC, abstractmethod
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
        raise NotImplementedError("Method 'fit' must be implemented in subclasses.")

    @abstractmethod
    def transform(self, data:np.ndarray) -> np.ndarray:
        raise NotImplementedError("Method 'transform' must be implemented in subclasses.")
    
    @abstractmethod
    def fit_transform(self, data:np.ndarray) -> np.ndarray:
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
    perplexity_vals : list[float], optional
        List of perplexity values. Default is range(2, 10, 2).
    metric : str, optional
        Metric to use. Default is "euclidean".
    circular : bool, optional
        Whether to use circular metrics. Default is False.
    n_components : int, optional
        Number of dimensions of the embedded space. Default is 2.
    learning_rate : float, optional
        Learning rate. Default is 100.0.
    range_n_clusters : list[int], optional
        Range of cluster values. Default is range(2, 10, 1).
    """

    def __init__(
            self, perplexity_vals:list[float]=range(2, 10, 2), 
            metric:str="euclidean", 
            circular:bool=False, 
            n_components:int=2, 
            learning_rate:float=100.0, 
            range_n_clusters:list[int] = range(2,10,1)):
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
        self.bestP, self.bestK, self.best_tsne, self.best_kmeans = self.get_best_results()
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

    def get_best_results(self) -> tuple:
        """
        Get the best combination of perplexity and number of clusters based on silhouette scores.

        Returns
        -------
        tuple
            A tuple containing the best perplexity, the best number of clusters, the corresponding
            t-SNE features, and the KMeans clustering model for the best combination.
        """
        best_result = max(self.results, key=lambda x: x['silhouette_product'])
        best_perplexity = best_result['perplexity']
        best_n_clusters = best_result['n_clusters']
        best_tsne = best_result['tsne_features']
        best_kmeans = best_result['kmeans_model']
        print("Best Perplexity:", best_perplexity)
        print("Best Number of Clusters:", best_n_clusters)
        return best_perplexity, best_n_clusters, best_tsne, best_kmeans

class DimenFixReduction(DimensionalityReduction):
    """
    Class for performing dimensionality reduction using DimenFix algorithm.

    Parameters
    ----------
    range_n_clusters : list[int], optional
        Range of cluster values. Default is range(1, 10, 1).
    """

    def __init__(self, range_n_clusters:list[int] = range(1,10,1)):
        self.range_n_clusters = range_n_clusters

    def fit(self, data:np.ndarray):
        return super().fit(data)
    
    def transform(self, data:np.ndarray) -> np.ndarray:
        return super().transform(data)
    
    def fit_transform(self, data:np.ndarray) -> np.ndarray:
        nfs = NeoForceScheme()
        self.projection = nfs.fit_transform(data)
        self.cluster()
        return self.projection
    
    def cluster(self) -> list:
        """
        Perform clustering using KMeans algorithm for each number of clusters in the specified range.

        Returns
        -------
        list
            A list of tuples containing the number of clusters and the corresponding silhouette score
            for each clustering result.
        """
        self.sil_scores = []
        for n_clusters in self.range_n_clusters:
            clusterer = KMeans(n_clusters=n_clusters, n_init="auto", random_state=10)
            cluster_labels = clusterer.fit_predict(self.projection)
            silhouette_avg = silhouette_score(self.projection, cluster_labels)
            self.sil_scores.append((n_clusters,silhouette_avg))
            print(
                "For n_clusters =",
                n_clusters,
                "The average silhouette_score is :",
                silhouette_avg,
            )
        return self.sil_scores

class MDSReduction(DimensionalityReduction):
    """
    Class for performing dimensionality reduction using Multidimensional Scaling (MDS) algorithm.

    Parameters
    ----------
    num_dim : int, optional
        Number of dimensions for the reduced space. Default is 2.
    """

    def __init__(self, num_dim:int=2):
        self.num_dim = num_dim

    def fit(self, data:np.ndarray):
        return super().fit(data)
    
    def transform(self, data:np.ndarray) -> np.ndarray:
        return super().transform(data)
    
    def fit_transform(self, data:np.ndarray) -> np.ndarray:    
        embedding = MDS(n_components=self.num_dim)
        feature_transformed = embedding.fit_transform(data)
        return feature_transformed
    
class UMAPReduction(DimensionalityReduction):
    """
    Class for performing dimensionality reduction using Uniform Manifold Approximation and Projection (UMAP) algorithm.

    Parameters
    ----------
    num_dim : int, optional
        Number of dimensions for the reduced space. Default is 2.
    n_neighbors : int, optional
        Number of neighbors to consider for each point in the input data. Default is 10.
    min_dist : float, optional
        The minimum distance between embedded points. Default is 0.1.
    metric : str, optional
        The metric to use for distance calculation. Default is 'euclidean'.
    range_n_clusters : range or list, optional
        Range of cluster values to consider for silhouette scoring. Default is range(2, 10, 1).
    """

    def __init__(self, num_dim=2, n_neighbors=10, min_dist =0.1 , metric='euclidean', range_n_clusters = range(2,10,1)):
        self.num_dim = num_dim
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.range_n_clusters = range_n_clusters
        self.sil_scores = []
    
    def fit(self, data):
        return super().fit(data)
    
    def transform(self, data) -> np.ndarray:
        return super().transform(data)

    def fit_transform(self, data) -> np.ndarray:
        umap_model = UMAP(n_neighbors=self.n_neighbors, min_dist=self.min_dist, n_components=self.num_dim, metric=self.metric)
        self.embedding = umap_model.fit_transform(data)
        self.cluster()
        return self.embedding
    
    def cluster(self) -> list:
        """
        Perform clustering using KMeans algorithm for each number of clusters in the specified range.

        Returns
        -------
        list
            A list of tuples containing the number of clusters and the corresponding silhouette score
            for each clustering result.
        """
        for n_clusters in self.range_n_clusters:
            clusterer = KMeans(n_clusters=n_clusters, n_init="auto", random_state=10)
            cluster_labels = clusterer.fit_predict(self.embedding)
            silhouette_avg = silhouette_score(self.embedding, cluster_labels)
            self.sil_scores.append((n_clusters,silhouette_avg))
            print(
                "For n_clusters =",
                n_clusters,
                "The average silhouette_score is :",
                silhouette_avg,
            )
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
        elif method == "dimenfix":
            return DimenFixReduction(*args, **kwargs)
        elif method == "mds":
            return MDSReduction(*args, **kwargs)
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
    dist = unit_vector_distance(a1, a2, sqrt=False)
    sim = np.exp(-gamma*dist)
    return sim