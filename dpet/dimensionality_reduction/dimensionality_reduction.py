from abc import ABC, abstractmethod
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
from sklearn.manifold import MDS
from neo_force_scheme import NeoForceScheme
from sklearn.metrics import silhouette_score

class DimensionalityReduction(ABC):
    @abstractmethod
    def fit(self, data):
        pass

    @abstractmethod
    def transform(self, data):
        pass
    
    @abstractmethod
    def fit_transform(self, data):
        pass

class PCAReduction(DimensionalityReduction):
    def __init__(self, num_dim=10):
        self.num_dim = num_dim

    def fit(self, data):
        self.pca = PCA(n_components=self.num_dim)
        self.pca.fit(data)
        return self.pca
    
    def transform(self, data):
        reduce_dim_data = self.pca.transform(data)
        return reduce_dim_data
    
    def fit_transform(self, data):
        self.pca = PCA(n_components=self.num_dim)
        transformed = self.pca.fit_transform(data)
        return transformed

class TSNEReduction(DimensionalityReduction):
    def __init__(self, perplexity_vals=range(2, 10, 2), metric: str="euclidean", 
                 circular: bool=False, n_components: int=2, learning_rate: float=100.0, range_n_clusters = range(2,10,1)):
        self.perplexity_vals = perplexity_vals
        self.metric = unit_vector_distance if circular else metric
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.results = []
        self.range_n_clusters = range_n_clusters

    def fit(self, data):
        return super().fit(data)
    
    def transform(self, data):
        return super().transform(data)

    def fit_transform(self, data):
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
            self.cluster(tsne, perplexity)
        self.bestP, self.bestK, self.best_tsne, self.best_kmeans = self.get_best_results()
        return self.best_tsne

    def cluster(self, tsne, perplexity):
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

    def get_best_results(self):
        # Select the best combination of perplexity and n_clusters
        # according to silhouette_ld*silhouette_hd.
        best_result = max(self.results, key=lambda x: x['silhouette_product'])
        best_perplexity = best_result['perplexity']
        best_n_clusters = best_result['n_clusters']
        best_tsne = best_result['tsne_features']
        best_kmeans = best_result['kmeans_model']
        print("Best Perplexity:", best_perplexity)
        print("Best Number of Clusters:", best_n_clusters)
        return best_perplexity, best_n_clusters, best_tsne, best_kmeans


class DimenFixReduction(DimensionalityReduction):
    def __init__(self, range_n_clusters = range(2,10,1)) -> None:
        self.range_n_clusters = range_n_clusters

    def fit(self, data):
        return super().fit(data)
    
    def transform(self, data):
        return super().transform(data)
    
    def fit_transform(self, data):
        nfs = NeoForceScheme()
        self.projection = nfs.fit_transform(data)
        self.cluster()
        return self.projection
    
    def cluster(self):
        for n_clusters in self.range_n_clusters:
            self.sil_scores = []
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

class MDSReduction(DimensionalityReduction):
    def __init__(self, num_dim):
        self.num_dim = num_dim

    def fit(self, data):
        return super().fit(data)
    
    def transform(self, data):
        return super().transform(data)
    
    def fit_transform(self, data):    
        embedding = MDS(n_components=self.num_dim)
        feature_transformed = embedding.fit_transform(data)
        return feature_transformed

class DimensionalityReductionFactory:
    @staticmethod
    def get_reducer(method, *args, **kwargs):
        if method == "pca":
            return PCAReduction(*args, **kwargs)
        elif method == "tsne":
            return TSNEReduction(*args, **kwargs)
        elif method == "dimenfix":
            return DimenFixReduction(*args, **kwargs)
        elif method == "mds":
            return MDSReduction(*args, **kwargs)
        else:
            raise NotImplementedError("Unsupported dimensionality reduction method.")

def unit_vectorize(a):
    """Convert an array with (*, N) angles in an array with (*, N, 2) sine and
    cosine values for the N angles."""
    v = np.concatenate([np.cos(a)[..., None], np.sin(a)[..., None]], axis=-1)
    return v

def unit_vector_distance(a0, a1):
    """Compute the sum of distances between two (*, N, 2) arrays storing the
    sine and cosine values of N angles."""
    v0 = unit_vectorize(a0)
    v1 = unit_vectorize(a1)
    # Distance between every pair of N angles.
    dist = np.sqrt(np.square(v0 - v1).sum(axis=-1))
    # We sum over the N angles.
    dist = dist.sum(axis=-1)
    return dist