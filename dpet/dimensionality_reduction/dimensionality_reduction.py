from abc import ABC, abstractmethod
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sklearn
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
    
    @abstractmethod
    def cluster(self, range_n_clusters):
        pass

class PCAReduction(DimensionalityReduction):
    def __init__(self, num_dim=10):
        self.num_dim = num_dim

    def fit(self, data):
        self.pca = sklearn.decomposition.PCA(n_components=self.num_dim)
        self.pca.fit(data)
        return self.pca
    
    def transform(self, data):
        reduce_dim_data = self.pca.transform(data)
        return reduce_dim_data
    
    def fit_transform(self, data):
        self.pca = sklearn.decomposition.PCA(n_components=self.num_dim)
        transformed = self.pca.fit_transform(data)
        return transformed
    
    def cluster(self, range_n_clusters):
        return super().cluster(range_n_clusters)

class TSNEReduction(DimensionalityReduction):
    def __init__(self, dir:str=".", perplexityVals=range(2, 10, 2), metric:str="euclidean", circular:bool=False, n_components:int=2, learning_rate:float=100.0):
        self.perplexityVals = perplexityVals
        if circular:
            self.metric = unit_vector_distance
        else:
            self.metric = metric
        self.dir = dir
        self.n_components=n_components
        self.learning_rate = learning_rate

    def fit(self, data):
        return super().fit(data)
    
    def transform(self, data):
        return super().transform(data)

    def fit_transform(self, data):
        self.data = data
        print("tsne is running...")
        for i in self.perplexityVals:
            tsne_file = os.path.join(self.dir, f"tsnep{i}")
            tsneObject = TSNE(
                n_components=self.n_components,
                perplexity=i,
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
            np.savetxt(tsne_file, tsne)
            print(f"tsne file for the perplexity value of {i} is saved in {self.dir} ")
        print(f"tsne is done! All files saved in {self.dir}")

    def cluster(self, range_n_clusters):
        # Clear the silhouette file before appending new data
        silhouette_file_path = os.path.join(self.dir, 'silhouette.txt')
        if os.path.exists(silhouette_file_path):
            with open(silhouette_file_path, 'w') as f:
                f.write("")
        
        for perp in self.perplexityVals:
            tsne = np.loadtxt(self.dir + '/tsnep'+str(perp))
            sil_scores = []
            for n_clusters in range_n_clusters:
                kmeans = KMeans(n_clusters=n_clusters, n_init= 'auto').fit(tsne)
                np.savetxt(self.dir + '/kmeans_'+str(n_clusters)+'clusters_centers_tsnep'+str(perp), kmeans.cluster_centers_, fmt='%1.3f')
                np.savetxt(self.dir + '/kmeans_'+str(n_clusters)+'clusters_tsnep'+str(perp)+'.dat', kmeans.labels_, fmt='%1.1d')
                    
                # Compute silhouette score based on low-dim and high-dim distances        
                silhouette_ld = silhouette_score(tsne, kmeans.labels_)
                silhouette_hd = silhouette_score(self.data, kmeans.labels_)
                
                # Append silhouette scores to the file
                with open(silhouette_file_path, 'a') as f:
                    f.write(f"{perp} {n_clusters} {silhouette_ld} {silhouette_hd} {silhouette_ld * silhouette_hd}\n")
                
                sil_scores.append((n_clusters, silhouette_ld))
            return sil_scores

class DimenFixReduction(DimensionalityReduction):
    def fit(self, data):
        return super().fit(data)
    
    def transform(self, data):
        return super().transform(data)
    
    def fit_transform(self, data):
        nfs = NeoForceScheme()
        self.projection = nfs.fit_transform(data)
        return self.projection
    
    def cluster(self, range_n_clusters):
        for n_clusters in range_n_clusters:
            sil_scores = []
            clusterer = KMeans(n_clusters=n_clusters, n_init="auto", random_state=10)
            cluster_labels = clusterer.fit_predict(self.projection)
            silhouette_avg = silhouette_score(self.projection, cluster_labels)
            sil_scores.append((n_clusters,silhouette_avg))
            print(
                "For n_clusters =",
                n_clusters,
                "The average silhouette_score is :",
                silhouette_avg,
            )
        return sil_scores

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
    
    def cluster(self, range_n_clusters):
        return super().cluster(range_n_clusters)

class DimensionalityReductionFactory:
    @staticmethod
    def get_reducer(method, dir, *args, **kwargs):
        if method == "pca":
            return PCAReduction(*args, **kwargs)
        elif method == "tsne":
            return TSNEReduction(dir, *args, **kwargs)
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