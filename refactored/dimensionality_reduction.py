from abc import ABC, abstractmethod
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sklearn
import numpy as np
from sklearn.manifold import MDS
from neo_force_scheme import NeoForceScheme
from tsne_utils import unit_vector_distance

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
    def __init__(self, num_dim):
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

class TSNEReduction(DimensionalityReduction):
    def __init__(self, dir=".", perplexityVals=range(2, 10, 2), metric="euclidean"):
        self.perplexityVals = perplexityVals
        self.metric = metric
        self.dir = dir

    def fit(self, data):
        return super().fit(data)
    
    def transform(self, data):
        return super().transform(data)

    def fit_transform(self, data):
        print("tsne is running...")
        for i in self.perplexityVals:
            tsne_file = os.path.join(self.dir, f"tsnep{i}")
            if os.path.exists(tsne_file):
                print(f"Dimensionality reduction for perplexity {i} already performed. Skipping.")
                continue
            tsneObject = TSNE(
                n_components=2,
                perplexity=i,
                early_exaggeration=10.0,
                learning_rate=100.0,
                n_iter=3500,
                metric=self.metric,
                n_iter_without_progress=300,
                min_grad_norm=1e-7,
                init="random",
                method="barnes_hut",
                angle=0.5,
            )
            tsne = tsneObject.fit_transform(data)
            os.makedirs(self.dir, exist_ok=True)
            np.savetxt(tsne_file, tsne)
            print(f"tsne file for the perplexity value of {i} is saved in {self.dir} ")
        print(f"tsne is done! All files saved in {self.dir}")

class DimenFixReduction(DimensionalityReduction):
    def fit(self, data):
        return super().fit(data)
    
    def transform(self, data):
        return super().transform(data)
    
    def fit_transform(self, data):
        nfs = NeoForceScheme()
        projection = nfs.fit_transform(data)
        return projection

class TSNECircularReduction(DimensionalityReduction):
    def __init__(self,  dir=".", perplexityVals=range(2, 10, 2), metric=unit_vector_distance):
        self.perplexityVals = perplexityVals
        self.metric = metric
        self.dir = dir

    def fit(self, data):
        return super().fit(data)
    
    def transform(self, data):
        return super().transform(data)

    def fit_transform(self, data):
        print("tsne for phi_psi is running...")
        for i in self.perplexityVals:
            tsne_file = os.path.join(self.dir, f"tsnep{i}")
            if os.path.exists(tsne_file):
                print(f"Dimensionality reduction for perplexity {i} already performed. Skipping.")
                continue
            tsneObject = TSNE(
                n_components=2,
                perplexity=i,
                early_exaggeration=10.0,
                learning_rate=100.0,
                n_iter=3500,
                metric=self.metric,
                n_iter_without_progress=300,
                min_grad_norm=1e-7,
                init="random",
                method="barnes_hut",
                angle=0.5,
            )
            tsne = tsneObject.fit_transform(data)
            os.makedirs(self.dir, exist_ok=True)
            np.savetxt(tsne_file, tsne)
            print(f"tsne file for the perplexity value of {i} is saved in {self.dir} ")
        print(f"tsne is done! All files saved in {self.dir}")

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
    def get_reducer(method, dir, *args, **kwargs):
        if method == "pca":
            return PCAReduction(*args, **kwargs)
        elif method == "tsne":
            return TSNEReduction(dir, *args, **kwargs)
        elif method == "dimenfix":
            return DimenFixReduction(*args, **kwargs)
        elif method == "tsne-circular":
            return TSNECircularReduction(dir, *args, **kwargs)
        elif method == "mds":
            return MDSReduction(*args, **kwargs)
        else:
            raise NotImplementedError("Unsupported dimensionality reduction method.")
