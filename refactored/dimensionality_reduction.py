from abc import ABC, abstractmethod
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sklearn
import numpy as np
from sklearn.manifold import MDS
from neo_force_scheme import NeoForceScheme
from tsne_utils import unit_vector_distance

class DimensionalityReduction(ABC):
    @abstractmethod
    def fit_transform(self, data):
        pass

class PCAReduction(DimensionalityReduction):
    def __init__(self, num_dim):
        self.num_dim = num_dim

    def fit_transform(self, data):
        pca = sklearn.decomposition.PCA(n_components=self.num_dim)
        pca.fit(data)
        return pca

class TSNEReduction(DimensionalityReduction):
    def __init__(self, perplexityVals=range(2, 10, 2), metric="euclidean", dir="."):
        self.perplexityVals = perplexityVals
        self.metric = metric
        self.dir = dir

    def fit_transform(self, data):
        print("tsne is running...")
        for i in self.perplexityVals:
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
            np.savetxt(self.dir + "/tsnep{0}".format(i), tsne)
            print(f"tsne file for the perplexity value of {i} is saved in {dir} ")
        print(f"tsne is done! All files saved in {self.dir}")

class DimenFixReduction(DimensionalityReduction):
    def fit_transform(self, data):
        nfs = NeoForceScheme()
        projection = nfs.fit_transform(data)
        return projection

class TSNECircularReduction(DimensionalityReduction):
    def __init__(self, perplexityVals=range(2, 10, 2), metric=unit_vector_distance, dir="."):
        self.perplexityVals = perplexityVals
        self.metric = metric
        self.dir = dir

    def fit_transform(self, data):
        print("tsne for phi_psi is running...")
        for i in self.perplexityVals:
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
            np.savetxt(self.dir + "/tsnep{0}".format(i), tsne)
            print(f"tsne file for the perplexity value of {i} is saved in {dir} ")
        print(f"tsne is done! All files saved in {self.dir}")

class MDSReduction(DimensionalityReduction):
    def __init__(self, num_dim):
        self.num_dim = num_dim
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
        elif method == "tsne-circular":
            return TSNECircularReduction(*args, **kwargs)
        elif method == "mds":
            return MDSReduction(*args, **kwargs)
        else:
            raise NotImplementedError("Unsupported dimensionality reduction method.")
