"""
Dimensionality reduction methods.
"""

import os
from typing import Union, List
from collections import OrderedDict
import tempfile
import numpy as np
import sklearn.decomposition
from sklearn.manifold import TSNE
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from dpet.logger import stream as st


#-------------------------------------------
# Main class for dimensionality reduction. -
#-------------------------------------------

class DimensionalityReduction:
    """
    Class for performing dimensionality reduction.
    """

    keyerr_msg = "Unknown dimensionality reduction method: {}"

    def __init__(self, method: str):
        if not method in ("pca", "kpca", "tsne"):
            self.raise_keyerr(method)

        self.method = method
        self.model = None
        self.clst = None
        self.clst_name = None

    
    def raise_keyerr(self, method):
        raise KeyError(self.keyerr_msg.format(method))


    def fit(
            self,
            features: np.ndarray,
            *args, **params):
        """
        TODO.
        """

        if self.method == "pca":
            return self.fit_pca(features, *args, **params)
        if self.method == "kpca":
            return self.fit_kpca(features, *args, **params)
        elif self.method == "tsne":
            st.write("tsne is your selected dimensionality reduction method!")
            return self.fit_transform_tsne(features, *args, **params)
        # elif method == "dimenfix":
        #     st.write("dimenfix is your selected dimensionality reduction method!")
        #     return fit_transform_dimenfix(features, *args, **params)
        # elif method == "mds":
        #     return fit_transform_mds(features, *args, **params)
        # elif method == "tsne-circular":
        #     return fit_transform_tsne_circular(features, *args, **params)
        else:
            raise KeyError(f"Unknown dimensionality reduction method: {self.method}")


    def fit_pca(
            self,
            features: np.ndarray,
            num_dim: int = 10) -> sklearn.decomposition.PCA:
        """
        TODO.
        description

        arguments.

        output.
        """

        pca = sklearn.decomposition.PCA(n_components=num_dim)
        pca.fit(features)
        self.model = pca
        return pca
    
    def fit_kpca(
            self,
            features: np.ndarray,
            num_dim: int = 10,
            kernel: str = "rbf",
            gamma: float = None,
            circular: bool = False
        ) -> sklearn.decomposition.KernelPCA:
        """
        TODO.
        description

        arguments.

        output.
        """
        # Use angular features and a custom similarity function.
        if circular:
            gamma = 1/features.shape[1] if gamma is None else gamma
            kernel = lambda a1, a2: unit_vector_kernel(a1, a2, gamma=gamma)
            pca_in = features
        # Use raw features.
        else:
            pca_in = features

        pca = sklearn.decomposition.KernelPCA(
            n_components=num_dim,
            kernel=kernel,
            gamma=gamma  # Ignored if using circular.
        )
        pca.fit(pca_in)
        self.model = pca
        return pca


    def fit_transform_tsne(
            self,
            features: np.ndarray,
            n_components: int = 2,
            learning_rate: float = 100.0,
            perplexity_vals: Union[float, List[float]] = range(2, 10, 2),
            range_n_clusters: List[int] = range(2, 10, 1),
            metric: str = "euclidean",
            circular: bool = False,
            # out_dir: str = "."
            ) -> dict:
        """
        Performs tSNE via sklearn.

        `features`: numpy array storing a (N, num_features) data matrix.
        `n_components`: number of output components.
        `learning_rate`: see the sklearn argument for TSNE.
        `perplexity_vals`: perplexity value(s) for tSNE. See the sklearn
            argument for more information. Can be a float or a list of floats.
            If it is a float, tSNE will be performed using only that perplexity
            value. If is is a list of floats, tSNE will performed with all
            values and a validation strategy using Silhouette score will be
            performed to choose a tSNE model and features with optimal results.
        `range_n_clusters`: number of clusters to use in KMeans clustering when
            searching for the optimal perplexity value. Takes effect only when
            using multiple perplexity values.
        `metric`: distance metric used for tSNE. See the sklearn argument for
            TSNE.
        `circular`: if True, a custom distance metric for angular features will
            be used (the 'metric' argument will be ignored). Should only be used
            input features storing angular values.

        output: TODO.
        """

        st.write("tsne is running...")

        if circular:
            metric = unit_vector_distance

        # Try multiple perplexity values and select the one with the best
        # silhouette score.
        if hasattr(perplexity_vals, "__iter__"):
            
            # Perform tSNE with different perplexity values and cluster numbers.
            results = []
            _perplexity_vals = []
            _clusters_vals = []

            # Store results in a temporary directory.
            temp_dir = tempfile.TemporaryDirectory()
            temp_dirname = temp_dir.name

            # Iter through perplexity values.
            for i, perp_i in enumerate(perplexity_vals):

                # Fit tSNE.
                tsneObject = TSNE(
                    n_components=n_components,
                    perplexity=perp_i,
                    early_exaggeration=10.0,
                    learning_rate=learning_rate,
                    n_iter=3500,
                    metric=metric,
                    n_iter_without_progress=300,
                    min_grad_norm=1e-7,
                    init="random",
                    method="barnes_hut",
                    angle=0.5
                )
                tsne_data = tsneObject.fit_transform(features)
                
                # Save a temporary copy of the tSNE features.
                np.save(
                    os.path.join(temp_dirname, f"tsnep_{i}.npy"),
                    tsne_data)
                st.write(f"tsne file for the perplexity value of {perp_i} is saved in {temp_dirname} ")

                # Cluster with KMeans, using different number of clusters.
                for j, n_clusters_j in enumerate(range_n_clusters):

                    # Run KMeans.
                    kmeans = KMeans(n_clusters=n_clusters_j, n_init='auto').fit(tsne_data)

                    # Save cluster centers and labels.
                    np.save(
                        os.path.join(temp_dirname, f'kmeans_{j}_clusters_centers_tsnep_{i}.npy'),
                        kmeans.cluster_centers_)
                    np.save(
                        os.path.join(temp_dirname, f'kmeans_{j}_clusters_tsnep_{i}.npy'),
                        kmeans.labels_)
                    
                    # Compute silhouette score based on low-dim and high-dim distances.
                    silhouette_ld = silhouette_score(tsne_data, kmeans.labels_)
                    silhouette_hd = metrics.silhouette_score(features, kmeans.labels_)
                    # print("silhouette_ld:", silhouette_ld)
                    # print("silhouette_hd:", silhouette_hd)

                    # Store results.
                    # with open(reduce_dim_params['tsne']['dir']  + '/silhouette.txt', 'a') as f:
                    #     f.write("\n")
                    #     print(perp, n_clusters, silhouette_ld, silhouette_hd, silhouette_ld*silhouette_hd, file =f)
                    results.append(
                        [i, j, silhouette_ld, silhouette_hd, silhouette_ld*silhouette_hd]
                    )
                    if i == 0:
                        _clusters_vals.append(n_clusters_j)

                _perplexity_vals.append(perp_i)

            results = np.array(results)

            st.write(f"tsne is done! All files saved in {temp_dirname}")

            # Select the best combination of perplexity and n_clusters
            # according to silhouette_ld*silhouette_hd.
            best_idx = np.argmax(results[:,4])
            
            # Get the indices of the selected perplexity and n_clusters values.
            bestP_idx = int(results[best_idx, 0])
            bestK_idx = int(results[best_idx, 1])
            # Get the selected perplexity and n_clusters values.
            bestP = _perplexity_vals[bestP_idx]
            bestK = _clusters_vals[bestK_idx]

            print("results:\n",results)
            print("best idx:", best_idx)
            print("perplexities:", _perplexity_vals)
            print("selected perplexity:", bestP)
            print("num_clusters", _clusters_vals)
            print("selected num_clusters:", bestK)

            # Get tSNE features.
            besttsne = np.load(os.path.join(temp_dirname, f'tsnep_{bestP_idx}.npy'))
            # Get the KMeans clustering object.
            best_kmeans = KMeans(n_clusters=int(bestK), n_init='auto').fit(besttsne)

            # Clean up.
            temp_dir.cleanup()

            # Store results.
            return_data = {"data": besttsne, "kmeans": best_kmeans}
        
        # Use a single perplexity value.
        else:

            tsneObject = TSNE(
                n_components=n_components,
                perplexity=perplexity_vals,
                early_exaggeration=10.0,
                learning_rate=learning_rate,
                n_iter=3500,
                metric=metric,
                n_iter_without_progress=300,
                min_grad_norm=1e-7,
                init="random",
                method="barnes_hut",
                angle=0.5
            )
            tsne_data = tsneObject.fit_transform(features)
            st.write(f"tsne is done!")
        
            return_data = {"data": tsne_data}

        return return_data


    def fit_transform_dimenfix(data):
        raise NotImplementedError("Not tested yet")
        from neo_force_scheme import NeoForceScheme

        nfs = NeoForceScheme()
        projection = nfs.fit_transform(data)
        return projection


    def fit_transform_mds(data, nume_dim=2):
        raise NotImplementedError("Not tested yet")
        from sklearn.manifold import MDS

        embedding = MDS(n_components=nume_dim)
        feature_transformed = embedding.fit_transform(data)
        return feature_transformed


    def transform(self, features: np.ndarray):
        if self.method in ("pca", "kpca"):
            return self.model.transform(features)
        else:
            self.raise_keyerr(self.method)

    
    def set_data(self, data: OrderedDict, concat_data: np.ndarray = None):
        self.data = data
        if concat_data is None:
            self.concat_data = np.concatenate(
                [self.data[code_i] for code_i in self.data],
                axis=0
            )
        else:
            self.concat_data = concat_data
        

    # NOTE: the same methods could be applied to the EnsembleAnalysis class.
    #       To implement it, we could use a mixin.
    def set_clusters(self, clst, name: str):
        if not name in ("kmeans", ):
            raise KeyError(name)
        self.clst = clst
        self.clst_name = name
    
    def get_num_clusters(self):
        if self.clst is not None:
            if self.clst_name == "kmeans":
                return self.clst.get_params()["n_clusters"]
            else:
                raise KeyError(clst)
        return None
    
    def get_clusters_labels(self, clst_id=None):
        if self.clst is not None:
            if self.clst_name == "kmeans":
                if clst_id is None:
                    return self.clst.labels_
                else:
                    return np.where(self.clst.labels_ == clst_id)[0]
            else:
                raise KeyError(clst)
        return None
    

#----------------------------------------------------------------------
# Functions for performing dimensionality reduction on circular data. -
#----------------------------------------------------------------------

def unit_vectorize(a: np.ndarray) -> np.ndarray:
    """Convert an array with (*, N) angles in an array with (*, N, 2) sine and
    cosine values for the N angles."""
    v = np.concatenate([np.cos(a)[...,None], np.sin(a)[...,None]], axis=-1)
    return v

def unit_vector_distance(a0: np.ndarray, a1: np.ndarray, sqrt: bool = True):
    """Compute the sum of distances between two (*, N) arrays storing the
    values of N angles."""
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