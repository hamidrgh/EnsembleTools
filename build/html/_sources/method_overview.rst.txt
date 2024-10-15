Methods' overview 
******************************************

t-SNE (t-distributed Stochastic Neighbor Embedding.)
-------

t-SNE is a technique that reduces the dimensionality of data while preserving relationships between data points. It minimizes the Kullback-Leibler divergence between the joint probabilities of the high-dimensional data and its low-dimensional embedding.
The process begins by randomly projecting data points into a low-dimensional space. These points are gradually adjusted, forming distinct clusters. At each step, points are attracted to nearby points and repelled from distant ones based on similarity measures.

Steps involved in t-SNE:

1- **Calculate Similarities:**
The similarity between points is calculated by measuring the distance between them in the high-dimensional space. A normal distribution is centered on each point to measure the "unscaled similarity."

2- **Normalize Similarities:**
The unscaled similarities are normalized so that they sum to 1. t-SNE averages the similarity scores for each pair of points, forming a similarity matrix.

3- **Low-Dimensional Projection:**
The points are projected into a low-dimensional space, and t-distribution is used to calculate similarity scores in this space.

4- **Iterative Optimization:**
t-SNE iteratively moves the points in the low-dimensional space to minimize the difference between the high-dimensional and low-dimensional similarity matrices.


.. code-block:: python

    analysis.reduce_features(method='tsne', perplexity_vals = [1, 20, 5], circular=True, range_n_clusters=range(2,10,1))

.. admonition:: perplexity_vals

   Perplexity in t-SNE controls the number of nearest neighbors considered when calculating joint probabilities. Adjusting perplexity affects cluster structure:

   **Higher Perplexity:** Considers more neighbors, resulting in larger, more spread-out clusters and a more global view of the data. Distant points may be grouped together.
  
   **Lower Perplexity:** Focuses on local structures, forming smaller, denser clusters. However, too low of a value can overemphasize local patterns, while too high can blur cluster boundaries.
   


.. admonition:: metric
   
   str, optional-Metric to use. Default is "euclidean". 
   
   This parameter specifies the distance measure to be used for calculating the similarities between points in the original dataset. 
   The choice of metric can influence the result of the embedding as it determines how the distances between points are evaluated.

.. admonition:: circular 
   
   bool, optional-Whether to use circular metrics. Default is False. 
   
   This parameter, if present, could indicate whether distance calculation should consider a circular or cyclic structure. 
   For example, if working with data that have a periodic nature (such as angles ranging from 0 to 360 degrees), using a circular metric can be useful for accurately capturing the relationships between points.

.. admonition:: n_components  
   
   int, optional-Number of dimensions of the embedded space. Default is 2.
   
   This parameter specifies the number of dimensions in which one wants to reduce the data.

.. admonition:: learning_rate 
   
   float, optional-Learning rate. Default is 'auto'.

   The learning rate, typically between 10.0 and 1000.0, controls how quickly an embedding is modified in each iteration. By adjusting it, one can regulate the convergence speed and the quality of the final embedding. A higher learning rate accelerates optimization and may lead to overly rapid adaptations that overlook certain data structures, making the visualization less stable. 
   Conversely, a too low value risks slowing down the algorithm's convergence. The 'auto' option sets the learning_rate to max(N / early_exaggeration / 4, 50) where N is the sample size. See the `documentation <https://scikit-learn.org/1.5/modules/generated/sklearn.manifold.TSNE.html>`_.

.. admonition:: range_n_clusters 
   
   list[int], optional-Range of cluster values. Default is range(2, 10, 1).
   
   This parameter refers to the range of possible numbers of clusters one wishes to consider in the analysis.      


UMAP (Uniform Manifold Approximation & Projection)
------
UMAP (Uniform Manifold Approximation and Projection) is a non-linear dimensionality reduction algorithm that preserves the topological structure of high-dimensional data. It calculates similarity scores to identify clusters and maintains these relationships in the low-dimensional space. By adjusting the `n_neighbors` parameter, UMAP balances local and global structure.

.. code-block:: python

    analysis.reduce_features(method='umap',  n_neighbors = [10, 20,  50, 100], circular=True, range_n_clusters=range(2,10,1))



UMAP has several hyperparameters that can significantly impact the resulting embedding:

.. admonition:: n_neighbors

   int, number of  nearest neighbors.[Default is 15].

   This parameter controls how UMAP balances the local and global structure in the data. 
   It does this by limiting the size of the local neighborhood UMAP will consider when learning the structure of the data manifold. 
   Lower values of *n_neighbors* will force UMAP to focus on very local structures (potentially at the expense of the overall view), while higher values will make UMAP consider larger neighborhoods around each point when estimating the data manifold structure, losing structural details to gain a broader view of the data.




.. admonition:: min_dist

   float, optional-Minimum distance. [Default is 0.1].

   This parameter provides the minimum distance that points can have in the low-dimensional representation. 
   This means that lower values of *min_dist* will lead to tighter clustering, potentially resulting in a loss of overall data vision. 
   In this case, even small variations in the data can become overly emphasized. 

   Conversely, higher values of *min_dist* will prevent UMAP from identifying distinct clusters, instead focusing on the overall structure. 
   This can lead to a loss of important details in the local relationships between points, resulting in a representation that, while preserving the general topology, lacks precision in detail.

.. admonition:: num_dim

   int, optional-Number of components.[Default is 2]

   This parameter determines the dimensionality of the reduced space in which we will embed the data.

.. admonition:: metric

   str, optional-Metric to use.[Default is "euclidean"].
   

   The "metric" parameter in UMAP controls how distances are calculated in the input data space, and naturally, the choice of metric depends on the specific characteristics of the data and the analytical objectives. 

   [For a comprehensive documentation of the possible metrics, you can consult the following: `link <https://umap-learn.readthedocs.io/en/latest/parameters.html#metric>`_]

   - Euclidean: This metric is based on the formula of the square root of the sum of the squares of the differences between the coordinates of the points. *It is commonly used when working with data that can be represented in a Euclidean space.*

   - Canberra: This metric calculates the distance as the sum of the absolute differences between the coordinates of the points divided by the sum of the coordinates of the points themselves. *It is suitable for data with very different value ranges and is effective in capturing covariance between variables. This makes it useful for analyzing multivariate data.*

   - Mahalanobis: A generalization of the Euclidean distance, this metric takes into account the covariance between variables. *It is particularly useful when working with multivariate data and when it is desired to consider the correlation between variables.*

   - Cosine: This metric measures the angle between two vectors, rather than their magnitude. *It is suitable for situations where the direction of the vectors is more important than their length.*


.. admonition:: range_n_clusters 
   
   list[int], optional-Range of cluster values. Default is range(2, 10, 1).
   
   This parameter refers to the range of possible numbers of clusters one wishes to consider in the analysis.  


PCA (Principal Component Analysis)
---------------------------------------

PCA is a dimensionality reduction technique based on the decomposition of the eigenvectors of the covariance matrix of high-dimensional data, aiming to identify a set of components that capture the maximum variance present in the data.

To achieve this, PCA projects the original data onto a new coordinate system defined by these principal components. The first principal component corresponds to the direction with the highest variance in the data, the second principal component to the direction with the second-highest variance, and so forth.

.. code-block:: python

    analysis.reduce_features(method='pca', num_dim=3)

.. admonition:: num_dim

   int, optional-Number of components to keep. [Default is 10]

   As the sole parameter, *"num_dim"* is optional and indicates the number of components to retain in the transformed dataset. 
   A too high value of the "num_dim" parameter could result in retaining too many principal components, leading to a less significant reduction in dimensionality and potentially preserving noise or irrelevant information. 
   Conversely, a too low value might excessively reduce the dimensionality of the data, causing the loss of important information.


KernelPCA (Kernel-Principal Component Analysis)
---------------------------------------

Kernel Principal Component Analysis (KPCA) using the scikit-learn library to study various properties of conformational ensembles. KPCA is particularly effective for analyzing non-linear features, such as angular data with periodic properties. To handle angular data appropriately, we transform each angle into its sine and cosine components, capturing the inherent circular nature of the data. Using these transformed values, we construct the kernel matrix by calculating pairwise distances between the data points with a custom kernel function. This procedure allows us to preserve the periodic nature of the angles and effectively analyze the data in a lower-dimensional space, providing insights into the conformational properties of the ensembles.

.. code-block:: python

    analysis.reduce_features(method='kpca', num_dim=3)

.. admonition:: num_dim

    Number of components to keep. Default is 10.

.. admonition:: gamma

   Kernel coefficient. Default is None. If gamma is None, then it is set to 1/n_features

   This parameter defines the influence of individual data points in the kernel function. Specifically, it acts as a scaling factor for the distances between data points when using kernels such as the Radial Basis Function (RBF) or Gaussian kernel. A low value of gamma means that distant points have more influence, leading to smoother decision boundaries, while a high value of gamma makes the influence more localized, resulting in more complex boundaries. The default value is typically `1/n_features`, where `n_features` is the number of features in the dataset, but this can be adjusted based on the specific characteristics of the data.
   

.. admonition:: circular

   Indicates whether to use circular metrics. This should be set to True when analyzing angular data. The default value is False.
