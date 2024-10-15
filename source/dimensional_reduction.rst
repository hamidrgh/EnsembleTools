Dimensionality Reduction Analysis
*********************************

This package also includes an analysis of dimensionality reduction, employing various methods: t-SNE, PCA, KernelPCA, and UMAP. 
Each of these methods offers a different perspective on reducing data dimensions and is used to visualize and better understand the complex structures of proteins, as well as to compare the effectiveness of these methods.

- **t-SNE (t-Distributed Stochastic Neighbor Embedding)**: t-SNE is a tool to visualize high-dimensional data. It converts similarities between data points to joint probabilities and tries to minimize the Kullback-Leibler divergence between the joint probabilities of the low-dimensional embedding and the high-dimensional data. t-SNE has a cost function that is not convex, i.e. with different initializations we can get different results.

- **PCA (Principal Component Analysis)**: Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space.

- **KernelPCA (Kernel Principal component analysis)**: Non-linear dimensionality reduction through the use of kernels and specifically useful for studying angularity (Periodicity) in data

- **UMAP (Uniform Manifold Approximation and Projection)**: (UMAP) is a dimension reduction technique that can be used for visualisation similarly to t-SNE, but also for general non-linear dimension reduction.

Initially, feature extraction is crucial because it captures essential aspects of protein structure that would otherwise be difficult to analyze directly in high-dimensional data. In IDPET, two groups of features can be analyzed: **distance-based** features and **angular** features.

- Distance-based features in IDPET include the pairwise RMSD matrix between conformations within an ensemble and the Cα-Cα distance matrices.
- For angular features, IDPET offers to analyze Phi (Φ) and Psi (Ψ) angles, t-Rosetta-style angles (omega and phi), and alpha angles.

In the next two parts of this demo we will see how we can use the dimensionality reduction modules of the IDPET: 

.. raw:: html

    <script type="text/javascript">
     function redirectToPage(url) {
      window.location.href = url;
     }
    </script>

    <style>
   .icons-container {
    display: flex;
    justify-content: space-between; /* Ordina verticalmente i contenitori delle icone */
    flex-wrap: center; /* Centra i contenitori delle icone horizontalmente */
    padding: 20px; /* Aggiunge uno spazio tra i contenitori delle icone */
   }

   .icon-item {
    text-align: center; /* Centra il testo all'interno del contenitore */
    margin-bottom: 20px;
   }

   .icon-item img {
   cursor: pointer; /* Cambia il cursore in un puntatore quando si passa sopra le icone */
   width: 200px; /* Imposta la larghezza delle icone */
   height: 200px;
   object-fit: cover; /* Mantieni la proporzione delle icone */
   }

   .icon-item div {
   font-size: 12px; /* Imposta la dimensione del testo */
   color: #001; /* Imposta il colore del testo */
   }
   </style>

   <div class='icons-container'>


   <div class='icon-item'>
    <img src="_static/images/icons/icon_dm.png" alt="dr_cadist" title="dimensional reduction on carbon alpha distances" onclick="redirectToPage('method_overview.html')" style="cursor: pointer; width: 200px; heigh: 200px">
    <div style="font-size: 12px; color: #001;">Dimensionality reduction methods' overview</div>
   </div>

    <div class='icon-item'>
     <img src="_static/images/icons/dr_phipsi.png" alt="dr_phipsi" title="dimensional reduction phi psi angles" onclick="redirectToPage('dr_phipsi.html')" style="cursor: pointer; width: 200px; height: 200px;">
     <div style="font-size: 12px; color: #001;">Dimensionality Reduction using angular features</div>
    </div>

   <div class='icon-item'>
    <img src="_static/images/icons/dr_ca.png" alt="dr_cadist" title="dimensional reduction on carbon alpha distances" onclick="redirectToPage('dr_cadist.html')" style="cursor: pointer; width: 200px; heigh: 200px">
    <div style="font-size: 12px; color: #001;">Dimensional Reduction using distance-based features</div>
   </div>



   
.. toctree::
   :hidden:

   dr_phipsi
   method_overview
   dr_cadist

   