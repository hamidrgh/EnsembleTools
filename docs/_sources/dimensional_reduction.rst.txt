Dimensional Reduction Analysis
*********************************

This package also includes an analysis of dimensionality reduction, employing various methods: t-SNE, PCA, DimenFix, and UMAP. 
Each of these methods offers a different perspective on reducing data dimensions and is used to visualize and better understand the complex structures of proteins, as well as to compare the effectiveness of these methods.

- **t-SNE (t-Distributed Stochastic Neighbor Embedding)**: This method preserves neighborhood relationships among points, allowing for the visualization of high-dimensional data in a two-dimensional or three-dimensional space.

- **PCA (Principal Component Analysis)**: Used to reduce data dimensions by identifying the principal components that explain most of the variance in the data, PCA is a linear method that facilitates the interpretation of the main directions of variation.

- **Dimenfix**: A method specifically designed for protein analysis, combining various dimensionality reduction techniques to improve the representation of protein conformations.

- **UMAP (Uniform Manifold Approximation and Projection)**: This method aims to preserve the global and local structure of data during dimensionality reduction, proving particularly effective for clustering and visualization applications.

However, initially, feature extraction is crucial because it allows capturing essential aspects of protein structure, which would otherwise be challenging to analyze directly in high-dimensional data. 
The following features were extracted:

- **Alpha Carbons (Cα)**: Alpha carbons are the central carbon atoms of each amino acid in the polypeptide chain of a protein. They are important because they represent the backbone of the protein structure, and their position determines the protein's three-dimensional conformation.

- **Phi (Φ) and Psi (Ψ) Angles**: These angles are the dihedral angles describing rotation around bonds between atoms in a protein. The Phi (Φ) angle describes rotation around the bond between the amide nitrogen and the alpha carbon, while the Psi (Ψ) angle describes rotation around the bond between the alpha carbon and the carbonyl carbon. Phi and Psi angles are crucial for determining the protein's secondary structure, such as alpha helices and beta sheets, and significantly influence its overall three-dimensional conformation.

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
     <img src="images/icons/dr_phipsi.png" alt="dr_phipsi" title="dimensional reduction phi psi angles" onclick="redirectToPage('dr_phipsi.html')" style="cursor: pointer; width: 200px; height: 200px;">
     <div style="font-size: 12px; color: #001;">Dimensional Reduction Phi Psi Angles</div>
    </div>

   <div class='icon-item'>
    <img src="images/icons/dr_ca.png" alt="dr_cadist" title="dimensional reduction on carbon alpha distances" onclick="redirectToPage('dr_cadist.html')" style="cursor: pointer; width: 200px; heigh: 200px">
    <div style="font-size: 12px; color: #001;">Dimensional Reduction Carbon Alpha Distances</div>
   </div>
   
.. toctree::
   :hidden:

   dr_phipsi

   dr_cadist
