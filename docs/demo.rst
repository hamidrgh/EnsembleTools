.. functions analysis documentation master file, created by
   sphinx-quickstart on Tue May 21 12:57:48 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Demo
==============================================

The present software has been designed to facilitate the study and analysis of intrinsically disordered proteins (IDPs), a critical area of research within molecular biology. These proteins, known for their dynamic and highly variable structure, play a fundamental role in various biological processes and serve as the basis for many diseases.

To address the complexity of IDP structures, the software implements advanced functions for generating graphs that illustrate structures at both the global and local levels. This approach allows for the exploration of both the general configuration of IDPs and interactions and structures at smaller scales, thus providing a comprehensive and detailed view of their structures.

Furthermore, to effectively manage the large volumes of data associated with these proteins, the software integrates dimensionality reduction analysis. These methods are essential for reducing data size without compromising the contained information, thereby making the analysis more efficient and manageable. Through the application of advanced techniques for dimensionality reduction, the software enables focusing on significant patterns and characteristics of IDP structures, thereby facilitating interpretation and analysis of the results.





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
    margin-bottom:20px;
   }

   .icon-item img {
   cursor: pointer; /* Cambia il cursore in un puntatore quando si passa sopra le icone */
   width: 200px; /* Imposta la larghezza delle icone */
   height: 200px; /* Mantieni la proporzione delle icone */
   }

   .icon-item div {
   font-size: 12px; /* Imposta la dimensione del testo */
   color: #001; /* Imposta il colore del testo */
   }
   </style>


   <div class='icons-container'>
    <div class='icon-item'>
     <img src="images/icons/gl_an.png" alt="global analysis" title="The global analysis of a protein's structure focuses on studying its total three-dimensional shape, aiming to understand how proteins fold and group together to form stable and functional structures." onclick="redirectToPage('Global_analysis.html')" style="cursor: pointer;width: 200px; height: 200px;">
     <div style="font-size: 12px; color: #001;">Global analysis</div>
    </div>

   <div class='icon-item'>
    <img src="images/icons/l_an.png" alt="local analysis" title="The local analysis of a protein's structure focuses on examining specific regions, concentrating on secondary structural elements, with the aim of understanding folding patterns and interactions within those localized areas." onclick="redirectToPage('local_analysis.html')" style="cursor: pointer;width: 200px; height: 200px;">
    <div style="font-size: 12px; color: #001;">Local analysis</div>
   </div>

   <div class='icon-item'>
    <img src="images/icons/tsne.png" alt="dimensional_reduction" title="Dimensionality reduction analysis focuses on simplifying complex data by reducing the number of independent variables needed to describe a system." onclick="redirectToPage('dimensional_reduction.html')" style="cursor: pointer;width: 200px; height: 200px;">
    <div style="font-size: 12px; color: #001;">Dimensional Reduction analysis</div>
   </div>






.. toctree::
   :hidden:

   Global_analysis

   local_analysis

   dimensional_reduction


