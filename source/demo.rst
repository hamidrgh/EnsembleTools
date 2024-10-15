.. functions analysis documentation master file, created by
   sphinx-quickstart on Tue May 21 12:57:48 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Demo
==============================================

We designed the IDPET Python package to facilitate the study and analysis of conformational ensembles of intrinsically disordered proteins. These proteins, known for their dynamic and highly variable structures, play a fundamental role in various biological processes.

Here, we highlight four different types of analyses that can be performed using IDPET.





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
     <img src="_static/images/icons/gl_an.png" alt="global analysis" title="The global analysis of a protein's structure focuses on studying its total three-dimensional shape, aiming to understand how proteins fold and group together to form stable and functional structures." onclick="redirectToPage('Global_analysis.html')" style="cursor: pointer;width: 200px; height: 200px;">
     <div style="font-size: 12px; color: #001;">Global analysis</div>
    </div>

   <div class='icon-item'>
    <img src="_static/images/icons/l_an.png" alt="local analysis" title="The local analysis of a protein's structure focuses on examining specific regions, concentrating on secondary structural elements, with the aim of understanding folding patterns and interactions within those localized areas." onclick="redirectToPage('local_analysis.html')" style="cursor: pointer;width: 200px; height: 200px;">
    <div style="font-size: 12px; color: #001;">Local analysis</div>
   </div>

   <div class='icon-item'>
    <img src="_static/images/icons/tsne.png" alt="dimensional_reduction" title="Dimensionality reduction analysis focuses on simplifying complex data by reducing the number of independent variables needed to describe a system." onclick="redirectToPage('dimensional_reduction.html')" style="cursor: pointer;width: 200px; height: 200px;">
    <div style="font-size: 12px; color: #001;">Dimensionality Reduction analysis</div>
   </div>

   <div class='icon-item'>
    <img src="_static/images/icons/comp_ensambles.png" alt="comparing_ensamble" title="We have explored techniques for comparing and analyzing datasets, focusing on the calculation of various metrics " onclick="redirectToPage('comparing_ensambles.html')" style="cursor: pointer;width: 200px; height: 200px;">
    <div style="font-size: 12px; color: #001;">Ensemble Comparison</div>
   </div>






.. toctree::
   :hidden:

   Global_analysis

   local_analysis

   dimensional_reduction

   comparing_ensambles

