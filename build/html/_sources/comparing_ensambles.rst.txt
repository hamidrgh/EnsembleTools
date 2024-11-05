Comparing Ensambles
***********************
IDPET implements different scores to compare protein ensembles with the same number of residues L, even if their sequences differ. These scores are based on JSD and they compare probability distributions of two interatomic distances or torsion angles.

To estimate JSD between such continuous variables, values are discretized in histograms of N :sub:`bins` bins, similar to previous studies. If we define as P and Q the distributions of a molecular feature in two ensembles, their JSD is:

.. math::

   \operatorname{JSD}(P \parallel Q) = \frac{1}{2} \left( \operatorname{KLD}(P \parallel M) + \operatorname{KLD}(Q \parallel M) \right)

where :math:`M = \frac{1}{2} (P + Q)` is a mixture, and :math:`KLD` is the Kullback-Leibler divergence, expressed as:

.. math::

    KLD(X || M) = \sum_k X_k \log \left( \frac{X_k}{M_k} \right)

where :math:`k` is the bin index and :math:`X_k` and :math:`M_k` are the frequencies for bin :math:`k` estimated from histogram data, with :math:`M_k = \frac{1}{2} (P_k + Q_k)`. No pseudo-counts are used and bins with :math:`X_k = 0` have zero contribution. JSD scores range from 0 (both :math:`P` and :math:`Q` have identical counts) to a maximum of :math:`\log (2) \approx 0.6931` (no bin has at least one count from both :math:`P` and :math:`Q`).


Ensemble compariosn module can provide 3 different scores based on comparing torsion angles and interatomic distances:

- **adaJSD** (carbon Alpha Distance Average JSD)
- **ataJSD** (Alpha Torsion Average JSD)
- **ramaJSD** (RAMAchandran plot average JSD)

adaJSD
======

The adaJSD score compares Cα-Cα distances in two ensembles and is defined as:

.. math::
   
   adaJSD=\frac{1}{Npairs} \sum_{j-i>1}^{ } JSD( D_{ij}^{A} || D_{ij}^{B})


where :math:`D_{ij}^{A}` and :math:`D_{ij}^{B}` are the distance distributions between residue i and j in ensembles A and B, and :math:`Npairs= {(L-1)(L-2)}/{2}` is the total number of distances evaluated. For each distance, its histogram range is defined by its minimum and maximum values in the ensembles.


.. _ataJSD:

ataJSD
======

The ataJSD (Alpha Torsion Average JSD) score compares α angle distributions:

.. math::

    \text{ataJSD} = \frac{1}{N_{\alpha}} \sum_{i=2} \text{JSD}\left( T^{[A]}_{i-1,i,i+2,i+3} \parallel T^{[B]}_{i-1,i,i+2,i+3} \right)

where :math:`N_{\alpha} = L - 3` is the number of α angles in a protein and :math:`T^{[A]}_{i-1,i,i+2,i+3}` and :math:`T^{[B]}_{i-1,i,i+2,i+3}` are the distributions of α angles formed by residue :math:`i` and its neighbors. The histogram range is always :math:`- \pi` to :math:`\pi`.

.. _ramaJSD:

ramaJSD
=======

The ramaJSD (RAMAchandran plot average JSD) score compares joint φ and ψ angle distributions:

.. math::

    \text{ramaJSD} = \frac{1}{N_{\text{rama}}} \sum_{i=2} \text{JSD}\left( R^{[A]}_{i} \parallel R^{[B]}_{i} \right)

where :math:`N_{\text{rama}} = L - 2` is the number of residues with both φ and ψ values and :math:`R^{[A]}_{i}` and :math:`R^{[B]}_{i}` are the joint distributions of φ and ψ for residue :math:`i`. For both angles, the histogram range is always :math:`- \pi` to :math:`\pi` split into :math:`N_{\text{bins}}` bins, resulting in a 2d histogram with an effective bin number of :math:`N_{\text{bins}}^2`.




In this Demo we just simply show how you can visualize a comparison matrix based on the loaded ensembles using visualization module.
A complete guide on ensemble comparison using IDPET has been provided in a separate jupyter notebook in the github repository. 


.. code-block:: python

   bins = "auto"

   # Create a figure with three panels.
   fig, ax = plt.subplots(1, 3, figsize=(15, 8))

   # Plot on the left panel all-vs-all adaJSD scores (compare Ca-Ca distances).
   vis.comparison_matrix(
      score="adaJSD",
      bins=bins,
      ax=ax[0],
      cmap="Oranges",
      verbose=True
   )

   # Plot on the right panel all-vs-all ramaJSD scores (compare phi/psi angles).
   vis.comparison_matrix(
      score="ramaJSD",
      bins=bins,
      ax=ax[1],
      cmap="Purples",
      verbose=True
   )

   # Plot on the right panel all-vs-all ataJSD scores (compare phi/psi angles).
   vis.comparison_matrix(
      score="ataJSD",
      bins=bins,
      ax=ax[2],
      cmap="Reds",
      verbose=True
   )



   plt.tight_layout()
   plt.show()

.. image:: images/sh3/comparing_ensambles/three_scores.png
   :align: center

