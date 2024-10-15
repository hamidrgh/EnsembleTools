Comparing Ensambles
***********************
We have explored techniques for comparing and analyzing datasets, focusing on the calculation of various metrics such as Jensen-Shannon Divergence (JSD) and Earth Mover's Distance (EMD).

Comparing datasets is crucial for understanding the similarities and differences between different datasets or models, and quantifying them can provide insights into the underlying patterns and distributions present in the data.

Practical examples are provided, demonstrating how to manually calculate these metrics, referring to 3 ensembles downloaded directly from the Protein Ensemble Database (PED): PED00156, PED00157, and PED00158. These ensembles represent structural states of the N-terminal SH3 domain of the Drk protein (residues 1-59) in its unfolded form, generated with different approaches to initialize pools of random conformations.

High level (moderate control)
---------------------------------

The code performs two comparative visualizations using the Jensen-Shannon Divergence (JSD) score:

1. **ca_dist**: Comparison of the distributions of the feature ca_dist, visualized with the "Oranges" color map.
2. **alpha_angle**: Comparison of the distributions of the feature alpha_angle, visualized with the "Purples" color map.

**JSD (Jensen-Shannon Divergence)** is a measure of similarity between two probability distributions. It is based on the Kullback-Leibler (KL) divergence but is symmetric and bounded between 0 and 1, where 0 indicates that the distributions are identical and 1 indicates maximum difference.


.. code-block:: python

   bootstrap_iters=5
   fig, ax = plt.subplots(1, 2, figsize=(10.0, 4), dpi=120)

   visualization.comparison_matrix(
      score="jsd",feature="ca_dist",
      bootstrap_iters=bootstrap_iters,
      bins="auto",
      ax=ax[0],
      cmap="Oranges",
      verbose=True
   )

   visualization.comparison_matrix(
      score="jsd",
      feature="alpha_angle",
      bootstrap_iters=bootstrap_iters,
      bins="auto",
      ax=ax[1],
      cmap="Purples",
      verbose=True
   )

   plt.tight_layout()
   plt.show()


.. image:: images/sh3/comparing_ensambles/comp_hl_JSD.png
   :align: center

We note that each feature is compared across 6 pairs of ensembles, with 5 bootstrap iterations for each pair, for a total of 30 comparisons.


.. image:: images/sh3/comparing_ensambles/comparing_hl_JSD.png
   :align: center


The Jensen-Shannon divergence between the distributions of *ca_dist* provides an indication of how different the calcium distributions are across the different pairs of ensembles. A low JSD value would indicate that the distributions are very similar, while a high value would indicate very different distributions.

Similarly, the JSD calculated for *alpha_angle* measures the difference between the distributions of alpha angles among the pairs of ensembles. This can help understand how the alpha angles vary between different conditions or configurations.



Similarly, now the code performs the same two comparative visualizations using the EMD metric.

**The Earth Mover's Distance (EMD)** is a measure of the distance between two probability distributions and is calculated as the minimum cost of transforming one distribution into the other, where the cost is defined in terms of the "work" needed to move the probability mass.


.. code-block:: python

   fig, ax = plt.subplots(1, 2, figsize=(10.0, 4), dpi=120)

   visualization.comparison_matrix(
    score="emd",
    feature="ca_dist",
    bootstrap_iters=bootstrap_iters,
    bins="auto",
    ax=ax[0],
    cmap="Reds",
    verbose=True
   )

   visualization.comparison_matrix(
    score="emd",
    feature="alpha_angle",
    bootstrap_iters=bootstrap_iters,
    bins="auto",
    ax=ax[1],
    cmap="Greens",
    verbose=True
   )

   plt.tight_layout()
   plt.show()

.. image:: images/sh3/comparing_ensambles/comp_hl_EMD.png
   :align: center

- each feature is compared across 6 pairs of ensembles, with 5 bootstrap iterations for each pair, for a total of 30 comparisons.
- Distance Function (RMSD): The Root Mean Square Deviation (RMSD) is used to compare the distributions of ca_dist. This function measures the mean quadratic difference between two distributions, providing a clear and intuitive indication of the differences between them.
- Distance Function (Angular L2): The L2 norm for angles is used to compare the distributions of alpha_angle. This metric is appropriate for measuring differences between angles, as it takes into account the cyclic nature of angles and possible discontinuities.


.. image:: images/sh3/comparing_ensambles/comparing_hl_EMD.png

The **EMD** can range from 0 to ∞ (infinity). 

- A low value of EMD indicates that the two distributions are very similar, requiring little "work" to transform one distribution into the other.
- A high value of EMD indicates that the two distributions are very different.

Intermediate level (slightly more control)
--------------------------------------------

The provided code manually calculates the JSD divergence scores between all ensemble pairs using the Ensemble class. 

Using EnsembleAnalysis to manually calculate divergence scores allows for detailed customization of the analysis process. You can choose the type of score, the number of bootstrap iterations, the method for bins, and more.

.. code-block:: python
   
   boostrap_iters = 5
   bins = "auto"
   score = "jsd"

   # Score divergences in Ca-Ca distances.
   ca_dist_scores, codes = analysis.comparison_scores(
    score=score,
    feature="ca_dist",
    bins=bins,
    bootstrap_iters=boostrap_iters,
    verbose=True
   )

   # Score divergences in alpha torsions.
   alpha_angles_scores, codes = analysis.comparison_scores(
    score=score,
    feature="alpha_angle",
    bins=bins,
    bootstrap_iters=boostrap_iters,
    verbose=True
   )

.. image:: images/sh3/comparing_ensambles/comp_il.png
   :scale: center

After using the comparison_scores function of the analysis class to manually calculate the JSD scores for each feature (ca_dist and alpha_angle), the plot_comparison_matrix function is used to visualize the results.

.. code-block:: python

   fig, ax = plt.subplots(1, 2, figsize=(14.0, 5.5))

   show_std = True
   plot_comparison_matrix(
    ax=ax[0],
    scores=ca_dist_scores,
    codes=codes,
    std=show_std,
    cmap="Oranges",
    title="Distance distributions (aJSD_d)",
    cbar_label="aJSD_d score",
    textcolors=("black", "white")
   )
   plot_comparison_matrix(
    ax=ax[1],
    scores=alpha_angles_scores,
    codes=codes,
    std=show_std,
    cmap="Purples",
    title="Alpha angles distributions (aJSD_t)",
    cbar_label="aJSD_t score",
    textcolors="gray"  # Changes text color.
   )
   plt.tight_layout()
   plt.show()

.. image:: images/sh3/comparing_ensambles/comparing_il.png
   :align: center

This function accepts manually calculated scores and displays them, allowing for greater customization of the graphs.

On the other hand, the previous code (high level-moderate control) directly uses the comparison_matrix function to calculate and display the results in a single step, without the possibility of further customizing the graphs. Therefore, it is simpler and more automated but with fewer customization options.

Low level (full control)
--------------------------


The following code is an example of how to manually calculate JSD (Jensen-Shannon Divergence) and EMD (Earth Mover's Distance) scores between two sets of data. This approach is more manual and provides the highest level of control, as features are extracted manually and scores are calculated without using automated library functions for visualization. 
It allows for customization at every step, from feature calculation to implementing one's own bootstrap strategy.


.. code-block:: python

   ens_1 = analysis["PED00156e001"]
   ens_2 = analysis["PED00157e001"]
   ens_3 = analysis["PED00158e001"]

Here, the aJSD score is calculated for calcium distances (aJSD_d) and for alpha angles (aJSD_t) between two sets of data. 
The variables score_ajsd_d and score_ajsd_t are functions that perform these calculations, also returning the number of bins used.


.. code-block:: python

   score, bins = score_ajsd_d(ens_1, ens_2, bins="auto", return_bins=True)
   print(f"- aJSD_d score: {score:.4f}, bins used: {bins}")

   score, bins = score_ajsd_t(ens_1, ens_2, bins="auto", return_bins=True)
   print(f"- aJSD_t score: {score:.4f}, bins used: {bins}")

.. image:: images/sh3/comparing_ensambles/comp_ll_12.png

.. code-block:: python

   score, bins = score_ajsd_d(ens_1, ens_3, bins="auto", return_bins=True)
   print(f"- aJSD_d score: {score:.4f}, bins used: {bins}")

   score, bins = score_ajsd_t(ens_1, ens_3, bins="auto", return_bins=True)
   print(f"- aJSD_t score: {score:.4f}, bins used: {bins}")

.. image:: images/sh3/comparing_ensambles/comp_ll_13.png

.. code-block:: python

   score, bins = score_ajsd_d(ens_2, ens_3, bins="auto", return_bins=True)
   print(f"- aJSD_d score: {score:.4f}, bins used: {bins}")

   score, bins = score_ajsd_t(ens_2, ens_3, bins="auto", return_bins=True)
   print(f"- aJSD_t score: {score:.4f}, bins used: {bins}")

.. image:: images/sh3/comparing_ensambles/comp_ll_23.png

This code demonstrates how to manually extract alpha angles from each dataset and calculate the average JSD score using these features.


.. code-block:: python

   # Manually compute features (alpha_angles).
   alpha_1 = ens_1.get_features(featurization="a_angle")
   alpha_2 = ens_2.get_features(featurization="a_angle")
   alpha_3 = ens_3.get_features(featurization="a_angle")

   print("- features 1 shape:", alpha_1.shape)
   print("- features 2 shape:", alpha_2.shape)
   # Manually compute average JSD approximation. You can also provide any other
   # 2d feature matrix as input to the `score_avg_jsd` function.
   score, bins = score_avg_jsd(alpha_1, alpha_2, bins="auto", return_bins=True)
   print(f"- aJSD_t score: {score:.4f}, bins used: {bins}")

   print("- features 1 shape:", alpha_1.shape)
   print("- features 3 shape:", alpha_3.shape)
   score, bins = score_avg_jsd(alpha_1, alpha_3, bins="auto", return_bins=True)
   print(f"- aJSD_t score: {score:.4f}, bins used: {bins}")

   print("- features 2 shape:", alpha_2.shape)
   print("- features 3 shape:", alpha_3.shape)
   score, bins = score_avg_jsd(alpha_2, alpha_3, bins="auto", return_bins=True)
   print(f"- aJSD_t score: {score:.4f}, bins used: {bins}")

.. image:: images/sh3/comparing_ensambles/comp_ll_1.png

Finally, the EMD score between the extracted features is manually computed. For angular features, angular_l2 is used as the comparison metric.

.. code-block:: python

   # Let's compute features (alpha_angles).
   alpha_1 = ens_1.get_features(featurization="a_angle")
   alpha_2 = ens_2.get_features(featurization="a_angle")
   alpha_3 = ens_3.get_features(featurization="a_angle")
   print("- features 1 shape:", alpha_1.shape)
   print("- features 2 shape:", alpha_2.shape)
   # Manually score EMD approximation. NOTE: since we are comparing angular features,
   # make sure to use `angular_l2` as the `metric` argument. For all other features
   # (e.g.: interatomic distances) you should use `rmsd` or `l2` instead.
   score = score_emd_approximation(alpha_1, alpha_2, metric="angular_l2")
   print(f"- EMD on alpha angles: {score:.4f}")

   print("- features 1 shape:", alpha_1.shape)
   print("- features 3 shape:", alpha_3.shape)
   score = score_emd_approximation(alpha_1, alpha_3, metric="angular_l2")
   print(f"- EMD on alpha angles: {score:.4f}")

   print("- features 2 shape:", alpha_2.shape)
   print("- features 3 shape:", alpha_3.shape)
   score = score_emd_approximation(alpha_2, alpha_3, metric="angular_l2")
   print(f"- EMD on alpha angles: {score:.4f}")

.. image:: images/sh3/comparing_ensambles/comp_ll_2.png

Note that visualization is not directly included in the code, unlike previous examples where comparison matrices were created and displayed automatically.
