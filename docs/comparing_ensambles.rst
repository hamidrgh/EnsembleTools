Comparing Ensambles
***********************
High level (moderate controll)
---------------------------------

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


.. image:: images/sh3/comparing_ensambles/comp_hl_JDS.png
   :align: center

.. image:: images/sh3/comparing_ensambles/comparing_hl_JDS.png
   :align: center

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

.. image:: images/sh3/comparing_ensambles/comparing_hl_EMD.png

Intermediate level (slightly more control)
--------------------------------------------

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

Low level (full control)
--------------------------

.. code-block:: python

   ens_1 = analysis["PED00156e001"]
   ens_2 = analysis["PED00157e001"]
   ens_3 = analysis["PED00158e001"]

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

.. code-block:: python

   # Manually compute features (alpha_angles).
   alpha_1 = ens_1.get_features(featurization="a_angle")
   print("- features 1 shape:", alpha_1.shape)
   alpha_2 = ens_2.get_features(featurization="a_angle")
   print("- features 2 shape:", alpha_2.shape)
   # Manually compute average JSD approximation. You can also provide any other
   # 2d feature matrix as input to the `score_avg_jsd` function.
   score, bins = score_avg_jsd(alpha_1, alpha_2, bins="auto", return_bins=True)
   print(f"- aJSD_t score: {score:.4f}, bins used: {bins}")

   # Manually compute features (alpha_angles).
   alpha_1 = ens_1.get_features(featurization="a_angle")
   print("- features 1 shape:", alpha_1.shape)
   alpha_3 = ens_3.get_features(featurization="a_angle")
   print("- features 3 shape:", alpha_3.shape)
   # Manually compute average JSD approximation. You can also provide any other
   # 2d feature matrix as input to the `score_avg_jsd` function.
   score, bins = score_avg_jsd(alpha_1, alpha_3, bins="auto", return_bins=True)
   print(f"- aJSD_t score: {score:.4f}, bins used: {bins}")

   # Manually compute features (alpha_angles).
   alpha_1 = ens_2.get_features(featurization="a_angle")
   print("- features 2 shape:", alpha_2.shape)
   alpha_2 = ens_3.get_features(featurization="a_angle")
   print("- features 3 shape:", alpha_3.shape)
   # Manually compute average JSD approximation. You can also provide any other
   # 2d feature matrix as input to the `score_avg_jsd` function.
   score, bins = score_avg_jsd(alpha_2, alpha_3, bins="auto", return_bins=True)
   print(f"- aJSD_t score: {score:.4f}, bins used: {bins}")

.. image:: images/sh3/comparing_ensambles/comp_ll_1.png

.. code-block:: python

   # Let's compute features (alpha_angles).
   alpha_1 = ens_1.get_features(featurization="a_angle")
   print("- features 1 shape:", alpha_1.shape)
   alpha_2 = ens_2.get_features(featurization="a_angle")
   print("- features 2 shape:", alpha_2.shape)
   # Manually score EMD approximation. NOTE: since we are comparing angular features,
   # make sure to use `angular_l2` as the `metric` argument. For all other features
   # (e.g.: interatomic distances) you should use `rmsd` or `l2` instead.
   score = score_emd_approximation(alpha_1, alpha_2, metric="angular_l2")
   print(f"- EMD on alpha angles: {score:.4f}")

   # Let's compute features (alpha_angles).
   alpha_1 = ens_1.get_features(featurization="a_angle")
   print("- features 1 shape:", alpha_1.shape)
   alpha_3 = ens_3.get_features(featurization="a_angle")
   print("- features 3 shape:", alpha_3.shape)
   # Manually score EMD approximation. NOTE: since we are comparing angular features,
   # make sure to use `angular_l2` as the `metric` argument. For all other features
   # (e.g.: interatomic distances) you should use `rmsd` or `l2` instead.
   score = score_emd_approximation(alpha_1, alpha_3, metric="angular_l2")
   print(f"- EMD on alpha angles: {score:.4f}")

   # Let's compute features (alpha_angles).
   alpha_2 = ens_2.get_features(featurization="a_angle")
   print("- features 2 shape:", alpha_2.shape)
   alpha_3 = ens_3.get_features(featurization="a_angle")
   print("- features 3 shape:", alpha_3.shape)
   # Manually score EMD approximation. NOTE: since we are comparing angular features,
   # make sure to use `angular_l2` as the `metric` argument. For all other features
   # (e.g.: interatomic distances) you should use `rmsd` or `l2` instead.
   score = score_emd_approximation(alpha_2, alpha_3, metric="angular_l2")
   print(f"- EMD on alpha angles: {score:.4f}")

.. image:: images/sh3/comparing_ensambles/comp_ll_2.png