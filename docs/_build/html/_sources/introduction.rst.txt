.. _INTRO:

Introduction to Disordered Protein Ensemble Tools
=================================================
TODO: Write the introduction

.. contents:: Table of Contents
    :local:

Basic Workflow Example
----------------------

TODO: Perhaps insert a graphical representation of the pipeline steps in order

The first step of the pipeline is to instantiate the EnsembleAnalysis class by providing a list of Ensembles to analyze and the path to the output directory. 
Different input types are supported. Ensembles can be loaded from a data file (for example, a PDB file), a trajectory file with an accompanying topology file, 
or the ensembles can be downloaded from a database.

.. code-block:: python

    ensembles = [
        Ensemble(code = 'PED00423e001', data_path = 'path/to/data/file')
        Ensemble(code = 'PED00424e001', data_path = 'path/to/traj/file', top_path = 'path/to/top/file')
    ]
    output_dir = 'path/to/output/directory'

    analysis = EnsembleAnalysis(ensembles = ensembles, output_dir = output_dir)
    analysis.load_trajectories()

Optionally, a random number of conformations of given size is sampled from the trajectories for further analysis.

.. code-block:: python

    analysis.random_sample_trajectories(sample_size = 200)

Next, feature extraction is performed on the ensemble trajectories by selecting one of the supported methods.
In this example, the features are phi and psi angles computed from the trajectories.

.. code-block:: python

    analysis.extract_features(featurization='phi_psi')

Finally, one of several dimensionality reduction methods is performed on the previously extracted features by calling the following function.
In this example, t-SNE dimensionality reduction is performed for perplexities 10, 50 and 100. The data is then clustered using K-means with
the given number of clusters. The best combination of perplexity and number of clusters is determined by computing the silhouette score and the
best results are kept. By setting circular to True, unit vector distance is used as t-SNE metric.

.. code-block:: python

    analysis.reduce_features(method='tsne', perplexity_vals = [10, 50, 100], circular=True, range_n_clusters=range(2,10,1))

Once the data transformation pipeline is completed, the Visualization class is instantiated to analyse the results. 
In the example below, the figure is saved in the data directory.

.. code-block:: python

    visualization = Visualization(analysis)
    visualization.tsne_scatter(save=True)

.. image:: images/tsnep10_kmeans2_scatter.png

End-to-End Example
------------------

In the example below, the complete data analysis pipeline is executed with one function call. The pipeline consists of:
    1. Downloading trajectories from the Atlas database
    2. Loading trajectories
    3. Randomly sampling 200 conformations
    4. Extracting features
    5. Performing dimensionality reduction on the features

.. code-block:: python

    ensembles = [
        Ensemble('3a1g_B', database='atlas')
    ]
    output_dir = 'path/to/output/directory'

    featurization_params = {'featurization': 'ca_dist'}
    reduce_dim_params = {'method': 'dimenfix', 'range_n_clusters':[2, 3, 4, 5, 6]}

    analysis = EnsembleAnalysis(ensembles = ensembles, output_dir = output_dir)
    analysis.execute_pipeline(featurization_params = featurization_params, reduce_dim_params = reduce_dim_params, subsample_size=  200)

As in the previous example, the transformed ensemble data can be visualised with supported plot functions.

.. code-block:: python

    visualization = Visualization(analysis)
    visualization.dimenfix_scatter()

.. image:: images/dimenfix_scatter.png

Plot Example
------------

In this example, analysis is performed on the same ensembles from Atlas, extracting Cα-Cα distances as features and
transforming them using PCA, which is fit only on replicate n°1 (3a1g_B_prod_R1_fit).

.. code-block:: python

    ensembles = [
        Ensemble('3a1g_B', database='atlas')
    ]
    output_dir = 'path/to/output/directory'

    analysis = EnsembleAnalysis(ensembles = ensembles, output_dir = output_dir)
    analysis.execute_pipeline(
        featurization_params={'featurization':'ca_dist'}, 
        reduce_dim_params={'method':'pca','fit_on':["3a1g_B_prod_R1_fit"]}
        subsample_size=200)

There is an option to automatically generate a PDF report containing all plots relevant to the conducted analysis.
The report is saved in the data directory.

.. code-block:: python

    visualization = Visualization(analysis)
    visualization.generate_report()

Alternatively, different plots can be called explicitly, optionally setting save to True to save the plots as PNGs in the data directory.

.. code-block:: python

    visualization.pca_plot_2d_landscapes(save=True)

.. image:: images/PCA_RG3a1g_B_prod_R1_fit.png

.. code-block:: python

    visualization.pca_plot_1d_histograms(save=True)

.. image:: images/PCA_histca_dist3a1g_B_prod_R1_fit.png

All plots called in one session get stored in a dictionary in the Vizualization class. 
Calling the following function outputs all of them into a PDF report.

.. code-block:: python

    visualization.generate_custom_report()

Coarse-Grained Models
---------------------
Coarse-Grained models are supported, however they are incompatible with some functionalities of the package.

Ensemble Files with Multiple Chains
-----------------------------------
When dealing with ensemble files that contain multiple chains, it is possible to specify which one to analyse.
In the context of MDtraj trajectories, chain identifiers are represented as numerical indexes (e.g., 0, 1, 2, etc.) and are assigned sequentially. 
For example, chains A, C, and D will be assigned chain indexes 0, 1, and 2, respectively.

.. code-block:: python

    ensembles = [
        Ensemble('PED00014e001', database='ped', chain_id=1),
    ]
    output_dir = 'path/to/output/directory'

    analysis = EnsembleAnalysis(ensembles = ensembles, output_dir = output_dir)
    analysis.load_trajectories()

If not specified, the program will prompt the user to select one chain per ensemble for analysis.