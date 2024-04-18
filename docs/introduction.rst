.. _INTRO:

Introduction to Disordered Protein Ensemble Tools
=================================================
TODO: Write the introduction

.. contents:: Table of Contents
    :local:

Workflow Example
----------------

TODO: Perhaps insert a graphical representation of the pipeline steps in order

The first step of the pipeline is to instantiate the 'EnsembleAnalysis'
class by providing a list of ensemble codes to analyse and the path to
the data directory.

.. code-block:: python

    ens_codes = [
        'PED00423e001',
        'PED00424e001'
    ]
    data_dir = 'delete'

    analysis = EnsembleAnalysis(ens_codes, data_dir)

The pipeline expects ensemble data in the 'data_dir' directory in one of the following formats:
    1. [ens_code].dcd (trajectory file) + [ens_code].top.pdb (topology file)
    2. [ens_code].xtc (trajectory file) + [ens_code].top.pdb (topology file)
    3. [ens_code].pdb
    4. Directory [ens_code] containing several .pdb files
    
Where [ens_code] corresponds to the input ensemble codes.

Alternatively, it is possible to download the ensemble files and store them in the data directory
with the following function call, by providing the appropriate database.


.. code-block:: python

    analysis.download_from_database(database='ped')

The next step is to generate MD trajectories from the ensemble files (or load them if trajectory data exists).

.. code-block:: python

    analysis.generate_trajectories()

Optionally, a random number of conformations of given size is sampled from the trajectories for further analysis.

.. code-block:: python

    analysis.random_sample_trajectories(sample_size=200)

Next, feature extraction is performed on the ensemble trajectories by selecting one of the supported methods.
In this example, the features are phi and psi angles computed from the trajectories.

.. code-block:: python

    analysis.perform_feature_extraction(featurization='phi_psi')

Finally, one of several dimensionality reduction methods is performed on the previously extracted features by calling the following function.
In this example, t-SNE dimensionality reduction is performed for perplexities 10, 50 and 100. The data is then clustered using K-means with
the given number of clusters. The best combination of perplexity and number of clusters is determined by computing the silhouette score and the
best results are kept. By setting circular to True, unit vector distance is used as t-SNE metric.

.. code-block:: python

    analysis.fit_dimensionality_reduction(method='tsne', perplexity_vals = [10, 50, 100], circular=True, range_n_clusters=range(2,10,1))

Once the data transformation pipeline is completed, the Visualization class is instantiated to analyse the results. 
In the example below, the figure is saved in the data directory.

.. code-block:: python

    visualization = Visualization(analysis)
    visualization.tsne_scatter_plot(save=True)

.. image:: images/tsnep10_kmeans2_scatter.png
   :width: 600