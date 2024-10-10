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
In the example below, the figure is saved in the output directory.

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
In this example, we perform analysis on ensembles from PED by displaying several plots on the same figure. 
If axes are not passed as a parameter to the plot functions, they will be displayed as separate plots.

.. code-block:: python

    # Define the ensemble codes with their respective data paths and topology files
    ens_codes = [
        Ensemble('PED00424e001', data_path='path/to/trajectory/file', top_path='path/to/topology/file'),
        Ensemble('PED00423e001', data_path='path/to/trajectory/file', top_path='path/to/topology/file')
    ]

    # Specify the directory where the data is stored
    output_dir = 'path/to/output/directory'

    # Create an instance of EnsembleAnalysis with the specified ensembles and data directory
    analysis = EnsembleAnalysis(ens_codes, output_dir)

    # Load the trajectories for the ensembles
    analysis.load_trajectories()

    # Create an instance of the Visualization class, passing in the analysis object
    vis = Visualization(analysis=analysis)

    # Create a figure and a 2x2 grid of subplots for visualization
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))

    # Plot the end-to-end distances on the first subplot (top-left) using a histogram
    vis.end_to_end_distances(ax=ax[0, 0], violin_plot=False)

    # Plot the asphericity on the second subplot (top-right)
    vis.asphericity(ax=ax[0, 1])

    # Plot the average distance maps on the bottom two subplots
    vis.average_distance_maps(ax=ax[1])

    # Display the figure with all subplots
    fig.show()

This code snippet demonstrates how to load trajectories for multiple ensembles and visualize different structural properties on a single figure. 

.. image:: images/analysis_visualization_example.png
   :alt: Visualization of end-to-end distances, asphericity, and average distance maps

Coarse-Grained Models
---------------------
Coarse-Grained models are supported, however they are incompatible with some functionalities of the package.

Ensemble Files with Multiple Chains
-----------------------------------

When dealing with PDB ensemble files that contain multiple chains, it is necessary to specify which chain to analyze when instantiating an `Ensemble` object.
Example usage:

.. code-block:: python

    # Define the ensemble with a specified chain ID
    ensembles = [
        Ensemble('PED00014e001', database='ped', chain_id='C'),
    ]

    # Specify the directory where the data is stored
    output_dir = 'path/to/output/directory'

    # Create an instance of EnsembleAnalysis with the specified ensembles and output directory
    analysis = EnsembleAnalysis(ensembles=ensembles, output_dir=output_dir)

    # Load the trajectories for the ensembles
    analysis.load_trajectories()

Notes:
    - The `chain_id` parameter should be specified as the chain identifier (e.g., 'A', 'B', 'C').
    - If multiple chains are present and `chain_id` is not specified, an error will be raised.
    - This feature is currently only supported for PDB files.
