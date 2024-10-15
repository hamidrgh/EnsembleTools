Overview
==========

IDPET is a Python library that can analyze multiple structural ensembles of disordered proteins in parallel. The data can be automatically downloaded from databases like the Protein Ensemble Database (PED) using the package or loaded from local data files. IDPET provides various functions to facilitate the visualization of results through different plots.
Using mdtraj as the backend engine, IDPET can read and load multiple datasets as input. Through web APIs, it can directly download, store, and analyze data from both PED and ATLAS, which are two important databases for studying disordered and flexible proteins.
Furthermore, by implementing different dimensionality reduction algorithms within the package, a wide variety of analyses can be performed using IDPET.

As an example:

.. code-block:: python

  # import idpet modules for reading, analyzing and visualizing of the IDP ensembles   		
 from dpet.ensemble import Ensemble
 from dpet.ensemble_analysis import EnsembleAnalysis
 from dpet.visualization import Visualization


There are two possibilities for loading the data:

- Downloading directly from specified databases: **PED & ATLAS**

.. code-block:: python

  ensembles = [
    Ensemble(code='3a1g_B', database='atlas')
  ]

  ensembles = [
    Ensemble(code='PED00156e001', database='ped'),
    Ensemble(code='PED00157e001', database='ped'),
    Ensemble(code='PED00158e001', database='ped')
  ]


- Loading from specified File Paths:

    - using multi-model pdb files 
    - using trajectory files such as .dcd or .xtc
    - using a directory with separate pdb files for each model

.. code-block:: python

  ensembles = [
    Ensemble(code='PED00156e001', data_path='path/to/data/PED00156e001.pdb'),
    Ensemble(code='PED00157e001', data_path='path/to/data/PED00157e001.dcd', top_path='path/to/data/PED00157e001.top.pdb'),
    Ensemble(code='PED00158e001', data_path='path/to/data/directory_contains_serpate_pdb_for_each_model')]


- How to visualize the analysis:  

.. code-block:: python
    
  # Create an EnsembleAnalysis object with the given ensembles and specify the output directory for saving the results
  analysis = EnsembleAnalysis(ensembles=ensembles, output_dir='path/to/output_directory')
  # Load the trajectories for each ensemble 
  analysis.load_trajectories()

  # Create a Visualization object using the EnsembleAnalysis object 
  #to enable visualization of the analysis results
  vis = Visualization(analysis)


 # Visualize the distribution of the radius of gyration 
 vis.radius_of_gyration()

 # Visualize the contact probability maps
 vis.contact_prob_maps()

 # Visualize the comparison matrix between loaded ensembles 
 vis.comparison_matrix()