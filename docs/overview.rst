Overview
==========

IDPET is a python library which can easily analyze multiple structural ensembles of disordered protein in paralle. The data could automatically downloaded from the databases like Protein Ensemble Database (PED) using the package or could be loaded from local data files. IDPET provide various functions to faciliatate the visualization of the results through various plots.
Using mdtraj as the backend engine, IDPET can read and load multiple data as the inputs and through web APIs it can directly download, store and analyzes the data from PED and ATLAS which are two important databases for disordered and flexible proteins. 
Moreover, by implementing different dimensionality reduction algorithms whithin the package more various types of analysis can be performed using IDPET. 

As an example:

.. code-block:: python

  # import idpet modules for reading, analyzing and visualizing of the IDP ensembles   		
 from dpet.ensemble import Ensemble
 from dpet.ensemble_analysis import EnsembleAnalysis
 from dpet.visualization import Visualization


There are three possibilities for loading the data:

- Downloading from the **atlas** database:

.. code-block:: python

  ensembles = [
    Ensemble(ens_code='3a1g_B', database='atlas')
  ]

- Downloading from the **PED** database:

.. code-block:: python

  ensembles = [
    Ensemble(code='PED00156e001', database='ped'),
    Ensemble(code='PED00157e001', database='ped'),
    Ensemble(code='PED00158e001', database='ped')
  ]

- Loading from specified File Paths:

.. code-block:: python

  ensembles = [
    Ensemble(code='PED00156e001', data_path='path/to/data/PED00156e001.pdb'),
    Ensemble(code='PED00158e001', data_path='path/to/data/PED00158e001.dcd', top_path='path/to/data/PED00158e001.top.pdb')]


- How to visualize the analysis:  

.. code-block:: python
    
  # Create an EnsembleAnalysis object with the given ensembles and specify the output directory
  analysis = EnsembleAnalysis(ensembles=ensembles, output_dir='path/to/output_directory')
  # Load the trajectories for each ensemble
  analysis.load_trajectories()

  # Create a Visualization object using the EnsembleAnalysis object 
  #to enable visualization of the analysis results
  visualization = Visualization(analysis)


 # Visualize the distribution of the radius of gyration 
 visualization.radius_of_gyration()

 # Visualize the contact probability maps
 visualization.contact_prob_maps()

 # Visualize the comparison matrix between loaded ensembles 
 visualization.comparison_matrix()