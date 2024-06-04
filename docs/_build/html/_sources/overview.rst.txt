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

  # read in the simulation trajectory 
  TO = SSTrajectory('traj.xtc', 'start.pdb')

  # once the trajectory has been read in, proteins can be extracted
  # from the proteinTrajectoryList
  
  protein = TO.proteinTrajectoryList[0]

  # calculate per-residue distance between residues 10 and 20
  d_10_20 = protein.get_inter_residue_COM_distance(10, 20)

  # calculate the ensemble asphericity
  asph = protein.get_asphericity()

  # calculate the ensemble asphericity
  rg = protein.get_radius_of_gyration()

  # calculate the ensemble distance map
  dm = protein.get_distance_map()