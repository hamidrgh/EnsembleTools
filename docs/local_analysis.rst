Local analysis
********************
Local protein analysis is a detailed study of specific regions of a protein to better understand their structure, dynamics, and functionality. This type of analysis is crucial for identifying how individual parts of a protein contribute to its overall behavior and molecular interactions.

In the following package, several functions have been implemented to support this analysis, including the distance map, contact map, and various flexibility and order parameters. These tools are designed to provide in-depth information on the local structure of proteins.

To illustrate the output of the implemented functions, the SH3 protein analysis was chosen as an example. Data for three distinct ensembles were directly downloaded from the Protein Ensemble Database (PED): PED00156, PED00157, and PED00158. These ensembles represent structural states of the N-terminal SH3 domain of the Drk protein (residues 1-59) in its unfolded form, generated using different approaches to initialize pools of random conformations.

- **PED00156**: Conformations generated randomly and optimized through an iterative process.
- **PED00157**: Conformations generated using the ENSEMBLE method, which creates a variety of realistic conformations of an unfolded protein.
- **PED00158**: A combination of conformations from the RANDOM and ENSEMBLE pools, offering greater conformational diversity.

The presented graphs compare the different structural and dynamic characteristics of the various ensembles, providing valuable information on the flexibility and order of the SH3 protein in different states.

Distance map
---------------
We extracted the trajectories' coordinates of the alpha carbon atoms, which are crucial for determining the overall structure and dynamics of proteins. 
Indeed, the spatial arrangement of alpha carbon atoms influences the folding process, stability, and interactions of proteins; they are covalently bonded to amino acid residues and serve as central points for the organization of the protein backbone. 
Therefore, analyzing the distances between alpha carbon atoms provides valuable insights into the local conformation and geometry of the protein chain.

We have printed the graphs related to the distance map for each protein, providing us with a representation of the spatial relationships and distances between these atoms within a protein molecule.

*"ticks_fontsize", "cbar_fontsize", "title_fontsize": Font size for tick labels on the plot axes, for labels on the color bar and for titles of individual subplots, respectively; default is 14.*

*"dpi": Dots per inch (resolution) of the output figure; default is 96.*

*"use_ylabel": If True, y-axis labels are displayed on the subplots; default is True.*

*"save": If True, the plot will be saved as an image file; default is False.*

*"ax": A list or 2D list of Axes objects to plot on; default is None, which creates new axes.*

.. code-block:: python

    visualization.average_distance_maps()

.. image:: images/sh3/local_analysis/dmap.png
   :align: center
  
Contact map
-------------
The contact map is a graphical representation of the contact matrix, whose elements represent the likelihood of contact between two residues, with values approaching 1 indicating close proximity and values approaching 0 indicating spatial separation. 
The graphs show the contact maps generated from the coordinates of the alpha carbon atoms of the proteins under study, aiming to understand the spatial relationships and local interactions within the protein structure.

*"norm": If True, use a log scale range; default is True.*

*"min_sep","max_sep": Minimum and Maximum separation distance between atoms to consider, respectively; default is 2 and None, respectively.*

*"threshold": Determines the threshold for calculating the contact frequencies; default is 0.8 nm.*

*"dpi": For changing the quality and dimension of the output figure; default is 96.*

*"save": If True, the plot will be saved as an image file; default is False.*

*"cmap_color": Select a color for the contact map; default is "Blues".*

*"ax": A list or array of Axes objects to plot on; default is None, which creates new axes.*

.. code-block:: python

    visualization.contact_prob_maps(threshold=0.7)

.. image:: images/sh3/local_analysis/probmap.png
   :align: center

Site-specific flexibility parameter
-------------------------------------
The "Site-specific flexibility parameter" is an indicator that provides a measure of the local flexibility or disorder of a residue within a protein. This measure is based on the circular variance of the Ramachandran angles (φ and ψ) for each residue.

A disorder value close to **0** suggests **rigidity or a stable conformation** of the residue, while a value close to **1** indicates a uniform distribution of dihedral angles and thus **greater flexibility or variability** in the residue's conformation.

*"pointer": A list of desired residues; vertical dashed lines will be added to point to these residues. Default is None.*

*"figsize": The size of the figure. Default is (15, 5).*

*"save": If True, the plot will be saved as an image file. Default is False.*

*"ax": The matplotlib Axes object on which to plot; if None, a new Axes object will be created. Default is None.*

.. code-block:: python

    visualization.ss_flexibility_parameter(pointer=[])

.. image:: images/sh3/local_analysis/ssflex_param.png
   :align: center

Site-specific order parameter 
--------------------------------
The "Site-specific order parameter" is an indicator that evaluates the local order within a protein chain. This parameter measures the orientation correlation between neighboring residues along the protein chain, based on the direction of the Cα-Cα vectors. 

*"pointer": A list of desired residues; vertical dashed lines will be added to point to these residues. Default is None.*

*"figsize": The size of the figure. Default is (15, 5).*

*"save": If True, the plot will be saved as an image file. Default is False.*

*"ax": The matplotlib Axes object on which to plot; if None, a new Axes object will be created. Default is None.*

.. code-block:: python

    visualization.ss_order_parameter(pointer=[])

.. image:: images/sh3/local_analysis/ssorder_param.png
   :align: center

Alpha angles dihedral distribution
--------------------------------------
The dihedral angles represent the rotation around the bonds between consecutive alpha carbons, and their distribution reflects the spatial arrangement of amino acids in the polypeptide chain, directly influencing its three-dimensional conformation.

*"bins": Number of bins for the histogram; default is 50.*

*"save": If True, saves the plot in the data directory; default is False.*

*"ax": The matplotlib Axes object on which to plot; if None, creates a new Axes object.*


.. code-block:: python

    visualization.alpha_angles()

.. image:: images/sh3/global_analysis/dihedral.png
   :align: center

Relative DSSP (Dictionary of Secondary Structure of Proteins) content
------------------------------------------------------------------------
The following function visualizes the relative content of a specific secondary structure (helix, coil, strand) for each residue in various protein ensembles. After checking the compatibility of the analysis, it retrieves the DSSP data of the proteins and creates a plot showing the frequency of the selected structure at each position.

*"dssp_code": This parameter specifies the type of secondary structure to analyze, which can be 'H' for Helix, 'C' for Coil, or 'E' for Strand.*

*"save":If True, the plot will be saved in the data directory. Default is False.*

*"ax": The matplotlib Axes object on which to plot; if None, creates a new Axes object.*

.. code-block:: python

    visualization.relative_dssp_content(self, dssp_code ='H') 

.. image:: images/sh3/global_analysis/contentH.png
   :align: center