Local analysis
********************
Local protein analysis is a detailed study of specific regions of a protein to better understand their structure, dynamics, and functionality. This type of analysis is crucial for identifying how individual parts of a protein contribute to its overall behavior and molecular interactions.

In the following package, several functions have been implemented to support this analysis, including the distance map, contact map, and various flexibility and order parameters. These tools are designed to provide in-depth information on the local structure of proteins.

To illustrate the output of the implemented functions, the SH3 protein analysis was chosen as an example. Data for three distinct ensembles were directly downloaded from the Protein Ensemble Database (PED): PED00156, PED00157, and PED00158. These ensembles represent structural states of the N-terminal SH3 domain of the Drk protein (residues 1-59) in its unfolded form, generated using different approaches to initialize pools of random conformations.

- **PED00156**: Conformations generated randomly and optimized through an iterative process.
- **PED00157**: Conformations generated using the ENSEMBLE method, which creates a variety of realistic conformations of an unfolded protein.
- **PED00158**: A combination of conformations from the RANDOM and ENSEMBLE pools, offering greater conformational diversity.

The presented graphs compare the different structural and dynamic characteristics of the various ensembles, providing valuable information on the flexibility and order of the SH3 protein in different states.

  
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

Alpha angles dihedral distribution
--------------------------------------
Alpha angles are a type of dihedral angle calculated using the backbone atoms of the protein, typically involving the C-alpha (Cα) atoms. So frist of all we remind that a dihedral angle, also known as a torsion angle, is the angle between two planes formed by four sequentially bonded atoms in a molecule and  it provides insight into the 3D conformation of the molecule.
Consequentially the calculation of alpha angles involves computing the dihedral angles formed by consecutive Cα atoms. Frist of all the code identify the indices of all Cα atoms in the protein and create sets of four consecutive Cα atoms. Afterwards, using these sets of atoms, the torsion angles are calculated using the *MDTraj* function, that takes in input the trajectory and a list of tuples, where each tuple contains the indices of four consecutive Cα atoms. The output is a numpy array that contains the dihedral angles calculated for each set of four consecutive Cα atoms. These dihedral angles represent the alpha angles of the protein and provide crucial insights into its three-dimensional conformation and dynamics.

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

Site-specific flexibility parameter
-------------------------------------
The "Site-specific flexibility parameter" quantifies the local flexibility of a protein chain at a specific residue, it anges from 0 (high flexibility) to 1 (no flexibility).
If all conformers have the same dihedral angles at a residue, the circular variance is equal to one, indicating no flexibility, conversely, for a large ensemble with a uniform distribution of dihedral angles, the circular variance tends to zero.

The site-specific flexibility parameter is defined using the circular variance of the Ramachandran angles :math:` \phi_{i}` and :math:`\psi_{i}`. The circular variance of :math:`\phi_{i}` is given by:

.. math::

   R_{\phi_{i}} =(\frac{1}{C} \sum_{c=1}^{C} w_{c} sin \phi_{i,c})^2 + (\frac{1}{C} \sum_{c=1}^{C} w_{c} cos \phi_{i,c})^2

An analogous expression applies for :math:`R_{\psi_{i}}`. The site-specific flexibility parameter :math:`f_{i}` is then defined as:

.. math::

    f_i = 1 - \frac{1}{2} \left( R_{\phi_i} + R_{\psi_i} \right) 
 
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
The parameter is derived by computing the ensemble mean of the cosine of the angle between these vectors and assessing its variance across conformers. 

The mean orientation correlation :math:`<cos \theta_{ij}>` is calculated as:

.. math::

   <cos \theta_{ij}> = \frac{1}{C} \sum_{c=1}^{C} w_{c} cos\theta_{ij,c}

Where:

- :math:`w_c` is the weight of conformer :math:`c`
- :math:`C` is the total number of conformers
- :math:`cos \theta_{ij,c}` is  the cosine of the angle between vectors :math:`r_{i,i+1}` and :math:`r_{j,j+1}` for conformer :math:`c`.

The variance :math:`<\sigma_{ij}^2>` of  :math:`<cos \theta_{ij}>` is given by:

.. math::

   \sigma_{ij}^2 = \frac{1}{C} \sum_{c=1}^{C} (w_{c} cos  \theta_{ij,c}-<cos  \theta_{ij}>)^2


The site-specific order parameter :math:`K_{ij}` is defined as:

.. math::

   K_{ij} = 1 - \sigma_{ij}^2

To characterize the order at residue :math:`i`  in relation to the entire chain, the site-specific order parameter :math:`o_{i}` is computed by summing :math:`K_{ij}` over all residues :math:`j` :

.. math::

   o_{i} = \frac{1}{N} \sum_{j=1}^{N} K_{ij}

where :math:`N` represents the total number of residues in the protein chain. 


*"pointer": A list of desired residues; vertical dashed lines will be added to point to these residues. Default is None.*

*"figsize": The size of the figure. Default is (15, 5).*

*"save": If True, the plot will be saved as an image file. Default is False.*

*"ax": The matplotlib Axes object on which to plot; if None, a new Axes object will be created. Default is None.*

.. code-block:: python

    visualization.ss_order_parameter(pointer=[])

.. image:: images/sh3/local_analysis/ssorder_param.png
   :align: center

