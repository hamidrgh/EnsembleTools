Local analysis
********************

In this part of the demo, we continue the analysis on the SH3 domain example from the Protein Ensemble Database (PED) and highlight how the IDPET package can provide insights into local structural information from conformational ensembles. Specifically, this demo shows how to extract the following information:

- Contact maps
- Ramachandran plots
- Alpha angle distribution
- Relative DSSP through each ensemble (secondary structure)
- Site-specific flexibility and order parameters

Initialize the analysis
------------------
Initializition of the analysis already described in the demo for the global analysis.
  
Contact map
-------------
The contact map is a graphical representation of the contact matrix, whose elements represent the likelihood of contact between two residues, with values approaching 1 indicating close proximity and values approaching 0 indicating spatial separation. 
The graphs show the contact maps generated from the coordinates of the alpha carbon atoms of the proteins under study, aiming to understand the spatial relationships and local interactions within the protein structure.

*"log_scale": If True, use a log scale range; default is True.*

*"threshold": Determines the threshold for calculating the contact frequencies; default is 0.8 nm.*

*"dpi": For changing the quality and dimension of the output figure; default is 96.*

*"save": If True, the plot will be saved as an image file; default is False.*

*"cmap_color": Select a color for the contact map; default is "Blues".*

*"ax": A list or array of Axes objects to plot on; default is None, which creates new axes.*

.. code-block:: python

    vis.contact_prob_maps(log_scale=True, threshold=0.7)

.. image:: images/sh3/local_analysis/probmap.png
   :align: center

2D Ramachandran histograms
---------------------------
The function generates Ramachandran plots to visualize the distribution of phi (ϕ) and psi (ψ) torsion angles of proteins within the ensembles. To calculate the torsion angles, MDTraj functions are used, and the results are then converted to degrees using *np.degrees*. 
If *two_d_hist* is set to False, it returns a simple scatter plot for all ensembles in a single plot. If set to True, it returns a 2D histogram for each ensemble, where the angles are grouped into a 2D histogram showing the population density of the conformations.

*"two_d_hist"*: Boolean that determines whether to display a 2D histogram (True) or a scatter plot (False).

*"linespaces"*: Tuple that specifies the range and the number of bins for the 2D histogram.

*"save"*: If True, saves the plot in the data directory; default is False.*

*"ax"*: The matplotlib Axes object on which to plot; if None, creates a new Axes object.


.. code-block:: python

   vis.ramachandran_plots(two_d_hist=True)

.. image:: images/sh3/local_analysis/rama.png
   :align: center


Alpha angles dihedral distribution
--------------------------------------
Alpha angles are a specific type of dihedral angle calculated using the C-alpha (Cα) atoms of a protein backbone. A dihedral angle, also known as a torsion angle, is the angle between two planes formed by four sequentially bonded atoms, providing insight into the 3D conformation of the molecule.

To calculate alpha angles, the indices of all Cα atoms in the protein are identified, and sets of four consecutive Cα atoms are grouped. Using these groups, the torsion (dihedral) angles are computed with the MDTraj function, which takes the trajectory and a list of tuples containing the indices of the four Cα atoms. The output is a numpy array with the dihedral angles for each set, representing the alpha angles that provide important insights into the protein's three-dimensional structure and dynamics.

*"bins": Number of bins for the histogram; default is 50.*

*"save": If True, saves the plot in the data directory; default is False.*

*"ax": The matplotlib Axes object on which to plot; if None, creates a new Axes object.*


.. code-block:: python

    vis.alpha_angles()

.. image:: images/sh3/global_analysis/dihedral.png
   :align: center

Relative DSSP (Dictionary of Secondary Structure of Proteins) content
------------------------------------------------------------------------
The following function visualizes the relative content of a specific secondary structure (helix, coil, strand) for each residue in various protein ensembles. It retrieves the DSSP data of the proteins and creates a plot showing the frequency of the selected structure at each position.
This function does not work for coarse-grained models.

*"dssp_code": This parameter specifies the type of secondary structure to analyze, which can be 'H' for Helix, 'C' for Coil, or 'E' for Strand.*

*"save":If True, the plot will be saved in the data directory. Default is False.*

*"ax": The matplotlib Axes object on which to plot; if None, creates a new Axes object.*

.. code-block:: python

    vis.relative_dssp_content(dssp_code ='H') 

.. image:: images/sh3/global_analysis/contentH.png
   :align: center

Site-specific flexibility parameter
-------------------------------------
The "Site-specific flexibility parameter" quantifies the local flexibility of a protein chain at a specific residue, it anges from 0 (high flexibility) to 1 (no flexibility).
If all conformers have the same dihedral angles at a residue, the circular variance is equal to one, indicating no flexibility, conversely, for a large ensemble with a uniform distribution of dihedral angles, the circular variance tends to zero.

The site-specific flexibility parameter is defined using the circular variance of the Ramachandran angles

:math:`\phi_{i}` and :math:`\psi_{i}`. The circular variance of :math:`\phi_{i}` is given by:

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

    vis.ss_flexibility(pointer=[5,20])

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

    vis.ss_order(pointer=[5, 20])

.. image:: images/sh3/local_analysis/ssorder_param.png
   :align: center

