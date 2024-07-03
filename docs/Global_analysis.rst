Global analysis
*******************
The global analysis of a protein is a fundamental technique for understanding its structural and dynamic properties. This approach allows the examination of various parameters, such as the radius of gyration, asphericity, and the spatial distribution of atoms, providing an overview of the protein's conformation and behavior.

In our package, we have implemented several functions for visualizing these parameters. These functions help interpret and compare results, making protein analysis easier.

To illustrate the output of our functions, we have chosen the analysis of the SH3 protein as an example. The graphs we present are related to the analysis of this protein, comparing three distinct ensembles downloaded directly from the Protein Ensemble Database (PED): PED00156, PED00157, and PED00158. These ensembles represent structural states of the N-terminal SH3 domain of the Drk protein (residues 1-59) in its unfolded form, generated with different approaches to initialize pools of random conformations.

- **PED00156**: This ensemble consists of conformations generated randomly and optimized through an iterative process.
- **PED00157**: This ensemble includes conformations generated using the ENSEMBLE method, which creates a variety of realistic conformations of an unfolded protein.
- **PED00158**: This ensemble is a combination of conformations from the RANDOM and ENSEMBLE pools, offering greater conformational diversity.

Radius of gyration
------------------
The radius of gyration of a protein is the square root of the mean square distances of the protein residues from the protein's center of mass. It is therefore a measure of the compactness of its three-dimensional structure.

*"bins": Number of bins for the histogram; default is 50.*

*"hist_range": A tuple defining the min and max values for the histogram; if None, uses the data range.*

*"multiple_hist_ax": If True, plots each histogram on a separate axis.*

*"violin_plot": If True, displays a violin plot; default is False.*

*"location": It allows you to specify whether to calculate and use the mean ('mean') or the median ('median') as the reference value.*

*"dpi": The DPI (dots per inch) of the output figure; default is 96.*

*"save": If True, saves the plot as an image file; default is False.*

*"ax": The matplotlib Axes object(s) on which to plot; if None, new Axes object(s) will be created.*

.. code-block:: python

    visualization.radius_of_gyration(violin_plot=True ,location='median')

.. image:: images/sh3/global_analysis/r_of_g.png 
   :align: center
  
.. code-block:: python

    visualization.radius_of_gyration(multiple_hist_ax=True ,hist_range=(0.5,4) ,mean=True, median=True, bins=40)

.. image:: images/sh3/global_analysis/rg.png
   :align: center

End to end distance
---------------------
The end_to_end_distances function is designed to visualize the distributions of end-to-end distances in molecular structure trajectories. The end-to-end distance refers to the distance between the first and last atom in a molecular chain, a parameter often used to understand the overall conformation of a molecule.

*"rg_norm"(bool,optional):If set to True, normalizes the end-to-end distances based on the average radius of gyration.[The default value is False]*

*"bins"(int,optional): Defines the number of bins for the histogram of distances. More bins provide a more detailed distribution, but it may be noisier.[The default value is 50]*

*"hist_range"(tuple, optional):A tuple specifying the minimum and maximum values for the histogram. If not specified (None), it uses the minimum and maximum values present in the data.*

*"violin_plot"(bool,optional):If set to True, the function will generate a violin plot of the distances. [The default value is True]*

*"location": It allows you to specify whether to calculate and use the mean ('mean') or the median ('median') as the reference value.*

*"save" (bool, optional):If set to True, the generated plot will be saved as an image file. [The default value is False]*

*"ax" (plt.Axes, optional):The axes on which to plot. [The default value is None]*

The end-to-end distances are computed by selecting the Cα atoms from each ensemble's trajectory and calculating the distance between the first and last Cα atoms in each frame:

.. math::

   d_{\text{end-to-end}} = \| \mathbf{r}_{\text{end}} - \mathbf{r}_{\text{start}} \|

Where:

- :math:`\mathbf{r}_{\text{start}}` is the position of the first Cα atom.
- :math:`\mathbf{r}_{\text{end}}` is the position of the last Cα atom.


This example generates a violin plot of the end-to-end distances without showing the means but including the medians.


.. code-block:: python

    visualization.end_to_end_distances(location='median')

.. image:: images/sh3/global_analysis/end_to_end.png 
   :align: center

It is also possible normaize the end-to-end distances based on the average radius of gyration, generating violin plots that include the means.

.. math::

   d_{\text{normalized}} = \frac{d_{\text{end-to-end}}}{RG}

Where RG is the mean Radius of Gyration of the trajectory.

.. code-block:: python

    visualization.end_to_end_distances(rg_norm=True, violin_plot=True, location='mean')

.. image:: images/sh3/global_analysis/end_to_end_norm.png 
   :align: center
  

Asphericity distribution
---------------------------
The asphericity the measure of deviation from the spherical shape of a molecule. It indicates how much a molecule differs from the ideal spherical form. A protein with an asphericity greater than zero is generally more elongated or flattened compared to a sphere.
In order to obtain this values, frits of all the gyration tensor is computed for each frame of the trajectory,then eigenvalues of this tensor, which indicate the principal moments of inertia and reflect the molecule's shape and symmetry, are sorted in ascending order. 
Finally asphericity is then computed using the formula:


.. math::

   \text{Asphericity} = 1-3 \frac{\lambda_{1}\lambda_{2}+\lambda_{2}\lambda_{3}+\lambda_{3}\lambda_{1}}{(\lambda_{1}+\lambda_{2}+\lambda_{3})^2}

where :math:`\lambda_{1},\lambda_{2},\lambda_{3}` are the sorted eigenvalues of the gyration tensor.



*"bins": Number of bins for the histogram; default is 50.*

*"hist_range": A tuple defining the min and max values for the histogram; if None, uses the data range.*

*"violin_plot": If True, displays a violin plot; default is True.*

*"location": It allows you to specify whether to calculate and use the mean ('mean') or the median ('median') as the reference value.*

*"save": If True, saves the plot as an image file; default is False.*

*"ax": The matplotlib Axes object on which to plot; if None, creates a new figure and axes.*

.. code-block:: python

    visualization.asphericity(location='mean')

.. image:: images/sh3/global_analysis/asphericity.png
   :align: center
  
.. code-block:: python

    visualization.asphericity(violin_plot=False)

.. image:: images/sh3/global_analysis/asphericity1.png
   :align: center

Prolatness distribution
--------------------------
 The prolateness the measure of a molecule's shape, indicating how elongated it is compared to its transverse dimensions. A protein with a prolateness greater than one is generally more elongated than a spherical shape.
After computing the gyration tensor for each frame of the trajectory and sorting the eigenvalues of the gyration tensor in ascending order, the prolateness is then calculated using the following formula:

.. math::

   \text{Prolatness} =  \frac{\lambda_{2}-\lambda_{1}}{\lambda_{3}}

where :math:`\lambda_{1},\lambda_{2},\lambda_{3}` are the sorted eigenvalues of the gyration tensor.


*"bins": Number of bins for the histogram; default is 50.*

*"hist_range": A tuple defining the min and max values for the histogram; if None, uses the data range.*

*"violin_plot": If True, displays a violin plot; default is True.*

*"means": If True, shows the means in the violin plot; default is True.*

*"median": If True, shows the medians in the violin plot; default is True.*

*"save": If True, saves the plot as an image file; default is False.*

*"ax": The matplotlib Axes object on which to plot; if None, creates a new figure and axes.*

.. code-block:: python

    visualization.prolatness()

.. image:: images/sh3/global_analysis/prolatness.png
   :align: center
  


Radius of gyration vs Asphericity
--------------------------------------
The function *rg_vs_asphericity* also prints the Pearson correlation coefficients, which measure the strength and direction of the linear relationship between the radius of gyration (Rg) and asphericity. A Pearson coefficient value close to 1 or -1 indicates a strong positive or negative correlation, respectively, while a value close to 0 indicates a weak or no correlation.

*"save": If True, saves the plot as an image file; default is False.*

*"ax": The matplotlib Axes object on which to plot; if None, creates a new figure and axes.*


.. code-block:: python

    visualization.rg_vs_asphericity()

.. image:: images/sh3/global_analysis/rgasp.png
   :align: center
   :scale: 70%

.. image:: images/sh3/global_analysis/rg_vs_asph.png
   :align: center


Radius of gyration vs Prolatness
---------------------------------
The function *rg_vs_prolateness* also prints the Pearson correlation coefficients, which measure the strength and direction of the linear relationship between the radius of gyration (Rg) and prolateness. A Pearson coefficient value close to 1 or -1 indicates a strong positive or negative correlation, respectively, while a value close to 0 indicates a weak or no correlation.

.. code-block:: python

    visualization.rg_vs_prolatness()

.. image:: images/sh3/global_analysis/rgproll.png
   :align: center
   :scale: 70%

.. image:: images/sh3/global_analysis/rg_vs_prol.png
   :align: center
  

Global sasa distribution
---------------------------
The acronym 'SASA' stands for 'Solvent Accessible Surface Area,' which denotes the surface area of a molecule that is accessible to the solvent. The Shrake-Rupley algorithm in MDTraj calculates the SASA based on the positions of atoms and the probe radius used for the calculation. At the conformational level, 'total SASA' indicates the overall surface area accessible to the solvent for each conformation within the molecule's trajectory. At the residue level, it is computed by aggregating the solvent-accessible surface areas of all residues, providing insights into the accessibility of individual residues to the solvent. We initially analyzed this feature for each conformation and subsequently for each residue, leading to the creation of the following graphs.

*"bins": Number of bins for the histogram; default is 50.*

*"hist_range": A tuple defining the min and max values for the histogram; if None, uses the data range.*

*"violin_plot": If True, displays a violin plot; default is True.*

*"location": It allows you to specify whether to calculate and use the mean ('mean') or the median ('median') as the reference value.*

*"save": If True, saves the plot in the data directory; default is False.*

*"ax": The matplotlib Axes object on which to plot; if None, creates a new Axes object.*

.. code-block:: python

    visualization.ensemble_sasa(location='mean')

.. image:: images/sh3/global_analysis/output.png
   :align: center
  

  

  
Flory scaling exponents
-------------------------
The following code block is used to calculate and print the Flory scaling exponents for different ensembles.
The Flory exponent, denoted as **ν**, is a parameter that describes the scaling behavior of a polymer chain in a solvent, used to characterize the conformation of chains and is particularly relevant for understanding the compaction of intrinsically disordered regions (IDRs) in proteins. 
It s related to the radius of gyration (Rg) and the end-to-end distance (Ree) of the polymer chain.
An ideal-chain polymer, achieving equilibrium among residue-residue, residue-solvent, and solvent-solvent interactions, exhibits a ν of 0.5, signifying a Gaussian chain structure. Deviations from this value indicate more compact (ν < 0.5) or more extended (ν > 0.5) conformations. 

As detailed in the paper (reference), Flory scaling exponents, ν, were determined by fitting mean-squared residue-residue distances, R⟨ij2⟩, calculated for sequential separations greater than five residues along the linear sequence.
Moreover, this analysis underscores the role of ν in elucidating the compaction of IDRs, revealing correlations with biological functions and cellular localizations of full-length proteins: proteins with compact IDRs (lower ν values) often participate in crucial functions like binding chromatin and DNA cis-regulatory sequences, suggesting a pivotal role for IDR compaction in protein functionality and phase behavior.







.. code-block:: python

    v_values = analysis.get_features("flory_exponent")
    for code in v_values:
    print(f"{code}: {v_values[code]:.4f}")

.. image:: images/sh3/global_analysis/flory.png
   :align: center
   :scale: 60%

Summary
----------

This code snippet calculates and displays a summary of features for each analyzed dataset. The function *get_features_summary_dataframe* is used to create a summary DataFrame that includes information about the selected key parameters. 

In the provided example, the following parameters are selected: radius of gyration (rg), end-to-end distance (end_to_end), end-to-end distance to radius of gyration ratio (ee_on_rg), and Flory exponent (flory_exponent). This DataFrame is then displayed using the display(summary) statement.

.. code-block:: python

    summary = analysis.get_features_summary_dataframe(
    selected_features=["rg", "end_to_end", "ee_on_rg", "flory_exponent"],
    show_variability=False
     )
    display(summary)

.. image:: images/sh3/global_analysis/summary.png
   :align: center
