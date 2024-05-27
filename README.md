# IDPensembleTools : an open-source library for analysis of conformational ensembles of disordered proteiens 

## Overview
IDPensembleTools is a python package by which you can load and analyze multiple conformational ensembles in different formats such as (`pdb`, `xtc`,`dcd`,...)

## Features 
With IDPensembleTools, you can:

- Extract and visualize **global features** of structural ensembles such as (`Rg`, `Asphericity`, `Prolateness`, `End to end distance`,...)
- Extract and visualize **local features** of structural ensembles sucha as ( `intera-atomic distances`, `phi-psi angles` , `alpha-helix content`, ...)
- Performing dimensionality reduction methods (`PCA`, `UMAP`, `t-SNE`) on different extracted features of structural ensemble
- Comparing structural ensembles of disordered proteins using different similarity scores such as Kullback-Leibler (KL) and Jensen-Shannon (JS) divergence methods and visualize similarity matrix
  
For details, you can check our user-centric documentation , **put the link to the documentation**, is generated from our repository using Sphinx. 

## Installation 
- Using pip:
Using python package managment systme "pip" you can easily install IDPensembleTools (and its dependencies if needed), upgrade or uninstall with just one terminal command. 

`pip install dpet (Not sure about the names)`<br>
`pip install --upgrade dpet`<br>
`pip uninstall dpet`

- From source:
  
```bash
git clone https://github.com/hamidrgh/EnsembleTools.git
cd EnsembleTools
python setup.py install
```
- Using Conda:

The instruction on how to use conda for the installation should be put here. 

## Python Requirements and Dependencies

- Recommend specific version of Python
- Point to the dependencies (Numpy, Sklearn, pandas, matplotlib, mdtraj, ...)

## Documentation
  
For details, you can check our user-centric documentation , **put the link to the documentation**, is generated from our repository using Sphinx. 

## Citation 
The link to the publication should be put here. 

## License
MIT License



