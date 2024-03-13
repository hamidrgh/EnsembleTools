# Disordered Protein Ensemble Tools

## About
Analyze 3D structural ensembles of disordered proteins with Python. Add more information.

## Installation
Describe how people can install it. I think we have two options to facilitate installation:
- Let users create a dedicated conda environment (this is what I usually do in the "Installation" part of my repositories)
- Let users download a pip package (i have no experience with this, but it should be quite easy to make one)

we could probably implement both and let users decide. But providing only one is also good. Installing a package using these methods typically requires 2-3 bash commands at most, I think easy-to-install is what matters for users.
### Requirements (optional)
Some journals ask for this section. Things like:
* OS: Linux/MacOS
* Python: >= 3.8
* etc...
### For developers only (should be removed from the final version)
To use the package in any directory (without having to perform any installation), simply add to your `.bashrc` file:
```bash
PYTHONPATH="$PYTHONPATH:/path/to/the/root/dir/of/this/repo"
```

so something like:

```bash
PYTHONPATH="$PYTHONPATH:/home/giacomo/projects/ensemble_analysis/git/EnsembleTools"
```

## User Guide
### Notebooks
Illustrate what notebooks we have and what they can be used for.
### Documentation
I think we should add a link to a complete documentation here, which illustrates at least the most useful features, classes and functions. Examples:
- [mdtraj analysis reference](https://www.mdtraj.org/1.9.8.dev0/analysis.html)
- [deeptime documentation](https://deeptime-ml.github.io/latest/index.html)
- [pyemma references](http://emma-project.org/latest/api/index_coor.html)
- [prody reference manual](http://prody.csb.pitt.edu/manual/reference/index.html)
it might take a little bit of time to set everything up here, but I think we can safely do this at the end.

Probably is not strictly necessary for publication, but it could help a bit.

There are also packages for doing this automatically, like (Sphinx)[https://www.sphinx-doc.org/en/master/tutorial/getting-started.html#setting-up-your-project-and-development-environment]. These packages parse the docstrings of methods and classes, this is why i am trying to insert the right amount of documentation. We don't need to do this for **all** methods and classes, only for the ones that users are most likely to use.

## Reference
Once a pre-print or article is submitted, this part should be updated.
Optional: also link to other useful resources for this project (PED, DisProt, mdtraj?).

## Contacts
Contacts go here.