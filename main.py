import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mdtraj
import random
from ensemble_analysis_lib import EnsembleAnalysis
import streamlit as st


st.title("Ensemble Analysis")
ens_codes = [
    "pdbfile",
    "PED00160",
]
# Directory with the PDB files.
pdb_dp = "/Users/amini_m/Desktop/repo/EnsembleTools/"

# Featurization options.
featurization = "phi_psi"  # choices: "ca_dist", "phi_psi"
featurization_params = {
    "ca_dist": {"seq_sep": 2, "normalize": True},
    "phi_psi": {},
    "a_angle": {},
}


# Dimensionality reduction options. 'tsne' or 'pca' or 'dimenfix'
reduce_dim_method = "tsne"
reduce_dim_params = {
    "pca": {"num_dim": None},
    "tsne": {
        "perplexityVals": range(500, 1001, 500),
        "metric": "euclidean",
        "dir": "/home/hamid/PED_Data_analysis/random_ensemble/",
    },
    "dimenfix": {},
}


pipeline = EnsembleAnalysis(pdb_dp=pdb_dp, ens_codes=ens_codes)

st.write(
    pipeline.featurize(
        featurization=featurization,
        featurization_params=featurization_params[featurization],
    )
)