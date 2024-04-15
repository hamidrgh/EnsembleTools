from typing import Union
import numpy as np
import mdtraj
from dpet.data.topology import slice_traj_to_com
from dpet.featurization.utils import get_max_sep


#--------------------------------------------------------------------
# Calculate (N, L, L) distance maps. Mostly used for visualization. -
#--------------------------------------------------------------------

ca_selector = "name CA" # "protein and name CA" is not working  for ensembles contain phopshorylation residues

def _calc_dmap(traj: mdtraj.Trajectory):
    ca_ids = traj.topology.select(ca_selector)
    ca_xyz = traj.xyz[:, ca_ids]
    dmap = np.sqrt(
        np.sum(np.square(ca_xyz[:, None, :, :] - ca_xyz[:, :, None, :]), axis=3)
    )
    return dmap

def calc_ca_dmap(traj: mdtraj.Trajectory):
    return _calc_dmap(traj=traj)

def calc_com_dmap(traj: mdtraj.Trajectory):
    traj = slice_traj_to_com(traj)
    return _calc_dmap(traj=traj)


#---------------------------------------------------------------------
# Calculate (N, *) distance features. Mostly used for featurization. -
#---------------------------------------------------------------------

def _featurize_dist(
        traj: mdtraj.Trajectory,
        min_sep: int = 2,
        max_sep: Union[None, int, float] = None,
        inverse: bool = False,
        get_names: bool = True
    ):
    # Get all C-alpha indices.
    ca_ids = traj.topology.select(ca_selector)
    atoms = list(traj.topology.atoms)
    max_sep = get_max_sep(L=len(atoms), max_sep=max_sep)
    # Get all pair of ids.
    pair_ids = []
    names = []
    for i, id_i in enumerate(ca_ids):
        for j, id_j in enumerate(ca_ids):
            if j - i >= min_sep:
                if j - i > max_sep:
                    continue
                pair_ids.append([id_i, id_j])
                if get_names:
                    names.append(
                        f"{repr(atoms[id_i].residue)}-{repr(atoms[id_j].residue)}"
                    )
    # Calculate C-alpha - C-alpha distances.
    ca_dist = mdtraj.compute_distances(traj=traj, atom_pairs=pair_ids)
    if inverse:
        ca_dist = 1 / ca_dist
    if get_names:
        return ca_dist, names
    else:
        return ca_dist

def seq_sep_deprecation(func):
    """
    NOTE: development only, to be removed when seq_sep has been substituted in
          all the code.
    """
    def inner(seq_sep=None, min_sep=2, *args, **kwargs):
        if seq_sep is not None:
            print(
                f"WARNING: the 'seq_sep' argument in '{func.__name__}' is"
                " deprecated by Giacomo, use min_sep instead! We should remove"
                " seq_sep.")
            min_sep = seq_sep
        return func(min_sep=min_sep, *args, **kwargs)
    return inner

@seq_sep_deprecation
def featurize_ca_dist(
        traj: mdtraj.Trajectory,
        seq_sep: int = None,  # Remove.
        min_sep: int = 2,
        max_sep: int = None,
        inverse: bool = False,
        get_names: bool = True):
    return _featurize_dist(traj=traj,
                           min_sep=min_sep,
                           max_sep=max_sep,
                           inverse=inverse,
                           get_names=get_names)

@seq_sep_deprecation
def featurize_com_dist(
        traj: mdtraj.Trajectory,
        seq_sep: int = None,  # Remove.
        min_sep: int = 2,
        max_sep: int = None,
        inverse: bool = False,
        get_names: bool = True):
    traj = slice_traj_to_com(traj)
    return _featurize_dist(traj=traj,
                           min_sep=min_sep,
                           max_sep=max_sep,
                           inverse=inverse,
                           get_names=get_names)