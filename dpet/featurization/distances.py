import numpy as np
import mdtraj
from dpet.data.topology import slice_traj_to_com


ca_selector = "protein and name CA"

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


def _featurize_dist(
        traj: mdtraj.Trajectory,
        seq_sep: int = 2,
        inverse: bool = False,
        get_names: bool = True):
    # Get all C-alpha indices.
    ca_ids = traj.topology.select(ca_selector)
    atoms = list(traj.topology.atoms)
    # Get all pair of ids.
    pair_ids = []
    names = []
    for i, id_i in enumerate(ca_ids):
        for j, id_j in enumerate(ca_ids):
            if j - i >= seq_sep:
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

def featurize_ca_dist(
        traj: mdtraj.Trajectory,
        seq_sep: int = 2,
        inverse: bool = False,
        get_names: bool = True):
    return _featurize_dist(traj=traj,
                           seq_sep=seq_sep,
                           inverse=inverse,
                           get_names=get_names)

def featurize_com_dist(
        traj: mdtraj.Trajectory,
        seq_sep: int = 2,
        inverse: bool = False,
        get_names: bool = True):
    traj = slice_traj_to_com(traj)
    return _featurize_dist(traj=traj,
                           seq_sep=seq_sep,
                           inverse=inverse,
                           get_names=get_names)