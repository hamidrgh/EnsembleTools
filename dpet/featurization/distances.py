import numpy as np
import mdtraj


ca_selector = "protein and name CA"

def calc_ca_dmap(traj: mdtraj.Trajectory):
    ca_ids = traj.topology.select(ca_selector)
    ca_xyz = traj.xyz[:, ca_ids]
    dmap = np.sqrt(
        np.sum(np.square(ca_xyz[:, None, :, :] - ca_xyz[:, :, None, :]), axis=3)
    )
    return dmap


def featurize_ca_dist(
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