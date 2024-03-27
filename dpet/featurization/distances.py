import numpy as np
import mdtraj


ca_selector = "name CA"

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
    
def get_distance_matrix(xyz):

    return np.sqrt(np.sum(np.square(xyz[:,None,:,:]-xyz[:,:,None,:]), axis=3))

def get_contact_map(dmap, threshold =0.45, pseudo_count=0.01):
    n= dmap.shape[0]
    cmap = ((dmap <= threshold).astype(int).sum(axis=0)+pseudo_count)/n
    return cmap

def get_contact_map_ensemble(dict_traj):
    contact_ens_dict = {}
    distance_matrix_ens_dict = {}
    for ens in dict_traj:
        xyz_ca_ens = dict_traj[ens].xyz[:,dict_traj[ens].topology.select(ca_selector)]
        distance_matrix_ens_dict[ens] = get_distance_matrix(xyz_ca_ens)
        contact_ens_dict[ens] = get_contact_map(distance_matrix_ens_dict[ens])
    return contact_ens_dict, distance_matrix_ens_dict


def contact_probability_map(traj, threshold = 0.8):
    distances = mdtraj.compute_contacts(traj, scheme="ca")[0]
    res_pair = mdtraj.compute_contacts(traj, scheme="ca")[1]
    contact_distance = mdtraj.geometry.squareform(distances, res_pair)
    matrix_contact_prob = np.zeros(contact_distance.shape)
    threshold = threshold
    for ens in range(contact_distance.shape[0]):
        for i in range(contact_distance.shape[1]):
            for j in range(contact_distance.shape[2]):
                if contact_distance[ens][i][j] < threshold:
                    matrix_contact_prob[ens][i][j] = 1
                else:
                    matrix_contact_prob[ens][i][j] = 0
    matrix_prob_avg = np.mean(matrix_contact_prob, axis=0)
    return matrix_prob_avg