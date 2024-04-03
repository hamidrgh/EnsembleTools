from typing import List, Tuple, Union
import numpy as np
import mdtraj
from dpet.featurization.utils import get_triu_indices


#------------------------------------------
# Commonly used protein structure angles. -
#------------------------------------------

def featurize_phi_psi(
        traj: mdtraj.Trajectory,
        get_names: bool = True,
        ravel: bool = True):
    if not ravel and get_names:
        raise ValueError("Cannot use ravel when returning feature names")

    phi_ids, phi = mdtraj.compute_phi(traj)
    psi_ids, psi = mdtraj.compute_psi(traj)
    if ravel:  # Will return a (N, (L-1)*2) array.
        phi_psi = np.concatenate([phi, psi], axis=1)
    else:  # Will return a (N, L-1, 2) array.
        phi_psi = np.concatenate([phi[...,None], psi[...,None]], axis=-1)

    if get_names:
        atoms = list(traj.topology.atoms)
        names = []
        for t in phi_ids:
            names.append(repr(atoms[t[1]].residue) + "-PHI")
        for t in psi_ids:
            names.append(repr(atoms[t[0]].residue) + "-PSI")
        return phi_psi, names
    else:
        return phi_psi


def featurize_a_angle(traj: mdtraj.Trajectory, get_names: bool = True):
    """
    # Get all C-alpha indices.
    ca_ids = traj.topology.select("protein and name CB")
    atoms = list(traj.topology.atoms)
    # Get all pair of ids.
    tors_ids = []
    names = []
    for i in range(ca_ids.shape[0] - 3):
        if get_names:
            names.append(
                "{}:{}:{}:{}".format(
                    atoms[ca_ids[i]],
                    atoms[ca_ids[i + 1]],
                    atoms[ca_ids[i + 2]],
                    atoms[ca_ids[i + 3]],
                )
            )
        tors_ids.append((ca_ids[i], ca_ids[i + 1], ca_ids[i + 2], ca_ids[i + 3]))
    tors = mdtraj.compute_dihedrals(traj, tors_ids)
    if get_names:
        return tors, names
    else:
        return tors
    """
    raise NotImplementedError()


#-------------------------------------------------------------------------------
# trRosetta angles. Code mostly adapted from:                                  -
# https://github.com/RosettaCommons/trRosetta2/blob/main/trRosetta/coords6d.py -
#-------------------------------------------------------------------------------

def featurize_tr_angle(
        traj: mdtraj.Trajectory,
        type: str,
        min_sep: int = 2,
        max_sep: Union[None, int, float] = None,
        ravel: bool = True,
        get_names: bool = True
    ) -> np.array:
    """
    Calculate trRosetta angles between pair of residue. See the original
    trRosetta paper (https://pubmed.ncbi.nlm.nih.gov/31896580/) for more
    information.

    arguments:
        traj: input mdtraj trajectory.
        type: string with the type of angle feature. "omega" for omega torsion
            angles, "phi" for phi angles.
        min_sep: minimum sequence separation between a pair of residues.
        max_sep: max sequence separation between a pair of residues. If `None`,
            there will be no maximum separation. If it is a float, the maximum
            separation will be `int(max_sep*L)` where `L` is the number of amino
            acids in the protein.
        ravel: if `True`, values from all the pair of residues indexed via the
            `min_sep`, `max_sep` will be flattened in a vector, so the function
            will return a `(N, *)` array, where `N` is the number of conformers
            in the trajectory and `*` will be the number of features being
            indexed. If `False`, a (N, L, L) array will be returned.
        get_names: when `ravel` is `True`, also return the names of the
            corresponding features.
    returns:
        angles: numpy array storing the angle values.
    """

    if type == "omega":
        angles = featurize_tr_omega(
            traj=traj,
            min_sep=min_sep,
            max_sep=max_sep,
            ravel=ravel,
            get_names=get_names
        )
    elif type == "phi":
        angles = featurize_tr_phi(
            traj=traj,
            min_sep=min_sep,
            max_sep=max_sep,
            ravel=ravel,
            get_names=get_names
        )
    else:
        raise KeyError(type)
    return angles


def featurize_tr_omega(
        traj: mdtraj.Trajectory,
        min_sep: int = 2,
        max_sep: int = None,
        ravel: bool = True,
        get_names: bool = True
    ):
    """
    Calculate omega angles from trRosetta. Those are torsion angles defined
    between a pair of residues `i` and `j` and involving the following atoms:
        `Ca(i) -- Cb(i) -- Cb(j) -- Ca(j)`
    if a residue does not have a Cb atom, an pseudo-Cb will be added
    automatically.
    """

    if not ravel and get_names:
        raise ValueError("Cannot use ravel when returning feature names")
    
    # Get atom indices.
    residues, atoms = _get_tr_topology_data(traj)
    
    # Calculate the angles.
    omega = np.zeros((traj.xyz.shape[0], len(residues), len(residues)))
    for i in range(len(residues)):
        pos_ca_i, pos_cb_i = _get_tr_omega_coords(traj=traj,
                                                  atoms=atoms[i])
        for j in range(len(residues)):
            # trRosetta omega angles matrix is symmetric, so we can compute only
            # the upper triangle part.
            if i < j:
                pos_ca_j, pos_cb_j = _get_tr_omega_coords(traj=traj,
                                                          atoms=atoms[j])
                omega_ij = get_dihedrals(pos_ca_i, pos_cb_i, pos_cb_j, pos_ca_j)
                omega[:,i,j] = omega_ij
                omega[:,j,i] = omega_ij
                
    if ravel:  # Will return a (N, *) array.
        triu_ids = get_triu_indices(L=len(residues),
                                    min_sep=min_sep,
                                    max_sep=max_sep)
        
        omega = omega[:,triu_ids[0], triu_ids[1]]
    else:  # Will return a (N, L, L) array.
        return omega

    if get_names:
        names = []
        for i, j in zip(*triu_ids):
            names.append(f"{repr(residues[i])}-{repr(residues[j])}")
        return omega, names
    else:
        return omega


def featurize_tr_phi(
        traj: mdtraj.Trajectory,
        min_sep: int = 2,
        max_sep: int = None,
        ravel: bool = True,
        get_names: bool = True
    ):
    """
    Calculate phi angles from trRosetta. Those are angles defined between a
    pair of residues `i` and `j` and involve the following atoms:
        `Ca(i) -- Cb(i) -- Cb(j)`
        `Ca(j) -- Cb(j) -- Cb(i)`
    if a residue does not have a Cb atom, an pseudo-Cb will be added
    automatically.
    """

    if not ravel and get_names:
        raise ValueError("Cannot use ravel when returning feature names")
    
    # Get atom indices.
    residues, atoms = _get_tr_topology_data(traj)
    
    # Calculate the angles.
    phi = np.zeros((traj.xyz.shape[0], len(residues), len(residues)))
    for i in range(len(residues)):
        pos_ca_i, pos_cb_i = _get_tr_omega_coords(traj=traj,
                                                  atoms=atoms[i])
        for j in range(len(residues)):
            # trRosetta omega angles matrix is asymmetric, so we need to compute
            # both the upper and lower trinagles.
            if i != j:
                pos_ca_j, pos_cb_j = _get_tr_omega_coords(traj=traj,
                                                          atoms=atoms[j])
                phi_ij = get_angles(pos_ca_i, pos_cb_i, pos_cb_j)
                phi_ji = get_angles(pos_ca_j, pos_cb_j, pos_cb_i)
                phi[:,i,j] = phi_ij
                phi[:,j,i] = phi_ji
                
    if ravel:  # Will return a (N, *) array.
        triu_ids = get_triu_indices(L=len(residues),
                                    min_sep=min_sep,
                                    max_sep=max_sep)
        
        phi = np.concatenate([
                phi[:,triu_ids[0], triu_ids[1]],
                phi[:,triu_ids[1], triu_ids[0]],
            ],
            axis=1
        )
    else:  # Will return a (N, L, L) array.
        return phi

    if get_names:
        names = []
        for i, j in zip(*triu_ids):
            names.append(f"{repr(residues[i])}-{repr(residues[j])}")
        for i, j in zip(*triu_ids):
            names.append(f"{repr(residues[j])}-{repr(residues[i])}")
        return phi, names
    else:
        return phi


def _get_tr_topology_data(traj: mdtraj.Trajectory) -> Tuple[list]:
    residues = [r for r in traj.topology.residues]
    atoms = []
    for i in range(len(residues)):
        ca_i = residues[i].atom("CA")
        try:
            cb_i = residues[i].atom("CB")
            atoms.append({"CA": ca_i, "CB": cb_i})
        except KeyError:
            n_i = residues[i].atom("N")
            c_i = residues[i].atom("C")
            atoms.append({"CA": ca_i, "N": n_i, "C": c_i})
    return residues, atoms


def _get_tr_omega_coords(
        traj: mdtraj.Trajectory,
        atoms: dict) -> Tuple[np.array]:
    """
    Get from an mdtraj trajectory the coordinates of the atoms needed to
    calculate trRosetta angles.
    """
    pos_ca = traj.xyz[:,atoms["CA"].index,:]
    if "CB" in atoms:  # Get the original Cbeta atom.
        pos_cb = traj.xyz[:,atoms["CB"].index,:]
    else:  # Get the coordinates of a pseudo-Cbeta atom.
        pos_n = traj.xyz[:,atoms["N"].index,:]
        pos_c = traj.xyz[:,atoms["C"].index,:]
        # recreate Cb given N,Ca,C
        b = pos_ca - pos_n
        c = pos_c - pos_ca
        a = np.cross(b, c)
        ma = 0.58273431
        mb = 0.56802827
        mc = 0.54067466
        pos_cb = -ma*a + mb*b - mc*c + pos_ca
    return pos_ca, pos_cb


def get_dihedrals(a, b, c, d):
    """
    calculate dihedral angles defined by 4 sets of points.
    """
    b0 = -1.0*(b - a)
    b1 = c - b
    b2 = d - c
    b1 /= np.linalg.norm(b1, axis=-1)[:,None]
    v = b0 - np.sum(b0*b1, axis=-1)[:,None]*b1
    w = b2 - np.sum(b2*b1, axis=-1)[:,None]*b1
    x = np.sum(v*w, axis=-1)
    y = np.sum(np.cross(b1, v)*w, axis=-1)
    return np.arctan2(y, x)

def get_angles(a, b, c):
    """
    Calculate planar angles defined by 3 sets of points.
    """
    v = a - b
    v /= np.linalg.norm(v, axis=-1)[:,None]
    w = c - b
    w /= np.linalg.norm(w, axis=-1)[:,None]
    x = np.sum(v*w, axis=1)
    return np.arccos(x)