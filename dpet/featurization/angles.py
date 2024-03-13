import numpy as np
import mdtraj


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