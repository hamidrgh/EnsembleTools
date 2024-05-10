from typing import List, Tuple, Union
import numpy as np
import mdtraj
from dpet.data.topology import slice_traj_to_com
from dpet.featurization.utils import get_max_sep


#--------------------------------------------------------------------
# Calculate (N, L, L) distance maps. Mostly used for visualization. -
#--------------------------------------------------------------------

ca_selector = "protein and name CA"

def _calc_dmap(traj: mdtraj.Trajectory):
    ca_ids = traj.topology.select(ca_selector)
    ca_xyz = traj.xyz[:, ca_ids]
    dmap = np.sqrt(
        np.sum(np.square(ca_xyz[:, None, :, :] - ca_xyz[:, :, None, :]), axis=3)
    )
    return dmap

def calc_ca_dmap(traj: mdtraj.Trajectory) -> np.ndarray:
    """
    Calculate the (N, L, L) distance maps between C-alpha atoms for visualization.

    Parameters
    ----------
    traj : mdtraj.Trajectory
        The MDtraj trajectory object.

    Returns
    -------
    dmap : numpy.ndarray
        The distance maps of shape (N, L, L), where N is the number of frames and L is the number of C-alpha atoms.

    Notes
    -----
    This function calculates the distance maps between C-alpha atoms for visualization purposes.
    """
    return _calc_dmap(traj=traj)

def calc_com_dmap(traj: mdtraj.Trajectory) -> np.ndarray:
    """
    Calculate the (N, L, L) distance maps between center of mass (COM) atoms for visualization.

    Parameters
    ----------
    traj : mdtraj.Trajectory
        The MDtraj trajectory object.

    Returns
    -------
    dmap : numpy.ndarray
        The distance maps of shape (N, L, L), where N is the number of frames and L is the number of center of mass (COM) atoms.

    Notes
    -----
    This function calculates the distance maps between center of mass (COM) atoms for visualization purposes.
    """
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
        get_names: bool = True,
        atom_selector: str = "name == CA"
    ):
    # Get all C-alpha indices.
    ca_ids = traj.topology.select(atom_selector)
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

def featurize_ca_dist(
        traj: mdtraj.Trajectory,
        min_sep: int = 2,
        max_sep: int = None,
        inverse: bool = False,
        get_names: bool = True,
        atom_selector: str = "name CA") -> Union[np.ndarray, Tuple[np.ndarray, List[str]]]:
    """
    Calculate C-alpha distances between pairs of residues.

    Parameters
    ----------
    traj : mdtraj.Trajectory
        The MDtraj trajectory object.
    min_sep : int, optional
        The minimum sequence separation for distance calculations. Default is 2.
    max_sep : int or None, optional
        The maximum sequence separation for distance calculations. Default is None.
    inverse : bool, optional
        Whether to calculate inverse distances. Default is False.
    get_names : bool, optional
        Whether to return the names of the calculated features. Default is True.
    atom_selector : str, optional
        The atom selection string. Default is "name CA".

    Returns
    -------
    distances : numpy.ndarray or tuple
        The calculated C-alpha distances. If get_names is True, returns a tuple containing distances and corresponding feature names.

    Notes
    -----
    This function calculates C-alpha distances between pairs of residues.
    """
    return _featurize_dist(traj=traj,
                           min_sep=min_sep,
                           max_sep=max_sep,
                           inverse=inverse,
                           get_names=get_names,
                           atom_selector=atom_selector)

def featurize_com_dist(
        traj: mdtraj.Trajectory,
        min_sep: int = 2,
        max_sep: int = None,
        inverse: bool = False,
        get_names: bool = True,
        atom_selector: str = "name == CA") -> Union[np.ndarray, Tuple[np.ndarray, List[str]]]:
    """
    Calculate center of mass (COM) distances between pairs of residues.

    Parameters
    ----------
    traj : mdtraj.Trajectory
        The MDtraj trajectory object.
    min_sep : int, optional
        The minimum sequence separation for distance calculations. Default is 2.
    max_sep : int or None, optional
        The maximum sequence separation for distance calculations. Default is None.
    inverse : bool, optional
        Whether to calculate inverse distances. Default is False.
    get_names : bool, optional
        Whether to return the names of the calculated features. Default is True.
    atom_selector : str, optional
        The atom selection string. Default is "name == CA".

    Returns
    -------
    distances : numpy.ndarray or tuple
        The calculated center of mass (COM) distances. If get_names is True, returns a tuple containing distances and corresponding feature names.

    Notes
    -----
    This function calculates center of mass (COM) distances between pairs of residues.
    """
    traj = slice_traj_to_com(traj)
    return _featurize_dist(traj=traj,
                           min_sep=min_sep,
                           max_sep=max_sep,
                           inverse=inverse,
                           get_names=get_names,
                           atom_selector=atom_selector)