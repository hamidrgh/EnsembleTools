"""
Calculate features at the ensemble level.
"""

from typing import Tuple
import numpy as np
from scipy.optimize import curve_fit
import mdtraj


ensemble_features = ("flory_exponent", )

def calc_flory_scaling_exponent(traj: mdtraj.Trajectory) -> Tuple[float]:
    """
    Calculate the apparent Flory scaling exponent in an ensemble. Code adapted
    from:
        https://github.com/KULL-Centre/_2023_Tesei_IDRome

    Arguments
    ----------
    traj : mdtraj.Trajectory
        Input trajectory object.

    Returns
    -------
    resuts: tuple
        Tuple containing the nu (Flory scaling exponent), error on nu, R0 and
        error on R0. All values are calculated by fitting the internal scaling
        profiles. For more information see https://pubmed.ncbi.nlm.nih.gov/38297118/
        and https://pubs.acs.org/doi/full/10.1021/acs.jpcb.3c01619.
    """
    # Compute all Ca-Ca distances.
    ca_ids = [a.index for a in traj.top.atoms if a.name == "CA"]
    ca_map = {ai: i for (i, ai) in enumerate(ca_ids)}
    nres = len(ca_ids)
    if nres < 6:
        raise ValueError(
            f"Chain is too short ({nres} residues) to compute Flory exponent")
    pairs = traj.top.select_pairs(ca_ids, ca_ids)
    d = mdtraj.compute_distances(traj, pairs)

    # Compute sqrt(mean(d_ij**2)) for each |i-j| value.
    ij = np.arange(2, nres, 1)
    diff = np.array([ca_map[x[1]]-ca_map[x[0]] for x in pairs])
    dij = np.empty(0)
    for i in ij:  # Collect data for each |i-j| value.
        dij = np.append(dij, np.sqrt((d[:, diff == i]**2).mean()))

    # Fit the following function, to find R0 and nu:
    #     sqrt(mean(d_ij**2)) = R0*|i-j|**nu
    f = lambda x,R0,v : R0*np.power(x, v)
    popt, pcov = curve_fit(f, ij[ij > 5], dij[ij > 5], p0=[.4, .5])

    # Return values.
    nu = popt[1]
    nu_err = pcov[1, 1]**0.5
    R0 = popt[0]
    R0_err = pcov[0, 0]**0.5
    return nu, nu_err, R0, R0_err