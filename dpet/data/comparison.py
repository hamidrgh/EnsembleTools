import numpy as np
from dpet.featurization.distances import *
from dpet.featurization.angles import *

def score_kld_approximation(v_true, v_pred, n_bins=50, pseudo_c=0.001):
    """
    Scores an approximation of KLD by discretizing the values in
    'v_true' (data points from a reference distribution) and 'v_pred'
    (data points from a predicted distribution).
    """
    # Define bins.
    _min = min((v_true.min(), v_pred.min()))
    _max = max((v_true.max(), v_pred.max()))
    bins = np.linspace(_min, _max, n_bins + 1)
    # Compute the frequencies in the bins.
    ht = (np.histogram(v_true, bins=bins)[0] + pseudo_c) / v_true.shape[0]
    hp = (np.histogram(v_pred, bins=bins)[0] + pseudo_c) / v_pred.shape[0]
    kl = -np.sum(ht * np.log(hp / ht))
    return kl, bins


def score_jsd_approximation(v_true, v_pred, n_bins=50, pseudo_c=0.001):
    """
    Scores an approximation of JS by discretizing the values in
    'v_true' (data points from a reference distribution) and 'v_pred'
    (data points from a predicted distribution).
    """
    # Define bins.
    _min = min((v_true.min(), v_pred.min()))
    _max = max((v_true.max(), v_pred.max()))
    bins = np.linspace(_min, _max, n_bins + 1)
    # Compute the frequencies in the bins.
    ht = (np.histogram(v_true, bins=bins)[0] + pseudo_c) / v_true.shape[0]
    hp = (np.histogram(v_pred, bins=bins)[0] + pseudo_c) / v_pred.shape[0]
    hm = (ht + hp) / 2
    kl_tm = -np.sum(ht * np.log(hm / ht))
    kl_pm = -np.sum(hp * np.log(hm / hp))
    js = 0.5 * kl_pm + 0.5 * kl_tm
    return js, bins


def score_akld_d(
    traj_ref: mdtraj.Trajectory,
    traj_hat: mdtraj.Trajectory,
    n_bins: int = 50,
    method: str = "js",
):
    """
    See the idpGAN article.
    """
    # Calculate distance maps.
    dmap_ref = calc_ca_dmap(traj_ref)
    dmap_hat = calc_ca_dmap(traj_hat)
    if dmap_ref.shape[1] != dmap_hat.shape[1]:
        raise ValueError(
            "Input trajectories have different number of residues:"
            f" ref={dmap_ref.shape[1]}, hat={dmap_hat.shape[1]}"
        )
    n_akld_d = []
    if method == "kl":
        score_func = score_kld_approximation
    elif method == "js":
        score_func = score_jsd_approximation
    else:
        raise KeyError(method)
    for i in range(dmap_ref.shape[1]):
        for j in range(dmap_ref.shape[1]):
            if i + 1 >= j:
                continue
            kld_d_ij = score_func(dmap_ref[:, i, j], dmap_hat[:, i, j], n_bins=n_bins)[
                0
            ]
            n_akld_d.append(kld_d_ij)
    return np.mean(n_akld_d), n_akld_d


def score_akld_t(
    traj_ref: mdtraj.Trajectory,
    traj_hat: mdtraj.Trajectory,
    n_bins: int = 50,
    method: str = "js",
):
    """
    Similar to 'score_akld_d', but evaluate alpha torsion angles.
    """
    # Calculate distance maps.
    tors_ref = featurize_a_angle(traj_ref)
    tors_hat = featurize_a_angle(traj_hat)
    if tors_ref[0].shape[1] != tors_hat[0].shape[1]:
        raise ValueError(
            "Input trajectories have different number torsion angles:"
            f" ref={tors_ref.shape[1]}, hat={tors_hat.shape[1]}"
        )
    n_akld_t = []
    if method == "kl":
        score_func = score_kld_approximation
    elif method == "js":
        score_func = score_jsd_approximation
    else:
        raise KeyError(method)
    for i in range(tors_ref[0].shape[1]):
        score_i = score_func(tors_ref[0][:, i], tors_hat[0][:, i], n_bins=n_bins)[0]
        n_akld_t.append(score_i)
    return np.mean(n_akld_t), n_akld_t
