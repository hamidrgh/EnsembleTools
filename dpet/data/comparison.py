import math
from typing import Union, List, Tuple
import numpy as np
import scipy
from dpet.ensemble import Ensemble


num_default_bins = 50
min_samples_auto_hist = 2500

def get_num_comparison_bins(
        bins: Union[str, int],
        x: List[np.ndarray] = None
    ):
    """
    Get the number of bins to be used in comparison between two ensembles using
    an histogram-based score (such as a JSD approximation).

    Parameters
    ----------
    bins: Union[str, int]
        Determines the number of bins to be used. When providing an `int`, the
        same value will simply be returned. When providing a string, the following
        rules to determine bin value will be applied:
        `auto`: applies `sqrt` if the size of the smallest ensemble is <
            `dpet.data.comparison.min_samples_auto_hist`. If it >= than this
            value, returns `dpet.data.comparison.num_default_bins`.
        `sqrt`: applies the square root rule for determining bin number using
            the size of the smallest ensemble (https://en.wikipedia.org/wiki/Histogram#Square-root_choice).
        `sturges`: applies Sturge's formula for determining bin number using
            the size of the smallest ensemble (https://en.wikipedia.org/wiki/Histogram#Sturges's_formula).

    x: List[np.ndarray], optional
        List of M feature matrices (one for each ensembles) of shape (N_i, *).
        N_i values are the number of structures in each ensemble. The minimum
        N_i will be used to apply bin assignment rule when the `bins` argument
        is a string.

    Returns
    -------
    num_bins: int
        Number of bins.

    """

    # Apply a rule to define bin number.
    if isinstance(bins, str):
        # Get minimum ensemble size.
        if x is None:
            raise ValueError()
        min_n = min([xi.shape[0] for xi in x])
        if bins == "auto":
            # If minimum ensemble size is larger than a threshold, use a
            # pre-defined bin number.
            if min_n >= min_samples_auto_hist:
                num_bins = num_default_bins
            # Otherwise, apply square root rule.
            else:
                num_bins = sqrt_rule(min_n)
        elif bins == "sqrt":
            num_bins = sqrt_rule(min_n)
        elif bins == "sturges":
            num_bins = sturges_rule(min_n)
        else:
            raise KeyError(bins)
    # Directly use a certain bin number.
    elif isinstance(bins, int):
        num_bins = bins
    else:
        raise KeyError(bins)
    return num_bins

def sturges_rule(n):
    return math.ceil(math.log(n, 2) + 1)

def sqrt_rule(n):
    return math.ceil(math.sqrt(n))

def check_feature_matrices(func):
    def wrapper(m1, m2, *args, **kwargs):
        if len(m1.shape) != 2 or len(m2.shape) != 2:
            raise ValueError()
        if m2.shape[1] != m2.shape[1]:
            raise ValueError()
        return func(m1, m2, *args, **kwargs)
    return wrapper


def score_jsd_approximation(
        x_1: np.ndarray,
        x_2: np.ndarray,
        bins: Union[int, str] = "auto",
        pseudo_c: float = 0.001,
        return_bins: bool = False
    ) -> Union[float, Tuple[float, np.ndarray]]:
    """
    Scores an approximation of Jensen-Shannon divergence by discretizing in a
    histogram the values in the two 1d samples provided as input. For more
    information, see S5 Text of [https://doi.org/10.1371/journal.pcbi.1012144].

    Parameters
    ----------
    x_1, x_2: np.ndarray
        NumPy arrays of shape (*, ) containing samples from two mono-dimensional
        distribution to be compared.
    bins: Union[int, str], optional
        Determines the number of bins to be used when constructing histograms.
        See `dpet.data.comparison.get_num_comparison_bins` for more information.
    pseudo_c: float, optional
        Pseudo-count value used in estimating the probabilities of the bins.
    return_bins: bool, optional
        If `True`, returns the number of bins used in the calculation.

    Returns
    -------
    results: Union[float, Tuple[float, np.ndarray]]
        If `return_bins` is `False`, only returns a float value for the JSD
        score. If `return_bins` is `True`, returns a tuple with the JSD score
        and the number of bins.

    """
    # Define bins.
    n_1 = x_1.shape[0]
    n_2 = x_2.shape[0]
    num_bins = get_num_comparison_bins(bins, x=[x_1, x_2])
    _min = min((x_1.min(), x_2.min()))
    _max = max((x_1.max(), x_2.max()))
    bins = np.linspace(_min, _max, num_bins + 1)

    # Compute the frequencies in the bins.
    ht = (np.histogram(x_1, bins=bins)[0] + pseudo_c) / n_1
    hp = (np.histogram(x_2, bins=bins)[0] + pseudo_c) / n_2
    hm = (ht + hp) / 2
    kl_tm = -np.sum(ht * np.log(hm / ht))
    kl_pm = -np.sum(hp * np.log(hm / hp))
    js = 0.5 * kl_pm + 0.5 * kl_tm

    if return_bins:
        return js, bins
    else:
        return js
    
@check_feature_matrices
def score_avg_jsd(
        features_1: np.ndarray,
        features_2: np.ndarray,
        bins: Union[int, str] = 25,
        return_bins: bool = False,
        *args, **kwargs
    ):
    """
    Takes as input two (*, F) feature matrices and an average JSD score over all
    F features.

    Parameters
    ----------
    features_1, features_2: np.ndarray
        NumPy arrays of shape (*, F) containing two ensembles with * samples
        described by F features. The number of samples in the two ensembles can
        be different.
    bins: Union[int, str], optional
        Determines the number of bins to be used when constructing histograms.
        See `dpet.data.comparison.get_num_comparison_bins` for more information.
    return_bins: bool, optional
        If `True`, returns the number of bins used in the calculation.

    Returns
    -------
    results: Union[float, Tuple[float, np.ndarray]]
        If `return_bins` is `False`, only returns a float value for the average
        JSD score over the F features. If `return_bins` is `True`, returns a
        tuple with the average JSD score and the number of bins used in the
        comparisons.

    """

    _bins = get_num_comparison_bins(bins, x=[features_1, features_2])
    jsd = []
    for l in range(features_1.shape[1]):
        jsd_l = score_jsd_approximation(
            features_1[:,l],
            features_2[:,l],
            bins=_bins,
            return_bins=False,
            *args, **kwargs
        )
        jsd.append(jsd_l)
    avg_jsd =  sum(jsd)/len(jsd)
    if not return_bins:
        return avg_jsd
    else:
        return avg_jsd, _bins

def score_ajsd_d(
        ens_1: Ensemble,
        ens_2: Ensemble,
        bins: Union[str, int],
        return_bins: bool = False,
        featurization_params: dict = {},
        *args, **kwargs
    ):
    """
    Utility function to calculate the aJSD_d score between two ensembles, as
    described in the S5 Text of [https://doi.org/10.1371/journal.pcbi.1012144].
    The score evaluates the divergence between distributions of Ca-Ca distances
    of two ensembles.

    Parameters
    ----------
    ens_1, ens_2: Ensemble
        Two Ensemble objects storing the ensemble data to compare.

    Remaining arguments and output
    ------------------------------
        See `dpet.data.comparison.score_avg_jsd` for more information.

    """
    # Calculate Ca-Ca distances.
    ca_dist_1 = ens_1.get_features(
        featurization="ca_dist",
        min_sep=featurization_params.get("min_sep", 2),
        max_sep=featurization_params.get("max_sep")
        )
    ca_dist_2 = ens_2.get_features(
        featurization="ca_dist",
        min_sep=featurization_params.get("min_sep", 2),
        max_sep=featurization_params.get("max_sep")
        )
    # Compute average JSD approximation.
    results = score_avg_jsd(
        ca_dist_1, ca_dist_2, bins=bins, return_bins=return_bins, *args, **kwargs
    )
    return results

def score_ajsd_t(
        ens_1: Ensemble,
        ens_2: Ensemble,
        bins: Union[str, int],
        return_bins: bool = False,
        *args, **kwargs
    ):
    """
    Utility function to calculate the aJSD_d score between two ensembles, as
    described in the S5 Text of [https://doi.org/10.1371/journal.pcbi.1012144].
    The score evaluates the divergence between distributions of alpha angles of
    two ensembles.

    Parameters
    ----------
    ens_1, ens_2: Ensemble
        Two Ensemble objects storing the ensemble data to compare.

    Remaining arguments and output
    ------------------------------
        See `dpet.data.comparison.score_avg_jsd` for more information.

    """
    # Calculate torsion angles (alpha_angles).
    alpha_1 = ens_1.get_features(featurization="a_angle")
    alpha_2 = ens_2.get_features(featurization="a_angle")
    # Compute average JSD approximation.
    results = score_avg_jsd(
        alpha_1, alpha_2, bins=bins, return_bins=return_bins, *args, **kwargs
    )
    return results


@check_feature_matrices
def score_emd_approximation(
        x_1: np.ndarray,
        x_2: np.ndarray,
        metric: str = "rmsd"
    ):
    """
    Scores an approximation of the Earth Mover's Distance (EMD) between ensembles
    featurized as (*, F) feature matrices.
    See the [https://doi.org/10.48550/arXiv.2106.14108] and
    [https://doi.org/10.1038/s41467-023-36443-x] for more information.

    Parameters
    ----------
    x_1, x_2: np.ndarray
        Two ensembles featurized as (*, F) NumPy arrays.
    metric: str, optional
        Metric used to compute distances between individual samples in the two
        ensembles. The following metrics are available:
            `rmsd`: root mean square deviation between two samples described by
                F features.
            `l2`: L2 norm of the differences between two samples described by
                F features.
            `angular_l2`: use it only for angular features. Given F angular
                features from two samples, it will compute the squared distance
                between the sin/cos unit vectors of each angular feature and
                the average over all F features.

    """

    # Get the distance function.
    if metric == "rmsd":
        score_func = emd_rmsd
    elif metric == "l2":
        # score_func = emd_rmsd
        score_func = emd_l2
    elif metric == "angular_l2":
        score_func = emd_angular_l2
    else:
        raise ValueError(metric)
    # Compute the distance matrix.
    dist = []
    for i, x_1_i in enumerate(x_1):
        # Confront every reference element to every element in the other sample.
        dist_i = score_func(x_1_i, x_2)
        dist.append(dist_i)
    dist = np.vstack(dist)
    # Hungarian algorithm.
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(dist)
    # Compute the EMD approximation.
    emd = dist[row_ind, col_ind]
    emd = emd.mean()
    return emd


def emd_rmsd(x_1_i, x_2):
    return np.sqrt(np.mean(np.square(x_1_i[None] - x_2), axis=1))

def emd_l2(x_1_i, x_2):
    return np.sum(np.square(x_1_i[None] - x_2), axis=1)

def emd_angular_l2(x_1_i, x_2):
    y_ref_i = np.concatenate(
        [np.cos(x_1_i[...,None]), np.sin(x_1_i)[...,None]], axis=-1
    )
    y_hat = np.concatenate(
        [np.cos(x_2[...,None]), np.sin(x_2)[...,None]], axis=-1
    )
    dist = np.mean(np.square(y_ref_i[None] - y_hat).sum(axis=-1), axis=-1)
    return dist