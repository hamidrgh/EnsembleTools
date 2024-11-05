import math
from typing import Union, List, Tuple
import numpy as np
from scipy.stats import mannwhitneyu
import mdtraj
from dpet.ensemble import Ensemble
from dpet.featurization.utils import get_triu_indices
from dpet.featurization.distances import featurize_ca_dist
from dpet.featurization.angles import featurize_a_angle, featurize_phi_psi


####################################################
# Common functions for computing JSD-based scores. #
####################################################

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
            `dpet.comparison.min_samples_auto_hist`. If it >= than this
            value, returns `dpet.comparison.num_default_bins`.
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


def calc_freqs(x, bins):
    return np.histogram(x, bins=bins)[0] / x.shape[0]

def calc_kld_for_jsd(x_h, m_h):
    """
    Calculates KLD between distribution x and m.
    x_h: histogram frequencies for sample p or q.
    m_h: histogram frequencies for m = 0.5*(p+q).
    """
    non_zero = x_h > 0
    if non_zero.sum() == 0:
        raise ValueError()
    return np.sum(x_h[non_zero]*np.log(x_h[non_zero]/m_h[non_zero]))

def calc_jsd(p_h, q_h):
    """
    Calculates JSD between distribution p and q.
    p_h: histogram frequencies for sample p.
    q_h: histogram frequencies for sample q.
    """
    m_h = 0.5*(p_h + q_h)
    kld_pm = calc_kld_for_jsd(p_h, m_h)
    kld_qm = calc_kld_for_jsd(q_h, m_h)
    jsd_pq = 0.5*(kld_pm + kld_qm)
    return jsd_pq


allowed_scores = {
            "jsd": ["ca_dist", "alpha_angle", "rama"],
        }
scores_data = {
    "adaJSD": ["jsd", "ca_dist"],
    "ramaJSD": ["jsd", "rama"],
    "ataJSD": ["jsd", "alpha_angle"],
}


#####################################################
# Functions to score 1-d distributions of features. #
#####################################################

def score_histogram_jsd(
        p_data: np.ndarray,
        q_data: np.ndarray,
        limits: Union[str, Tuple[int]],
        bins: Union[int, str] = "auto",
        return_bins: bool = False
    ) -> Union[float, Tuple[float, np.ndarray]]:
    """
    Scores an approximation of Jensen-Shannon divergence by discretizing in a
    histogram the values two 1d samples provided as input.

    Parameters
    ----------
    p_data, q_data: np.ndarray
        NumPy arrays of shape (*, ) containing samples from two mono-dimensional
        distribution to be compared.
    limits: Union[str, Tuple[int]]
        Define the method to calculate the minimum and maximum values of the
        range spanned by the bins. Accepted values are:
            "m": will use the minimum and maximum values observed by
                concatenating samples in `p_data` and `q_data`.
            "p": will use the minimum and maximum values observed by
                concatenating samples in `p_data`. If `q_data` contains values
                outside that range, new bins of the same size will be added to
                cover all values of q. Currently, this is not used in any IDPET
                functionality. Note that the `bins` argument will determine
                only the bins originally spanned by `p_data`.
            "a": limits for scoring angular features. Will use a
                (-math.pi, math.pi) range for scoring such features.
            (float, float): provide a custom range. Currently, not used in any
                IDPET functionality.
    bins: Union[int, str], optional
        Determines the number of bins to be used when constructing histograms.
        See `dpet.comparison.get_num_comparison_bins` for more information. The
        range spanned by the bins will be define by the `limits` argument.
    return_bins: bool, optional
        If `True`, returns the bins used in the calculation.

    Returns
    -------
    results: Union[float, Tuple[float, np.ndarray]]
        If `return_bins` is `False`, only returns a float value for the JSD
        score. The score will range from 0 (no common support) to log(2)
        (same distribution). If `return_bins` is `True`, returns a tuple with
        the JSD score and the number of bins.

    """
    
    n_bins = get_num_comparison_bins(bins, x=[p_data, q_data])

    if isinstance(limits, str) and limits in ("m", "p"):
        # Get the minumum and max values of each array.
        p_min = p_data.min()
        p_max = p_data.max()
        q_min = q_data.min()
        q_max = q_data.max()
    
    if isinstance(limits, str) and limits == "m":
        linspace = np.linspace(min(p_min, q_min), max(p_max, q_max), n_bins+1)
        linspace_data = {"m": linspace, "p": linspace}

    elif isinstance(limits, str) and limits == "p":
        # Reference linspace.
        linspace = np.linspace(p_min, p_max, n_bins+1)
        linspace_p = linspace
        assert(linspace[0] == p_min)
        # Calculate the interval size (assuming equal spacing).
        interval = linspace[1] - linspace[0]
        # Extend bins.
        if q_min < p_min:
            # Create a new array extending from p_max to q_max.
            num_extra = (p_min - q_min) / interval
            num_extra = math.ceil(num_extra)+offset  # Number of points to extend.
            lin_extra = [p_min - interval*k for k in range(1, num_extra)]
            lin_extra.reverse()
            lin_extra = np.array(lin_extra)
            # Concatenate the original and extended arrays.
            linspace = np.concatenate((lin_extra, linspace))

        if q_max > p_max:
            # Create a new array extending from p_max to q_max.
            num_extra = (q_max - p_max) / interval
            num_extra = math.ceil(num_extra)+offset  # Number of points to extend.
            lin_extra = [p_max + interval*k for k in range(1, num_extra)]
            lin_extra = np.array(lin_extra)
            # Concatenate the original and extended arrays.
            linspace = np.concatenate((linspace, lin_extra))

        assert(linspace.min() <= q_data.min())
        assert(linspace.max() >= q_data.max())
        
        linspace_data = {"m": linspace, "p": linspace_p}
    
    elif isinstance(limits, str) and limits == "a":
        linspace = np.linspace(-math.pi, math.pi, n_bins+1)
        linspace_data = {"m": linspace, "p": linspace}
    
    elif isinstance(limits, tuple):
        linspace = np.linspace(limits[0], limits[1], n_bins+1)
        linspace_data = {"m": linspace, "p": linspace}
        
    else:
        raise TypeError(limits)

    p_h = calc_freqs(p_data, linspace)
    q_h = calc_freqs(q_data, linspace)
    jsd_pq = calc_jsd(p_h, q_h)

    data = {"linspace": linspace_data, "p_h": p_h, "q_h": q_h}

    if return_bins:
        return jsd_pq, data
    else:
        return jsd_pq

    
@check_feature_matrices
def score_avg_jsd(
        features_1: np.ndarray,
        features_2: np.ndarray,
        limits: str,
        bins: Union[int, str] = 25,
        return_bins: bool = False,
        return_scores: bool = False,
        *args, **kwargs
    ):
    """
    Takes as input two (*, F) feature matrices and computes an average JSD score
    over all F features by discretizing each feature in histograms.

    Parameters
    ----------
    features_1, features_2: np.ndarray
        NumPy arrays of shape (*, F) containing two ensembles with * samples
        described by F features. The number of samples in the two ensembles can
        be different.
    limits: Union[str, Tuple[int]]
        Define the method to calculate the minimum and maximum values of the
        range spanned by the bins. See documentation of `score_histogram_jsd`
        in this module.
    bins: Union[int, str], optional
        Determines the number of bins to be used when constructing histograms.
        See `dpet.comparison.get_num_comparison_bins` for more information.
    return_bins: bool, optional
        If `True`, returns the number of bins used in the calculation.
    return_scores: bool, optional
        If `True`, returns the a tuple with with (avg_score, all_scores), where
        all_scores is an array with all the F scores (one for each feature) used
        to compute the average score.

    Returns
    -------
    results: Union[float, Tuple[float, np.ndarray], Tuple[Tuple[float, np.ndarray], np.ndarray]]
        If `return_bins` is `False`, only returns a float value for the average
        JSD score over the F features. If `return_bins` is `True`, returns a
        tuple with the average JSD score and the number of bins used in the
        comparisons. If `return_scores` is `True` will also return the F scores
        used to compute the average JSD score.

    """

    if limits not in ("m", "a"):
        raise ValueError()

    _bins = get_num_comparison_bins(bins, x=[features_1, features_2])
    jsd = []
    for l in range(features_1.shape[1]):
        jsd_l = score_histogram_jsd(
            p_data=features_1[:,l],
            q_data=features_2[:,l],
            limits=limits,
            bins=_bins,
            return_bins=False,
            *args, **kwargs
        )
        jsd.append(jsd_l)
    avg_jsd =  sum(jsd)/len(jsd)
    if not return_scores:
        jsd_results =  avg_jsd
    else:
        jsd_results =  (avg_jsd, np.array(jsd))
    if not return_bins:
        return jsd_results
    else:
        return jsd_results, _bins


def _get_ada_jsd_features(
        ens: Union[Ensemble, mdtraj.Trajectory],
        min_sep: int,
        max_sep: int,
    ):
    if isinstance(ens, Ensemble):
        return ens.get_features(
            featurization="ca_dist", min_sep=min_sep, max_sep=max_sep
        )
    elif isinstance(ens, mdtraj.Trajectory):
        return featurize_ca_dist(
            traj=ens, 
            get_names=False,
            min_sep=min_sep,
            max_sep=max_sep,
            atom_selector="name CA",
        )
    else:
        raise TypeError(ens.__class__)

    
def score_adaJSD(
        ens_1: Union[Ensemble, mdtraj.Trajectory],
        ens_2: Union[Ensemble, mdtraj.Trajectory],
        bins: Union[str, int] = "auto",
        return_bins: bool = False,
        return_scores: bool = False,
        featurization_params: dict = {},
        *args, **kwargs
    ):
    """
    Utility function to calculate the adaJSD (carbon Alfa Distance Average JSD)
    score between two ensembles. The score evaluates the divergence between
    distributions of Ca-Ca distances of the ensembles.

    Parameters
    ----------
    ens_1, ens_2: Union[Ensemble, mdtraj.Trajectory],
        Two Ensemble or mdtraj.Trajectory objects storing the ensemble data to
        compare.
    bins: Union[str, int], optional
        Determines the number of bins to be used when constructing histograms.
        See `dpet.comparison.get_num_comparison_bins` for more information.
    return_bins: bool, optional
        If `True`, returns the number of bins used in the calculation.
    return_scores: bool, optional
        If `True`, returns the a tuple with with (avg_score, all_scores), where
        all_scores is an array with all the F scores (one for each feature) used
        to compute the average score.
    featurization_params: dict, optional
        Optional dictionary to customize the featurization process to calculate
        Ca-Ca distances. See the `Ensemble.get_features` function for more
        information.

    Remaining arguments and output
    ------------------------------
        See `dpet.comparison.score_avg_jsd` for more information.

    """
    
    min_sep = featurization_params.get("min_sep", 2)
    max_sep = featurization_params.get("max_sep")

    # Calculate Ca-Ca distances.
    ca_dist_1 = _get_ada_jsd_features(
        ens=ens_1, min_sep=min_sep, max_sep=max_sep
    )
    ca_dist_2 = _get_ada_jsd_features(
        ens=ens_2, min_sep=min_sep, max_sep=max_sep
    )

    # Compute average JSD approximation.
    results = score_avg_jsd(
        ca_dist_1, ca_dist_2,
        bins=bins,
        limits="m",
        return_bins=return_bins,
        return_scores=return_scores,
        *args, **kwargs
    )
    return results


def get_adaJSD_matrix(
        ens_1: Union[Ensemble, mdtraj.Trajectory],
        ens_2: Union[Ensemble, mdtraj.Trajectory],
        bins: Union[str, int] = "auto",
        return_bins: bool = False,
        featurization_params: dict = {},
        *args, **kwargs
    ):
    """
    Utility function to calculate the adaJSD score between two ensembles and
    return a matrix with JSD scores for each pair of Ca-Ca distances.

    Parameters
    ----------
    ens_1, ens_2: Union[Ensemble, mdtraj.Trajectory]
        Two Ensemble objects storing the ensemble data to compare.

    Remaining arguments
    -------------------
        See `dpet.comparison.score_adaJSD` for more information.
    
    Output
    ------
        If `return_bins` is False, it will return a tuple containing the
        adaJSD score and a (N, N) NumPy array (where N is the number of residues
        of the protein in the ensembles being compared) containing the JSD
        scores of individual Ca-Ca distances. If `return_bins` is True, it will
        return also the bin value used in all the comparisons.

    """
    min_sep = featurization_params.get("min_sep", 2)
    max_sep = featurization_params.get("max_sep")
    out = score_adaJSD(
        ens_1=ens_1,
        ens_2=ens_2,
        bins=bins,
        return_bins=return_bins,
        return_scores=True,
        featurization_params=featurization_params,
        *args, **kwargs
    )
    if return_bins:
        (avg_score, all_scores), _bins = out
    else:
        (avg_score, all_scores) = out
    n_res = ens_1.get_num_residues()
    res_ids = get_triu_indices(
        L=n_res,
        min_sep=min_sep,
        max_sep=max_sep
    )
    if len(res_ids[0]) != len(all_scores):
        raise ValueError()
    matrix = np.empty((n_res, n_res,))
    matrix[:] = np.nan
    for i, j, s_ij in zip(res_ids[0], res_ids[1], all_scores):
        matrix[i, j] = s_ij
        matrix[j, i] = s_ij
    if return_bins:
        return (avg_score, matrix), _bins
    else:
         return (avg_score, matrix)


def _get_ata_jsd_features(
        ens: Union[Ensemble, mdtraj.Trajectory],
    ):
    if isinstance(ens, Ensemble):
        return ens.get_features(featurization="a_angle")
    elif isinstance(ens, mdtraj.Trajectory):
        return featurize_a_angle(
            traj=ens, get_names=False, atom_selector="name CA"
        )
    else:
        raise TypeError(ens.__class__)


def score_ataJSD(
        ens_1: Union[Ensemble, mdtraj.Trajectory],
        ens_2: Union[Ensemble, mdtraj.Trajectory],
        bins: Union[str, int],
        return_bins: bool = False,
        return_scores: bool = False,
        *args, **kwargs
    ):
    """
    Utility function to calculate the ataJSD (Alpha Torsion Average JSD) score
    between two ensembles. The score evaluates the divergence between
    distributions of alpha torsion angles (the angles formed by four consecutive
    Ca atoms in a protein) of the ensembles.

    Parameters
    ----------
    ens_1, ens_2: Union[Ensemble, mdtraj.Trajectory]
        Two Ensemble objects storing the ensemble data to compare.

    Remaining arguments and output
    ------------------------------
        See `dpet.comparison.score_avg_jsd` for more information.

    """
    # Calculate torsion angles (alpha_angles).
    alpha_1 = _get_ata_jsd_features(ens_1)
    alpha_2 = _get_ata_jsd_features(ens_2)
    # Compute average JSD approximation.
    results = score_avg_jsd(
        alpha_1,
        alpha_2,
        limits="a",
        bins=bins,
        return_bins=return_bins,
        return_scores=return_scores,
        *args, **kwargs
    )
    return results


def get_ataJSD_profile(
        ens_1: Union[Ensemble, mdtraj.Trajectory],
        ens_2: Union[Ensemble, mdtraj.Trajectory],
        bins: Union[str, int],
        return_bins: bool = False,
        *args, **kwargs
    ):
    """
    Utility function to calculate the ataJSD score between two ensembles and
    return a profile with JSD scores for each alpha angle in the proteins.

    Parameters
    ----------
    ens_1, ens_2: Union[Ensemble, mdtraj.Trajectory]
        Two Ensemble objects storing the ensemble data to compare.

    Remaining arguments
    -------------------
        See `dpet.comparison.score_ataJSD` for more information.
    
    Output
    ------
        If `return_bins` is False, it will return a tuple containing the
        ataJSD score and a (N-3, ) NumPy array (where N is the number of
        residues of the protein in the ensembles being compared) containing the
        JSD scores of individual alpha angles. If `return_bins` is True, it will
        return also the bin value used in all the comparisons.

    """

    out = score_ataJSD(
        ens_1=ens_1,
        ens_2=ens_2,
        bins=bins,
        return_bins=return_bins,
        return_scores=True,
        *args, **kwargs
    )
    if return_bins:
        (avg_score, all_scores), _bins = out
    else:
        (avg_score, all_scores) = out
    n_res = ens_1.get_num_residues()
    if n_res - 3 != len(all_scores):
        raise ValueError()
    if return_bins:
        return (avg_score, all_scores), _bins
    else:
         return (avg_score, all_scores)


#####################################################
# Functions to score 2-d distributions of features. #
#####################################################

def score_avg_2d_angle_jsd(
        array_1: np.ndarray,
        array_2: np.ndarray,
        bins: int,
        return_scores: bool = False,
        return_bins: bool = False,
        *args, **kwargs
    ):
    """
    Takes as input two (*, F, 2) bidimensional feature matrices and computes an
    average JSD score over all F bidimensional features by discretizing them
    in 2d histograms. The features in this functions are supposed to be angles
    whose values range from -math.pi to math.pi. For example, int the
    `score_ramaJSD` function the F features represent the phi-psi values of F
    residues in a protein of length L=F+2 (first and last residues don't have
    both phi and psi values).

    Parameters
    ----------
    p_data, q_data: np.ndarray
        NumPy arrays of shape (*, F, 2) containing samples from F bi-dimensional
        distributions to be compared.
    bins: Union[int, str], optional
        Determines the number of bins to be used when constructing histograms.
        See `dpet.comparison.get_num_comparison_bins` for more information. The
        range spanned by the bins will be -math.pi to math.pi. Note that the
        effective number of bins used in the functio will be the square of
        the number returned by `dpet.comparison.get_num_comparison_bins`, since
        we are building a 2d histogram.
    return_bins: bool, optional
        If `True`, returns the square root of the effective number of bins used
        in the calculation.

    Returns
    -------
    results: Union[float, Tuple[float, np.ndarray]]
        If `return_bins` is `False`, only returns a float value for the JSD
        score. The score will range from 0 (no common support) to log(2)
        (same distribution). If `return_bins` is `True`, returns a tuple with
        the JSD score and the number of bins. If `return_scores` is `True` it
        will also return the F scores used to compute the average JSD score.
    """

    if not array_1.shape[1:] == array_2.shape[1:]:
        raise ValueError()

    n_bins = get_num_comparison_bins(bins, x=[array_1, array_2])

    angle_linspace = np.linspace(-np.pi, np.pi, n_bins+1)
    n_distr = array_1.shape[1]
    jsh_list = []
    for i in range(n_distr):
        h2d_1 = np.histogram2d(
            x=array_1[:,i,0], y=array_1[:,i,1], bins=angle_linspace
        )[0]

        h2d_2 = np.histogram2d(
            x=array_2[:,i,0], y=array_2[:,i,1], bins=angle_linspace
        )[0]

        p_h = h2d_1.ravel()
        p_h = p_h / p_h.sum()
        q_h = h2d_2.ravel()
        q_h = q_h / q_h.sum()

        jsd_pq = calc_jsd(p_h, q_h)
        jsh_list.append(jsd_pq)

    avg_score = np.mean(jsh_list)

    if not return_scores:
        jsd_results =  avg_score
    else:
        jsd_results =  (avg_score, np.array(jsh_list))
    if not return_bins:
        return jsd_results
    else:
        return jsd_results, n_bins


def _get_rama_jsd_features(
        ens: Union[Ensemble, mdtraj.Trajectory],
    ):
    if isinstance(ens, Ensemble):
        if ens.coarse_grained:
            raise ValueError(
                "ramaJSD cannot be computed for coarse-grained ensembles"
            )
        return ens.get_features("phi_psi", ravel=False)
    elif isinstance(ens, mdtraj.Trajectory):
        # TODO: should check for a CG ensemble here.
        return featurize_phi_psi(traj=ens,  get_names=False, ravel=False)
    else:
        raise TypeError(ens.__class__)


def score_ramaJSD(
        ens_1: Union[Ensemble, mdtraj.Trajectory],
        ens_2: Union[Ensemble, mdtraj.Trajectory],
        bins: int,
        return_scores: bool = False,
        return_bins: bool = False,
    ):
    """
    Utility unction to calculate the ramaJSD (Ramachandran plot average JSD)
    score between two ensembles. The score evaluates the divergence between
    distributions of phi-psi torsion angles of every residue in the ensembles.

    Parameters
    ----------
    ens_1, ens_2: Union[Ensemble, mdtraj.Trajectory]
        Two Ensemble objects storing the ensemble data to compare.

    Remaining arguments and output
    ------------------------------
        See `dpet.comparison.score_avg_jsd` for more information.
    """

    phi_psi_1 = _get_rama_jsd_features(ens_1)
    phi_psi_2 = _get_rama_jsd_features(ens_2)
    
    return score_avg_2d_angle_jsd(
        array_1=phi_psi_1,
        array_2=phi_psi_2,
        bins=bins,
        return_scores=return_scores,
        return_bins=return_bins
    )


def get_ramaJSD_profile(
        ens_1: Union[Ensemble, mdtraj.Trajectory],
        ens_2: Union[Ensemble, mdtraj.Trajectory],
        bins: Union[str, int],
        return_bins: bool = False,
        *args, **kwargs
    ):
    """
    Utility function to calculate the ramaJSD score between two ensembles and
    return a profile with JSD scores for the Ramachandran plots of pair of
    corresponding residue in the proteins.

    Parameters
    ----------
    ens_1, ens_2: Union[Ensemble, mdtraj.Trajectory]
        Two Ensemble objects storing the ensemble data to compare.

    Remaining arguments
    -------------------
        See `dpet.comparison.score_ramaJSD` for more information.
    
    Output
    ------
        If `return_bins` is False, it will return a tuple containing the
        ramaJSD score and a (N-2, ) NumPy array (where N is the number of
        residues of the protein in the ensembles being compared) containing the
        JSD scores of individual residues. If `return_bins` is True, it will
        return also the bin value used in all the comparisons.

    """

    out = score_ramaJSD(
        ens_1=ens_1,
        ens_2=ens_2,
        bins=bins,
        return_bins=return_bins,
        return_scores=True,
        *args, **kwargs
    )
    if return_bins:
        (avg_score, all_scores), _bins = out
    else:
        (avg_score, all_scores) = out
    n_res = ens_1.get_num_residues()
    if n_res - 2 != len(all_scores):
        raise ValueError()
    if return_bins:
        return (avg_score, all_scores), _bins
    else:
         return (avg_score, all_scores)


########################
# All-vs-all analysis. #
########################

def all_vs_all_comparison(
        ensembles: List[Ensemble],
        score: str,
        featurization_params: dict = {},
        bootstrap_iters: int = None,
        bootstrap_frac: float = 1.0,
        bootstrap_replace: bool = True,
        bins: Union[int, str] = 50,
        random_seed: int = None,
        verbose: bool = False
    ) -> dict:
    """
    Compare all pair of ensembles using divergence scores.
    Implemented scores are approximate average Jensen–Shannon divergence
    (JSD) over several kinds of molecular features. The lower these scores
    are, the higher the similarity between the probability distribution of
    the features of the ensembles. JSD scores here range from a minimum of 0
    to a maximum of log(2) ~= 0.6931.

    Parameters
    ----------
    ensembles: List[Ensemble]
        Ensemble objectes to analyze.
    score: str
        Type of score used to compare ensembles. Choices: `adaJSD` (carbon
        Alfa Distance Average JSD), `ramaJSD` (RAMAchandran average JSD) and
        `ataJSD` (Alpha Torsion Average JSD). `adaJSD` scores the average
        JSD over all Ca-Ca distance distributions of residue pairs with
        sequence separation > 1. `ramaJSD` scores the average JSD over the
        phi-psi angle distributions of all residues. `ataJSD` scores the
        average JSD over all alpha torsion angles, which are the angles
        formed by four consecutive Ca atoms in a protein.
    featurization_params: dict, optional
        Optional dictionary to customize the featurization process for the
        above features.
    bootstrap_iters: int, optional
        Number of bootstrap iterations. By default its value is None. In
        this case, IDPET will directly compare each pair of ensemble $i$ and
        $j$ by using all of their conformers and perform the comparison only
        once. On the other hand, if providing an integer value to this
        argument, each pair of ensembles $i$ and $j$ will be compared
        `bootstrap_iters` times by randomly selecting (bootstrapping)
        conformations from them. Additionally, each ensemble will be
        auto-compared with itself by subsampling conformers via
        bootstrapping. Then IDPET will perform a statistical test to
        establish if the inter-ensemble ($i != j$) scores are significantly
        different from the intra-ensemble ($i == j$) scores. The tests work
        as follows: for each ensemble pair $i != j$ IDPET will get their
        inter-ensemble comparison scores obtained in bootstrapping. Then, it
        will get the bootstrapping scores from auto-comparisons of ensemble
        $i$ and $j$ and the scores with the higher mean here are selected as
        reference intra-ensemble scores. Finally, the inter-ensemble and
        intra-ensemble scores are compared via a one-sided Mann-Whitney U
        test with the alternative hypothesis being: inter-ensemble scores
        are stochastically greater than intra-ensemble scores. The p-values
        obtained in these tests will additionally be returned. For small
        protein structural ensembles (less than 500 conformations) most
        comparison scores in IDPET are not robust estimators of
        divergence/distance. By performing bootstrapping, you can have an
        idea of how the size of your ensembles impacts the comparison. Use
        values >= 50 when comparing ensembles with very few conformations
        (less than 100). When comparing large ensembles (more than
        1,000-5,000 conformations) you can safely avoid bootstrapping.
    bootstrap_frac: float, optional
        Fraction of the total conformations to sample when bootstrapping.
        Default value is 1.0, which results in bootstrap samples with the
        same number of conformations of the original ensemble.
    bootstrap_replace: bool, optional
        If `True`, bootstrap will sample with replacement. Default is `True`.
    bins: Union[int, str], optional
        Number of bins or bin assignment rule for JSD comparisons. See the
        documentation of `dpet.comparison.get_num_comparison_bins` for
        more information.
    random_seed: int, optional
        Random seed used when performing bootstrapping.
    verbose: bool, optional
        If `True`, some information about the comparisons will be printed to
        stdout.

    Returns
    -------
    results: dict
        Aa dictionary containing the following key-value pairs:
            `scores`: a (M, M, B) NumPy array storing the comparison
                scores, where M is the number of ensembles being
                compared and B is the number of bootstrap iterations (B
                will be 1 if bootstrapping was not performed).
            `p_values`: a (M, M) NumPy array storing the p-values
                obtained in the statistical test performed when using
                a bootstrapping strategy (see the `bootstrap_iters`)
                method. Returned only when performing a bootstrapping
                strategy.
    """

    score_type, feature = scores_data[score]
    
    ### Check arguments.
    if score_type == "jsd":
        if feature in ("ca_dist", "alpha_angle"):
            score_func = score_avg_jsd
        elif feature == "rama":
            score_func = score_avg_2d_angle_jsd
        else:
            raise ValueError(
                f"Invalid feature for JSD-based scores: {feature}"
            )
    else:
        raise ValueError(
            "The type of similarity score should be selected among:"
            f" {list(allowed_scores.keys())}"
        )
    if not feature in allowed_scores[score_type]:
        raise ValueError(
            f"The '{score_type}' score must be calculated based on the"
            f" following features: {allowed_scores[score_type]}"
        )

    min_bootstrap_iters = 2
    if isinstance(bootstrap_iters, int):
        if bootstrap_iters < min_bootstrap_iters:
            raise ValueError(
                f"Need at leasts {min_bootstrap_iters} bootstrap_iters"
            )
        comparison_iters = bootstrap_iters
    elif bootstrap_iters is None:
        comparison_iters = 1
    else:
        raise TypeError(bootstrap_iters)

    if not 0 <= bootstrap_frac <= 1:
        raise ValueError(f"Invalid bootstrap_frac: {bootstrap_frac}")

    ### Check the ensembles.
    num_residues = set([e.get_num_residues() for e in ensembles])
    if len(num_residues) != 1:
        raise ValueError(
            "Can only compare ensembles with the same number of residues."
            " Ensembles in this analysis have different number of residues."
        )
    
    ### Define the random seed.
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
        rand_func = rng.choice
    else:
        rand_func = np.random.choice

    ### Featurize (run it here to avoid re-calculating at every comparison).
    features = []
    # Compute features.
    for ensemble_i in ensembles:
        if feature == "ca_dist":
            feats_i = ensemble_i.get_features(
                normalize=False,
                featurization="ca_dist",
                min_sep=featurization_params.get("min_sep", 2),
                max_sep=featurization_params.get("max_sep"),
            )
        elif feature == "alpha_angle":
            feats_i = ensemble_i.get_features(featurization="a_angle")
        elif feature == "rama":
            feats_i = ensemble_i.get_features(
                featurization="phi_psi", ravel=False
            )
        else:
            raise ValueError(f"Invalid feature for comparison: {feature}")
        features.append(feats_i)
    
    ### Setup the comparisons.
    if verbose:
        print(f"# Scoring '{score_type}' using features '{feature}'")
    n = len(ensembles)

    # Define the parameters for the evaluation.
    if score_type == "jsd":
        # Apply the same bin number to every comparison, based on the number
        # of conformers in the smallest ensemble.
        num_bins = get_num_comparison_bins(bins, x=features)
        print("num_bins:", num_bins)
        scoring_params = {"bins": num_bins}
        if feature in ("ca_dist", ):
            scoring_params["limits"] = "m"
        elif feature in ("alpha_angle", ):
            scoring_params["limits"] = "a"
        else:
            pass
        if verbose:
            print(f"- Number of bins for all comparisons: {num_bins}")
    else:
        raise ValueError(score_type)

    # Initialize a (n, n, *) matrices for storing the comparison scores.
    score_matrix = np.zeros((n, n, comparison_iters))
    
        # Get the pairs to compare.
    pairs_to_compare = []
    for i in range(n):
        for j in range(n):
            if j > i:
                pairs_to_compare.append((i, j))
            elif i == j and comparison_iters > 1:
                pairs_to_compare.append((i, j))
    
    # Get the total number of comparisons to perform.
    if comparison_iters > 1:
        permutation_scores = np.zeros((n, n, comparison_iters))
        tot_comparisons = (len(pairs_to_compare) + len(pairs_to_compare))*comparison_iters
    else:
        tot_comparisons = len(pairs_to_compare)
    if verbose:
        print(
            f"- We have {len(pairs_to_compare)} pairs of ensembles"
            f" and will perform a total of {tot_comparisons} comparisons."
        )

    ### Perform the comparisons.

    ## Use all conformers, repeat only once.
    if comparison_iters == 1:
        
        for i, j in pairs_to_compare:
            # Score.
            score_ij = score_func(
                features[i], features[j], **scoring_params
            )
            # Store the results.
            score_matrix[i, j, 0] = score_ij
            score_matrix[j, i, 0] = score_ij

    ## Bootstrap analysis, compare ensembles multiple times by subsampling.
    else:
        
        for i, j in pairs_to_compare:

            for k in range(comparison_iters):

                # Features for ensemble i.
                n_i = features[i].shape[0]
                rand_ids_ik = rand_func(
                    n_i,
                    max(int(n_i*bootstrap_frac), 1),
                    replace=bootstrap_replace
                )
                features_ik = features[i][rand_ids_ik]
                # Features for ensemble j.
                n_j = features[j].shape[0]
                rand_ids_jk = rand_func(
                    n_j,
                    max(int(n_j*bootstrap_frac), 1),
                    replace=bootstrap_replace
                )
                features_jk = features[j][rand_ids_jk]
                # Score.
                score_ijk = score_func(
                    features_ik, features_jk, **scoring_params
                )
                # Store the results.
                score_matrix[i, j, k] = score_ijk
                score_matrix[j, i, k] = score_ijk


    ### Prepare the output.
    output = {"scores": score_matrix}

    ### Evaluate statistical significance if necessary.
    if comparison_iters > 1:
        # Perform mannwhitneyu test to check if inter-ensembles scores are
        # different in a statistically significant way from intra-ensemble
        # scores.
        p_values = np.zeros((n, n, ))
        for i in range(n):
            for j in range(n):
                if i >= j:
                    continue
                scores_i = score_matrix[i, i]
                scores_j = score_matrix[j, j]
                # Get the higher intra-ensemble scores between ensemble
                # i and j.
                scores_ref = scores_i if scores_i.mean() > scores_j.mean() \
                                        else scores_j
                # Run the statistical test.
                u_ij = mannwhitneyu(
                    x=score_matrix[i, j],
                    y=scores_ref,
                    alternative='greater'
                )
                # Store p-values.
                p_values[i, j] = u_ij[1]
                p_values[j, i] = u_ij[1]
        output["p_values"] = p_values

    return output


######################
# Statistical tests. #
######################

def percentile_func(a, q):
    return np.percentile(a=a, q=q, axis=-1)

def confidence_interval(
        theta_boot, theta_hat=None, confidence_level=0.95, method='percentile'
    ):
    """
    Returns bootstrap confidence intervals.
    Adapted from: https://github.com/scipy/scipy/blob/v1.14.0/scipy/stats/_resampling.py
    """
    alpha = (1 - confidence_level)/2
    interval = alpha, 1-alpha
    # Calculate confidence interval of statistic
    ci_l = percentile_func(theta_boot, interval[0]*100)
    ci_u = percentile_func(theta_boot, interval[1]*100)
    if method == 'basic':
        if theta_hat is None:
            theta_hat = np.mean(theta_boot)
        ci_l, ci_u = 2*theta_hat - ci_u, 2*theta_hat - ci_l
    return [ci_l, ci_u]

def process_all_vs_all_output(
        comparison_out: dict,
        confidence_level: float = 0.95
    ):
    """
    Takes as input a dictionary produced as output of the `all_vs_all_comparison`
    function. If a bootstrap analysis was performed in `all_vs_all_comparison`,
    this function will assign bootstrap confidence intervals.
    """
    if not isinstance(comparison_out, dict):
        raise TypeError()

    if len(comparison_out) == 1:
        if "scores" not in comparison_out:
            raise KeyError()
        comparison_out["mode"] = "single"
        comparison_out["scores_mean"] = comparison_out["scores"][:,:,0]
    elif len(comparison_out) == 2:
        if "scores" not in comparison_out:
            raise KeyError()
        if "p_values" not in comparison_out:
            raise KeyError()
        comparison_out["mode"] = "bootstrap"
        scores_mean = comparison_out["scores"].mean(axis=2)
        comparison_out["scores_mean"] = scores_mean
        conf_intervals = np.zeros(
            (scores_mean.shape[0], scores_mean.shape[1], 2)
        )
        for i in range(conf_intervals.shape[0]):
            for j in range(conf_intervals.shape[1]):
                if i > j:
                    continue
                c_ij = confidence_interval(
                    comparison_out["scores"][i, j],
                    confidence_level=confidence_level
                )
                conf_intervals[i, j][0] = c_ij[0]
                conf_intervals[i, j][1] = c_ij[1]
                conf_intervals[j, i][0] = c_ij[0]
                conf_intervals[j, i][1] = c_ij[1]
        comparison_out["confidence_intervals"] = conf_intervals
    else:
        raise KeyError()

    return comparison_out