"""Maximum-likelihood estimation of Ne and IBD-decay (time/admixture) parameters."""

from typing import NamedTuple

import numpy as np
from scipy.optimize import minimize_scalar, brentq, OptimizeResult

from typing import Literal, Tuple

from IBDecay.utils import chromosome_lengthsM_human
from IBDecay.expectations import (
    roh_density_Ne,
    roh_count_Ne,
    ibd_density_Ne,
    ibd_count_Ne,
    ibd_decay,
)

def log_likelihood_Ne(Ne: float, lengths_observed: np.ndarray, nb_observations: float,
                        data_type: Literal['IBD', 'ROH'],
                        lengths_bounds: Tuple[float, float], 
                        chr_lgts=chromosome_lengthsM_human) -> float:
    """Calculates the log-likelihood for a given Ne, based on observed IBD/ROH lengths.
    Computation is done assuming independence between segments, using a Poisson point process model.
    Args:
        Ne: effective population size.
        lengths_observed: array containing the length of the observed IBD/ROH segments.
        nb_observations: number of observations (nb of individuals for ROH / pairs of individuals for IBD)
        data_type: type of data, either 'IBD' or 'ROH'.
        lengths_bounds: tuple containing the lower and upper bounds [in Morgan] for the length of segments to use.
        chr_lgts: chromosome lengths [in Morgan]"""
    if data_type == 'IBD':
        density_func = ibd_density_Ne
        integrale_func = ibd_count_Ne
    elif data_type == 'ROH':
        density_func = roh_density_Ne
        integrale_func = roh_count_Ne
    else:
        raise ValueError(f"data_type must be 'IBD' or 'ROH', not {data_type}")

    lengths_observed = np.asarray(lengths_observed)
    lengths_observed = lengths_observed[(lengths_observed > lengths_bounds[0]) & (lengths_observed < lengths_bounds[1])]
    pdf_vals = density_func(lengths_observed, Ne, chr_lgts)
    return np.sum(np.log(pdf_vals)) - nb_observations * integrale_func(lengths_bounds, Ne, chr_lgts)


def estimate_Ne(lengths_observed: np.ndarray, nb_observations: float,
                 data_type: Literal['IBD', 'ROH'], 
                 lengths_bounds: Tuple[float, float],
                 Ne_bounds=(10, 10e6),
                 chr_lgts=chromosome_lengthsM_human) -> Tuple[float, Tuple[float, float]]:
    """Estimates Ne and a 95% confidence interval using the maximum log likelihood.
    Args:
        lengths_observed: array containing the length of the observed IBD/ROH segments.
        nb_observations: number of observations (nb of individuals for ROH / pairs of individuals for IBD).
        data_type: type of data, either 'IBD' or 'ROH'.
        lengths_bounds: tuple containing the lower and upper bounds [in Morgan] for the length of segments to use.
        Ne_bounds: bounds on Ne for the maximum search
        chr_lgts: chromosome lengths [in Morgan]
    Returns:
        The optimal Ne and the 95% confidence interval."""
    res = minimize_scalar(
        lambda Ne: -log_likelihood_Ne(Ne=Ne,
                                        lengths_observed=lengths_observed,
                                        nb_observations=nb_observations,
                                        data_type=data_type,
                                        lengths_bounds=lengths_bounds,
                                        chr_lgts=chr_lgts
        ),
        method='bounded', bounds=Ne_bounds
    )
    assert isinstance(res, OptimizeResult)

    # get 95% CI with Wilks' theorem
    def root_func(Ne):
        return res.fun + 3.841 / 2 + log_likelihood_Ne(Ne=Ne,
                                        lengths_observed=lengths_observed,
                                        nb_observations=nb_observations,
                                        data_type=data_type,
                                        lengths_bounds=lengths_bounds,
                                        chr_lgts=chr_lgts
        )

    ci_lower = brentq(root_func, Ne_bounds[0], res.x - 1e-5, xtol=1e-5)
    ci_upper = brentq(root_func, res.x + 1e-5, Ne_bounds[1], xtol=1e-5)

    return res.x, (ci_lower, ci_upper)


def log_likelihood_IBDecay(t: np.ndarray, admix: np.ndarray, bins: np.ndarray,
                            lengths_ancestral: np.ndarray, nb_pairs_ancestral: float,
                            lengths_between: np.ndarray, nb_pairs_between: float) -> np.ndarray:
    """Returns the log-likelihood of observing the data at time t since common ancestor.
    Args:
        t: time since common ancestor
        admix: proportion of admixture from a source with no shared ancestry (ie no IBD)
        bins: bins to discretize the IBD lengths
        lengths_ancestral: IBD lengths in the ancestral population.
        nb_pairs_ancestral: number of pairs in the ancestral population.
        lengths_between: IBD lengths between the two populations.
        nb_pairs_between: number of pairs between the two populations.
    Returns:
        A 2D array of shape (len(t), len(admix)) with the log-likelihood for each combination of t and admix."""
    expected = ibd_decay(t, admix, bins, lengths_ancestral, nb_pairs_ancestral)  # shape(time,admix, bins)

    xt = np.histogram(lengths_between, bins=bins, density=False)[0]  # shape(bins)

    xt_b = xt[np.newaxis, np.newaxis, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        log_expected = np.log(expected)
        xlogx_term = np.where(xt_b == 0, 0.0, xt_b * log_expected)

    ll = xlogx_term - nb_pairs_between * expected   # shape(time, admix, bins)
    ll = np.sum(ll, axis=2)  # shape(time, admix)
    ll -= np.max(ll)
    return ll

class IBDecayEstimate(NamedTuple):
    t: float
    x: float
    t_ci: np.ndarray
    x_ci: np.ndarray
    ll: np.ndarray

def estimate_IBDecay(t_grid: np.ndarray, admix_grid: np.ndarray, bins: np.ndarray,
                      lengths_ancestral: np.ndarray, nb_pairs_ancestral: float,
                      lengths_between: np.ndarray, nb_pairs_between: float) -> IBDecayEstimate:
    """Returns the optimal t and admix that maximize the log-likelihood of observing the data.
    Args:
        t_grid: grid of time since common ancestor to test
        admix_grid: grid of admixture proportion to test
        bins: bins to discretize the IBD lengths
        lengths_ancestral: IBD lengths in the ancestral population.
        nb_pairs_ancestral: number of pairs in the ancestral time.
        lengths_between: IBD lengths between the two populations.
        nb_pairs_between: number of pairs between the two populations.
    Returns:
        IBDecayEstimate with fields:
            t: the maximum-likelihood estimate of t
            x: the maximum-likelihood estimate of admix
            t_ci: [lower, upper] 95% profile-likelihood CI bounds for t
            x_ci: [lower, upper] 95% profile-likelihood CI bounds for admix
            ll: log-likelihood for each combination of t and admix (shape t_grid x admix_grid)"""
    ### Compute the log likelihood along the grid
    ll = log_likelihood_IBDecay(t_grid, admix_grid, bins, lengths_ancestral, nb_pairs_ancestral,
                                 lengths_between, nb_pairs_between)

    ### Find the maximum
    idx_max = np.unravel_index(np.argmax(ll), ll.shape)
    t_opt = t_grid[idx_max[0]]
    admix_opt = admix_grid[idx_max[1]]

    ### Get 95% CI interval
    threshold = np.max(ll) - 3.841/2   # chi^2(1, 0.95)/2
    time_profile = np.max(ll, axis=1)
    time_ci = t_grid[np.where(time_profile >= threshold)[0][[0,-1]]]
    admix_profile = np.max(ll, axis=0)
    admix_ci = admix_grid[np.where(admix_profile >= threshold)[0][[0,-1]]]
    return IBDecayEstimate(t_opt, admix_opt, time_ci, admix_ci, ll)