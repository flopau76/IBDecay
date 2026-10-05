"""Verify that the module expectations agrees with msprime simulations of ROH/IBD"""

import numpy as np
import pytest

from IBDecay.expectations import (
    ibd_count_Ne,
    ibd_decay,
    ibd_sum_Ne,
    roh_count_Ne,
    roh_sum_Ne,
)
from IBDecay.simulations import simulate_ibd, simulate_roh

# Parameters shared by the integration tests: small enough to keep msprime fast,
# large enough to get a few hundred segments per bin.
NE = 200
CHR_LGTS = (1.5, 1.0, 0.5)
MIN_L = 0.02
MAX_T = 1000
N_SIM_ROH = 200
N_SIM_IBD = 150
T_DECAY = 25  # generations between the two samples in the decay test
BINS = np.array([0.02, 0.04, 0.08, 0.16, 0.4, max(CHR_LGTS)])

# Tolerance on |observed - expected|: RTOL * expected + N_SIGMA * sampling noise.
# RTOL absorbs model approximations (e.g. short segments are slightly more frequent
# under msprime's coalescent than predicted by the Ne-based formulas).
RTOL = 0.2
N_SIGMA = 4


def _assert_matches(observed, expected, sigma):
    """Check per-bin agreement between observed and expected values."""
    tol = RTOL * expected + N_SIGMA * sigma
    assert np.all(np.abs(observed - expected) <= tol), (
        f"\nobserved: {observed}\nexpected: {expected}\ntolerance: {tol}"
    )


def _bin_counts(lengths, n_samples):
    """Average number of segments per bin and its Poisson standard error
    (floored at one segment, so that empty bins get a non-zero tolerance)."""
    counts = np.histogram(lengths, BINS)[0]
    return counts / n_samples, np.sqrt(np.maximum(counts, 1)) / n_samples


def _bin_sums(lengths, n_samples):
    """Average summed length per bin and its standard error
    (floored at one segment of the bin's upper length)."""
    sums = np.histogram(lengths, BINS, weights=lengths)[0]
    sums_sq = np.histogram(lengths, BINS, weights=lengths**2)[0]
    return sums / n_samples, np.sqrt(np.maximum(sums_sq, BINS[1:] ** 2)) / n_samples


@pytest.fixture(scope="module")
def roh_lengths():
    df = simulate_roh(
        n_sim=N_SIM_ROH, min_l=MIN_L, max_t=MAX_T, Ne=NE, chr_lgts=CHR_LGTS, seed=1
    )
    return df["lengthM"].to_numpy(dtype=float)


@pytest.fixture(scope="module")
def ibd_lengths():
    """IBD between contemporaneous pairs of diploid individuals."""
    df = simulate_ibd(
        n_sim=N_SIM_IBD, min_l=MIN_L, max_t=MAX_T, Ne=NE, chr_lgts=CHR_LGTS, seed=2
    )
    return df["lengthM"].to_numpy(dtype=float)


@pytest.fixture(scope="module")
def ibd_lengths_decayed():
    """IBD between pairs of diploid individuals sampled T_DECAY generations apart."""
    df = simulate_ibd(
        n_sim=N_SIM_IBD,
        min_l=MIN_L,
        max_t=MAX_T,
        Ne=NE,
        chr_lgts=CHR_LGTS,
        t1=0,
        t2=T_DECAY,
        seed=3,
    )
    return df["lengthM"].to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# ROH based on Ne (integration: exercises real msprime simulations)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestRohNe:
    def test_count_matches_simulation(self, roh_lengths):
        observed, sigma = _bin_counts(roh_lengths, N_SIM_ROH)
        expected = roh_count_Ne((BINS[:-1], BINS[1:]), Ne=NE, chr_lgts=CHR_LGTS)
        _assert_matches(observed, expected, sigma)

    def test_sum_matches_simulation(self, roh_lengths):
        observed, sigma = _bin_sums(roh_lengths, N_SIM_ROH)
        expected = roh_sum_Ne((BINS[:-1], BINS[1:]), Ne=NE, chr_lgts=CHR_LGTS)
        _assert_matches(observed, expected, sigma)


# ---------------------------------------------------------------------------
# IBD based on Ne (integration: exercises real msprime simulations)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestIbdNe:
    def test_count_matches_simulation(self, ibd_lengths):
        observed, sigma = _bin_counts(ibd_lengths, N_SIM_IBD)
        expected = ibd_count_Ne((BINS[:-1], BINS[1:]), Ne=NE, chr_lgts=CHR_LGTS)
        _assert_matches(observed, expected, sigma)

    def test_sum_matches_simulation(self, ibd_lengths):
        observed, sigma = _bin_sums(ibd_lengths, N_SIM_IBD)
        expected = ibd_sum_Ne((BINS[:-1], BINS[1:]), Ne=NE, chr_lgts=CHR_LGTS)
        _assert_matches(observed, expected, sigma)


# ---------------------------------------------------------------------------
# IBD across generations (integration: exercises real msprime simulations)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestIbdDecay:
    """In a constant-size population, IBD between individuals sampled at times 0 and T
    results from the decay over T meioses of the IBD between contemporaneous individuals.
    """

    def test_decay_matches_simulation(self, ibd_lengths, ibd_lengths_decayed):
        observed, sigma = _bin_counts(ibd_lengths_decayed, N_SIM_IBD)
        expected = ibd_decay(
            t=[T_DECAY],
            admix=[1],
            bins=BINS,
            lengths_ancestral=ibd_lengths,
            nb_pairs_ancestral=N_SIM_IBD,
        )[0, 0]
        _assert_matches(observed, expected, sigma)

    def test_long_segments_decay(self, ibd_lengths, ibd_lengths_decayed):
        """Sanity check that the time gap removes long segments, so the test above is informative."""
        ancestral, _ = _bin_counts(ibd_lengths, N_SIM_IBD)
        decayed, _ = _bin_counts(ibd_lengths_decayed, N_SIM_IBD)
        assert np.all(decayed[-2:] < 0.5 * ancestral[-2:])
