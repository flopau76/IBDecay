"""Functions to compute expected ROH/IBD distributions, counts, and sums."""

import numpy as np

from IBDecay.utils import chromosome_lengthsM_human


#### ROH based on Ne

def _roh_density_Ne_chr(x, Ne: float, chr_l: float):
    """helper function: roh density for a single chromosome"""
    return 8 * Ne * (1 + 4 * chr_l * Ne) / (1 + 4 * x * Ne) ** 3 * (0 <= x) * (x <= chr_l)


def roh_density_Ne(x, Ne: float, chr_lgts=chromosome_lengthsM_human):
    """Returns the expected ROH distribution, given an effective population size Ne.
    Args:
        x: length [in Morgan] where to evaluate the density. Can be a float or an array of float
        Ne: effective population size
        chr_lgts: chromosome lengths [in Morgan]"""
    pdfs = [_roh_density_Ne_chr(x, Ne, chr_l) for chr_l in chr_lgts]
    pdf_total = np.sum(pdfs, axis=0)
    return pdf_total


def _roh_count_Ne_chr(x, Ne: float, chr_l: float):
    """helper function: integrand of roh_density(x)"""
    x = np.clip(x, 0, chr_l)
    return 8 * Ne * (chr_l - x) * (1 + 2 * Ne * (chr_l + x)) / ((1 + 4 * chr_l * Ne) * (1 + 4 * x * Ne) ** 2)


def roh_count_Ne(bins=(0, np.inf), Ne: float = 100, chr_lgts=chromosome_lengthsM_human):
    """Returns the expected number of ROH blocks in a given interval, given an effective population size.
    Args:
        bins: tuple or array size (2,n) with bin edges [in Morgan]
        Ne: effective population size
        chr_lgts: chromosome lengths [in Morgan]"""
    counts = [_roh_count_Ne_chr(bins[0], Ne, chr_l) - _roh_count_Ne_chr(bins[1], Ne, chr_l) for chr_l in chr_lgts]
    counts_total = np.sum(counts, axis=0)
    return counts_total


def _roh_sum_Ne_chr(x, Ne: float, chr_l: float):
    """helper function: integrand of x * roh_density(x)"""
    x = np.clip(x, 0, chr_l)
    return 8 * Ne * (chr_l - x) * (1 + 2 * Ne * (chr_l + x)) / ((1 + 4 * chr_l * Ne) * (1 + 4 * x * Ne) ** 2)


def roh_sum_Ne(bins=(0, np.inf), Ne: float = 100, chr_lgts=chromosome_lengthsM_human):
    """Returns the expected summed length [in Morgan] of ROH blocks in a given interval, given an effective population size.
    Args:
        bins: tuple or array size (2,n) with bin edges [in Morgan]
        Ne: effective population size
        chr_lgts: chromosome lengths [in Morgan]"""
    sums = [_roh_sum_Ne_chr(bins[0], Ne, chr_l) - _roh_sum_Ne_chr(bins[1], Ne, chr_l) for chr_l in chr_lgts]
    sums_total = np.sum(sums, axis=0)
    return sums_total


#### IBD based on Ne

def ibd_density_Ne(x, Ne: float, chr_lgts=chromosome_lengthsM_human):
    """Returns the expected IBD length distribution, given an effective population size.
    Args:
        x: length [in Morgan] where to evaluate the density. Can be a float or an array of float.
        Ne: effective population size.
        chr_lgts: chromosome lengths [in Morgan]"""
    return 4 * roh_density_Ne(x, Ne, chr_lgts)


def ibd_count_Ne(bins=(0, np.inf), Ne: float = 100, chr_lgts=chromosome_lengthsM_human):
    """Returns the expected number of IBD blocks in a given interval, given an effective population size.
    Args:
        bins: tuple or array size (2,n) with bin edges [in Morgan]
        Ne: effective population size
        chr_lgts: chromosome lengths [in Morgan]"""
    return 4 * roh_count_Ne(bins, Ne, chr_lgts)


def ibd_sum_Ne(bins=(0, np.inf), Ne: float = 100, chr_lgts=chromosome_lengthsM_human):
    """Returns the expected summed length [in Morgan] of IBD blocks in a given interval, given an effective population size.
    Args:
        bins: tuple or array size (2,n) with bin edges [in Morgan]
        Ne: effective population size
        chr_lgts: chromosome lengths [in Morgan]"""
    return 4 * roh_sum_Ne(bins, Ne, chr_lgts)


#### HBD based on a pedigree

def coalescence_prob_pedigree(nb_meiosis: int, comm_anc: int = 1) -> float:
    """Returns the coalescence probability of two alleles, given a pedigree.
    Args:
        nb_meiosis: length of the genealogic path between the two alleles
        comm_anc: nb of such paths"""
    return comm_anc * 2 * (1 / 2) ** nb_meiosis  # factor two because two potential ancestral alleles (diploid)


def _block_density_chr(x, nb_meiosis: float, chr_l: float):
    pdf = ((chr_l - x) * nb_meiosis ** 2 + nb_meiosis) * np.exp(-nb_meiosis * x)
    return pdf * (0 < x) * (x < chr_l)


def block_density(x, nb_meiosis: float, chr_lgts=chromosome_lengthsM_human):
    """Returns the expected DNA length distribution, given a number of meiosis.
    Args:
        x: length [in Morgan] where to evaluate the density. Can be a float or an array of float
        nb_meiosis: nb of meiosis -> average nb of recombination per Morgan
        chr_lgts: chromosome lengths [in Morgan]"""
    pdfs = [_block_density_chr(x, nb_meiosis, chr_l) for chr_l in chr_lgts]
    pdf_total = np.sum(pdfs, axis=0)
    return pdf_total


def _block_count_chr(x, nb_meiosis: float, chr_l: float):
    count = (chr_l - x) * nb_meiosis * np.exp(-nb_meiosis * x)
    return count * (0 < x) * (x < chr_l)


def block_count(bins, nb_meiosis: float, chr_lgts=chromosome_lengthsM_human):
    """Returns the expected nb of DNA segments in a given interval, given a number of meiosis.
    Args:
        bins: tuple or array size (2,n) with bin edges [in Morgan]
        nb_meiosis: nb of meiosis -> average nb of recombination per Morgan
        chr_lgts: chromosome lengths [in Morgan]"""
    counts = [_block_count_chr(bins[0], nb_meiosis, chr_l) - _block_count_chr(bins[1], nb_meiosis, chr_l) for chr_l in chr_lgts]
    counts_total = np.sum(counts, axis=0)
    return counts_total


# TODO: check the factors (nb of meiosis between the parents/ between the two alleles of the individual...)
def roh_density_pedigree(x, nb_meiosis: int, comm_anc: int = 1, chr_lgts=chromosome_lengthsM_human):
    """Returns the expected ROH distribution within an individual, given the relatedness of its parents.
    Args:
        x: length [in Morgan] where to evaluate the density. Can be a float or an array of float
        nb_meiosis: length of the loop in the genealogy (ex: 4 for the offsping of siblings)
        comm_anc: nb of such paths (ex: 1 for half-siblings, 2 for full-siblings)
        chr_lgts: chromosome lengths [in Morgan]"""
    p_coal = coalescence_prob_pedigree(nb_meiosis, comm_anc)
    pdf = block_density(x, nb_meiosis, chr_lgts)
    return p_coal * pdf


# TODO: check the factors (nb of meiosis between the parents/ between the two alleles of the individual...)
def ibd_density_pedigree(x, nb_meiosis: int, comm_anc: int = 1, chr_lgts=chromosome_lengthsM_human):
    """Returns the expected IBD distribution between two individuals, given their pedigrees.
    Args:
        x: length [in Morgan] where to evaluate the density. Can be a float or an array of float
        nb_meiosis: length of the genealogic path between the two individuals
        comm_anc: nb of such paths
        chr_lgts: chromosome lengths [in Morgan]"""
    p_coal = coalescence_prob_pedigree(nb_meiosis, comm_anc)
    pdf = block_density(x, nb_meiosis, chr_lgts)
    return 4 * p_coal * pdf


# TODO: check the factors (nb of meiosis between the parents/ between the two alleles of the individual...)
def roh_count_pedigree(bins, nb_meiosis: int, comm_anc: int = 1, chr_lgts=chromosome_lengthsM_human):
    """Returns the expected number of ROH in a given interval, given the relatedness of its parents.
    Args:
        bins: tuple or array size (2,n) with bin edges [in Morgan]
        nb_meiosis: length of the genealogic path between its parents
        comm_anc: nb of such paths
        chr_lgts: chromosome lengths [in Morgan]"""
    p_coal = 2 * coalescence_prob_pedigree(nb_meiosis + 2, comm_anc)
    pdf = block_count(bins, nb_meiosis + 2, chr_lgts)
    return p_coal * pdf


# TODO: check the factors (nb of meiosis between the parents/ between the two alleles of the individual...)
def ibd_count_pedigree(bins, nb_meiosis: int, comm_anc: int = 1, chr_lgts=chromosome_lengthsM_human):
    """Returns the expected number of IBD in a given interval, given their relatedness.
    Args:
        bins: tuple or array size (2,n) with bin edges [in Morgan]
        nb_meiosis: length of the genealogic path between the two individuals
        comm_anc: nb of such paths
        chr_lgts: chromosome lengths [in Morgan]"""
    p_coal = coalescence_prob_pedigree(nb_meiosis, comm_anc)
    pdf = block_count(bins, nb_meiosis, chr_lgts)
    return 4 * p_coal * pdf


#### IBD across generations

def ibd_decay(t: np.ndarray, admix: np.ndarray, bins: np.ndarray,
                     lengths_ancestral: np.ndarray, nb_pairs_ancestral: float):
    """Returns the expected number of IBD in each bin, resulting from the decay of
    ancestral segments.
    Args:
        t: nb of generations between samples, shape (T,)
        admix: admixture coefficients, shape (A,)
        bins: bin edges, shape (B+1,)
        lengths_ancestral: IBD lengths observed in ancestral population, shape (N,)
        nb_pairs_ancestral: number of pairs observed in the ancestral population
    Returns:
        Array of shape (T, A, B) with expected IBD counts per bin.
    """
    t = np.asarray(t, dtype=float)                                  # (T,)
    admix = np.asarray(admix, dtype=float)                          # (A,)
    bins = np.asarray(bins, dtype=float)                            # (B,)
    lengths_ancestral = np.asarray(lengths_ancestral, dtype=float)  # (N,)

    above = lengths_ancestral[np.newaxis, :] >= bins[:, np.newaxis]     # (B, N)
    N_above = above.sum(axis=1)                                         # (B)
    S_above = (above * lengths_ancestral[np.newaxis, :]).sum(axis=1)    # (B)

    t_b = np.outer(t, bins)                                             # (T, B)
    integrand = np.exp(-t_b) * (np.outer(t, S_above) + (1-t_b) * N_above[np.newaxis, :])    # (T, B)

    ibd_per_pair = integrand[:, :-1] - integrand[:, 1:]                 # (T, B-1)

    return ibd_per_pair[:, np.newaxis, :] * admix[np.newaxis, :, np.newaxis] / nb_pairs_ancestral   # (T, A, B-1)
