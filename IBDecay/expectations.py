from IBDecay.utils import chromosome_lengthsM_human

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, brentq

from typing import Literal, Tuple

class Calculator:
    """Class that calculates expected ROH/IBD"""

    def __init__(self, chr_lgts=chromosome_lengthsM_human):
        self.chr_lgts = chr_lgts

#### ROH based on Ne
    def roh_density_Ne(self, x, Ne:float):
        """"Returns the expected ROH distribution, given an effective population size Ne.
        Args:
            x: length [in Morgan] where to evaluate the density. Can be a float or an array of float
            Ne: effective population size"""
        
        def roh_density_Ne_chr(x, Ne: float, chr_l:float):
            return 8 * Ne * (1+4*chr_l*Ne) / (1+4*x*Ne)**3  * (0<x) * (x<chr_l)

        pdfs = [roh_density_Ne_chr(x, Ne, chr_l) for chr_l in self.chr_lgts]
        pdf_total = np.sum(pdfs, axis=0)
        return pdf_total

    def roh_count_Ne(self, bins=(0, np.inf), Ne:float=100):
        """"Returns the expected number of ROH blocks in a given interval, given an effective population size.
        Args:
            bins: tuple or array size (2,n) with bin edges [in Morgan]
            Ne: effective population size"""
        
        def roh_count_Ne_chr(x, Ne:float, chr_l:float):
            if x <= 0 or x >= chr_l:    # necessary to avoid numerical issues when x is very small or very large (otherwise we get a 0/0 form)
                return 0
            return 8 * Ne * (chr_l-x) * (1+2*Ne*(chr_l+x)) / ((1+4*chr_l*Ne) * (1+4*x*Ne)**2)  * (0<x) * (x<chr_l)

        counts = [roh_count_Ne_chr(bins[0], Ne, chr_l)-roh_count_Ne_chr(bins[1], Ne, chr_l) for chr_l in self.chr_lgts]
        counts_total = np.sum(counts, axis=0)
        return counts_total

    def roh_sum_Ne(self, bins=(0, np.inf), Ne:float=100):
        """"Returns the expected summed length [in Morgan] of ROH blocks in a given interval, given an effective population size.
        Args:
            bins: tuple or array size (2,n) with bin edges [in Morgan]
            Ne: effective population size"""
        
        def roh_sum_Ne_chr(x, Ne:float, chr_l:float):
            return 8 * Ne * (chr_l-x) * (1+2*Ne*(chr_l+x)) / ((1+4*chr_l*Ne) * (1+4*x*Ne)**2)  * (0<x) * (x<chr_l)

        sums = [roh_sum_Ne_chr(bins[0], Ne, chr_l)-roh_sum_Ne_chr(bins[1], Ne, chr_l) for chr_l in self.chr_lgts]
        sums_total = np.sum(sums, axis=0)
        return sums_total

#### IBD based on Ne
    def ibd_density_Ne(self, x, Ne:float):
        """"Returns the expected IBD length distribution, given an effective population size.
        Args:
            x: length [in Morgan] where to evaluate the density. Can be a float or an array of float.
            Ne: effective population size."""
        return 4 * self.roh_density_Ne(x, Ne)

    def ibd_count_Ne(self, bins=(0, np.inf), Ne:float=100):
        """"Returns the expected number of IBD blocks in a given interval, given an effective population size.
        Args:
            bins: tuple or array size (2,n) with bin edges [in Morgan]
            Ne: effective population size"""
        return 4 * self.roh_count_Ne(bins, Ne)

    def ibd_sum_Ne(self, bins=(0, np.inf), Ne:float=100):
        """"Returns the expected summed length [in Morgan] of IBD blocks in a given interval, given an effective population size.
        Args:
            bins: tuple or array size (2,n) with bin edges [in Morgan]
            Ne: effective population size"""
        return 4 * self.roh_sum_Ne(bins, Ne)

#### HBD based on a pedigree
    def coalescence_prob_pedigree(self, nb_meiosis:int, comm_anc:int=1) -> float:
        """Returns the coalescence probability of two alleles, given a pedigree.
        Args:
            nb_meiosis: length of the genealogic path between the two alleles
            comm_anc: nb of such paths"""
        return comm_anc * 2 * (1 / 2) ** nb_meiosis # factor two because two potential ancestral alleles (diploid)

    def block_density(self, x, nb_meiosis:float):
        """Returns the expected DNA length distribution, given a number of meiosis.
        Args:
            x: length [in Morgan] where to evaluate the density. Can be a float or an array of float
            nb_meiosis: nb of meiosis -> average nb of recombination per Morgan"""

        def block_density_chr(x, nb_meiosis:float, chr_l:float):
            pdf = ((chr_l-x) * nb_meiosis**2 + nb_meiosis ) * np.exp(-nb_meiosis*x)
            return pdf * (0<x) * (x<chr_l)

        pdfs = [block_density_chr(x, nb_meiosis, chr_l) for chr_l in self.chr_lgts]
        pdf_total = np.sum(pdfs, axis=0)
        return pdf_total

    def block_count(self, bins, nb_meiosis:float):
        """Returns the expected nb of DNA segments in a given interval, given a number of meiosis.
        Args:
            bins: tuple or array size (2,n) with bin edges [in Morgan]
            nb_meiosis: nb of meiosis -> average nb of recombination per Morgan"""

        def block_count_chr(x, nb_meiosis:float, chr_l:float):
            count = (chr_l-x) * nb_meiosis * np.exp(-nb_meiosis*x)
            return count * (0<x) * (x<chr_l)

        counts = [block_count_chr(bins[0], nb_meiosis, chr_l)-block_count_chr(bins[1], nb_meiosis, chr_l) for chr_l in self.chr_lgts]
        counts_total = np.sum(counts, axis=0)
        return counts_total

# TODO: check the factors (nb of meiosis between the parents/ between the two alleles of the individual...)
    def roh_density_pedigree(self, x, nb_meiosis:int, comm_anc:int=1):
        """Returns the expected ROH distribution within an individual, given the relatedness of its parents.
        Args:
            x: length [in Morgan] where to evaluate the density. Can be a float or an array of float
            nb_meiosis: length of the loop in the genealogy (ex: 4 for the offsping of siblings)
            comm_an: nb of such paths (ex: 1 for half-siblings, 2 for full-siblings )"""
        p_coal = self.coalescence_prob_pedigree(nb_meiosis, comm_anc)
        pdf = self.block_density(x, nb_meiosis)
        return p_coal * pdf

# TODO: check the factors (nb of meiosis between the parents/ between the two alleles of the individual...)
    def ibd_density_pedigree(self, x, nb_meiosis:int, comm_anc:int=1):
        """Returns the expected IBD distribution between two individuals, given their pedigrees.
        Args:
            x: length [in Morgan] where to evaluate the density. Can be a float or an array of float
            nb_meiosis: length of the genealogic path between the two individuals
            comm_an: nb of such paths"""
        p_coal = self.coalescence_prob_pedigree(nb_meiosis, comm_anc)
        pdf = self.block_density(x, nb_meiosis)
        return 4 * p_coal * pdf

# TODO: check the factors (nb of meiosis between the parents/ between the two alleles of the individual...)
    def roh_count_pedigree(self, bins, nb_meiosis:int, comm_anc:int=1):
        """Returns the expected number of ROH in a given interval, given the relatedness of its parents.
        Args:
            bins: tuple or array size (2,n) with bin edges [in Morgan]
            nb_meiosis: length of the genealogic path between its parents
            comm_an: nb of such paths"""
        p_coal = 2 * self.coalescence_prob_pedigree(nb_meiosis+2, comm_anc)
        pdf = self.block_count(bins, nb_meiosis+2)
        return p_coal * pdf

# TODO: check the factors (nb of meiosis between the parents/ between the two alleles of the individual...)
    def ibd_count_pedigree(self, bins, nb_meiosis:int, comm_anc:int=1):
        """Returns the expected number of IBD in a given interval, given their relatedness.
        Args:
            bins: tuple or array size (2,n) with bin edges [in Morgan]
            nb_meiosis: length of the genealogic path between the two individuals
            comm_an: nb of such paths"""
        p_coal = self.coalescence_prob_pedigree(nb_meiosis, comm_anc)
        pdf = self.block_count(bins, nb_meiosis)
        return 4 * p_coal * pdf

#### IBD accross generations
# TODO fix this
    def ibd_decay(self, t:np.ndarray, admix:np.ndarray, bins:np.ndarray, lengths_0:np.ndarray, nb_pairs_0:float):
        """"Returns the expected IBD length distribution, after a certain number of generations.
        Args:
            t: nb of generations between the two populations. Can be a float or an array of float
            admix: admixture coefficient. Can be a float or an array of float
            bins: bins to discretize the IBD lengths.
            lengths_0: IBD lengths at time 0.
            nb_pairs_0: number of pairs at time 0.
        Returns:
            A 2D array of shape (len(t), len(admix))"""
        t = np.asarray(t)
        admix = np.asarray(admix)

        # discretization and cumsums approximatimg integrals
        bin_sizes = bins[1:] - bins[:-1]
        bin_mids = bins[:-1] + bin_sizes / 2
        x0 = np.histogram(lengths_0, bins=bins, density=False)[0] / nb_pairs_0
        cumsum1 = x0 * bin_sizes
        cumsum1 = np.cumsum(cumsum1[::-1])[::-1]
        cumsum2 = x0 * bin_mids * bin_sizes
        cumsum2 =  np.cumsum(cumsum2[::-1])[::-1]

        res = np.exp(-bin_mids * t[:, np.newaxis]) * (x0 + (2*t[:, np.newaxis] - t[:, np.newaxis]**2 * bin_mids)*cumsum1 + t[:, np.newaxis]**2 * cumsum2)
        res = res * admix[np.newaxis, :, np.newaxis]
        return res

class Estimator:
    """Class implementing the most likelihood estimation."""
    def __init__(self, chr_lgts=chromosome_lengthsM_human):
        self.chr_lgts = chr_lgts

    def log_likelihood_Ne(self, Ne: float, observed_length:np.ndarray, data_type:Literal['IBD', 'ROH'], nb_observations:float, bin: Tuple[float, float]) -> float:
        """Calculates the log-likelihood for a given Ne, based on observed IBD/ROH lengths.
        Computation is done assuming independence between segments, using a Poisson point process model.
        Args:
            Ne: effective population size.
            observed_length: array containing the length of the observed IBD/ROH segments.
            data_type: type of data, either 'IBD' or 'ROH'.
            nb_observations: number of observations considered.
            bin: tuple containing the lower and upper bounds for the length of the IBD/ROH segments.
        """
        calculator = Calculator(self.chr_lgts)
        if data_type == 'IBD':
            density_func = calculator.ibd_density_Ne
            integrale_func = calculator.ibd_count_Ne
        elif data_type == 'ROH':
            density_func = calculator.roh_density_Ne
            integrale_func = calculator.roh_count_Ne
        else:
            raise ValueError("data_type must be 'IBD' or 'ROH'")

        observed_length = np.asarray(observed_length)
        observed_length = observed_length[(observed_length > bin[0]) & (observed_length < bin[1])]
        pdf_vals = density_func(observed_length, Ne)
        return np.sum(np.log(pdf_vals + 1e-30)) - nb_observations * integrale_func(bin, Ne)

    def estimate_Ne(self, observed_length:np.ndarray, data_type:Literal['IBD', 'ROH'], nb_observations:float, bin: Tuple[float, float],
                Ne_bounds=(10, 10e6)
            ) -> (float, (float, float)):
        """Estimates Ne and a 95% confidence interval using the maximum log likelihood.
        Args:
            observed_length: array containing the length of the observed IBD/ROH segments.
            data_type: type of data, either 'IBD' or 'ROH'.
            nb_observations: number of observations considered.
            bin: tuple containing the lower and upper bounds for the length of the IBD/ROH segments.
        Returns:
            The optimal Ne and the 95% confidence interval."""

        res = minimize_scalar(lambda Ne: -self.log_likelihood_Ne(Ne, observed_length, data_type, nb_observations, bin), method='bounded', bounds=Ne_bounds)

        # get 95% CI with Wilks' theorem
        def root_func(Ne):
            return res.fun + self.log_likelihood_Ne(Ne, observed_length, data_type, nb_observations, bin) + 3.84/2
        ci_lower = brentq(root_func, Ne_bounds[0], res.x - 1e-5, xtol=1e-5)
        ci_upper = brentq(root_func, res.x + 1e-5, Ne_bounds[1], xtol=1e-5)

        return res.x, (ci_lower, ci_upper)

# TODO: adapt this
class Estimator_IBDecay:
    def __init__(self, df_0:pd.DataFrame, df_t:pd.DataFrame, nb_pairs_0:float=1, nb_pairs_t:float=1,
                 bins_size=0.01, x_min=0.04, x_max=0.2, bins=None):
        """Initializes the estimator.
        All IBD segments longer than x_min are used to compute the expectations, but only those within [x_min, x_max] are used for the likelihood.
        """
        if bins is None:
            bins = np.arange(x_min, df_0['lengthM'].max()+bins_size, bins_size)
        if x_max < bins[-1]:
            x_max_id = np.searchsorted(bins, x_max)
        else:
            x_max_id = len(bins)
        self.id_max = x_max_id

        self.calculator = Calculator_IBD(bins=bins, df_0=df_0, nb_pairs_0=nb_pairs_0)
        self.xt = np.histogram(df_t['lengthM'], bins=bins, density=False)[0]  # total IBD amount
        self.nb_pairs_t = nb_pairs_t

    def log_likelihood(self, t_grid, admix_grid):
        """Returns the log-likelihood of observing the data at time t since common ancestor.
        Args:
            t: time since common ancestor
            admix: proportion of admixture from a source with no shared ancestry (ie no IBD)"""
        t_grid = np.asarray(t_grid)
        admix_grid = np.asarray(admix_grid)
        expected = self.calculator.ibd_decay_analytics(t_grid)  # expected[time, bin]
        expected = expected[:, np.newaxis, :] * admix_grid[np.newaxis, :, np.newaxis]  # expected[time, admix, bin]
        ll = self.xt[np.newaxis, np.newaxis, :] * np.log(expected+1e-30) - self.nb_pairs_t * expected # ll[time, admix, bin]
        ll = ll[:, :, :self.id_max]
        ll = np.sum(ll, axis=2)  # ll[time, admix]
        ll -= np.max(ll)

        time_opt, admix_opt = np.unravel_index(np.argmax(ll), ll.shape)
        time_opt = t_grid[time_opt]
        admix_opt = admix_grid[admix_opt]
        return time_opt, admix_opt, ll
