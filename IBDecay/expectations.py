from IBDecay.utils import chromosome_lengthsM_human

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, brentq

class Calculator:
    """Class that calculates expected ROH/IBD"""

    def __init__(self, chr_lgts=chromosome_lengthsM_human):
        self.chr_lgts = chr_lgts

    def _roh_density_Ne_chr(self, x, Ne: float, chr_l:float):
        """Returns the expected ROH distribution for one chromosome, given an effective population size Ne.
        Args:
            x: Length [in Morgan] where to evaluate the density. Can be a float or an array of float
            Ne: effective population size
            chr_l: length of the chromosome [in Morgan]"""
        a = 2*x + 1/(2*Ne)
        inside = 4*(chr_l-x) / a**3
        edge = 2 / a**2
        return (inside + edge) / Ne * (0<x) * (x<chr_l)

    def roh_density_Ne(self, x, Ne:float):
        """"Returns the expected ROH distribution, given an effective population size Ne.
        Args:
            x: Length [in Morgan] where to evaluate the density. Can be a float or an array of float
            Ne: effective population size"""
        pdfs = [self._roh_density_Ne_chr(x, Ne, chr_l) for chr_l in self.chr_lgts]
        pdf_full = np.sum(pdfs, axis=0)
        return pdf_full

    def _roh_count_Ne_chr(self, min_l, Ne:float, chr_l:float):
        """Returns the expected number of ROHs longer than min_l, given an effective population size Ne.
        Args:
            min_l: minimum length [in Morgan]. Can be a float or an array of float
            Ne: effective population size
            chr_l: length of the chromosome [in Morgan]"""
        def F(x):
            inside = (1 + 8*Ne*x - 4*chr_l*Ne) / (1 + 8*Ne*x + 16*(Ne*x)**2)
            edge = -2 / (1 + 4*Ne*x)
            return inside + edge
        return (F(chr_l) - F(min_l)) * (0<min_l) * (min_l<chr_l)
    
    def roh_count_Ne(self, min_l, Ne:float):
        """"Returns the expected number of ROH blocks longer than min_l for the whole genome, given an effective population size.
        Args:
            min_l: minimum length [in Morgan]. Can be a float or an array of float
            Ne: effective population size"""
        totals = [self._roh_count_Ne_chr(min_l, Ne, chr_l) for chr_l in self.chr_lgts]
        total_full = np.sum(totals, axis=0)
        return total_full

    def ibd_density_Ne(self, x, Ne:float):
        """"Returns the expected IBD length distribution for the whole genome, given an effective population size.
        Args:
            x: Length [in Morgan] where to evaluate the density. Can be a float or an array of float.
            Ne: effective population size."""
        return 4 * self.roh_density_Ne(x, Ne)

    def ibd_count_Ne(self, min_l, Ne:float):
        """"Returns the expected number of IBD blocks longer than min_l for the whole genome, given an effective population size.
        Args:
            min_l: minimum length [in Morgan]. Can be a float or an array of float
            Ne: effective population size"""
        return 4 * self.roh_count_Ne(min_l, Ne)

    def _block_density_chr(self, x, chr_l:float, nb_meiosis:float):
        """Returns the expected DNA length distribution for one chromosome, given a number of meiosis.
        Args:
            x: Length [in Morgan] where to evaluate the density. Can be a float or an array of float
            chr_l: length of the chromosome [in Morgan]
            nb_meiosis: nb of meiosis -> average nb of recombination per Morgan"""
        pdf = ((chr_l-x) * nb_meiosis**2 + nb_meiosis ) * np.exp(-nb_meiosis*x)
        return pdf * (0<x) * (x<chr_l)

    def block_density(self, x, nb_meiosis:float):
        """Returns the expected DNA length distribution for one chromosome, given a number of meiosis.
        Args:
            x: Length [in Morgan] where to evaluate the density. Can be a float or an array of float
            nb_meiosis: nb of meiosis -> average nb of recombination per Morgan"""
        pdfs = [self._block_density_chr(x, chr_l, nb_meiosis) for chr_l in self.chr_lgts]
        pdf_full = np.sum(pdfs, axis=0)
        return pdf_full

    def coalescence_prob_pedigree(self, nb_meiosis:int, comm_anc:int=1) -> float:
        """Returns the coalescence probability within an individual, given its pedigree.
        Args:
            nb_meiosis: nb of Meiosis, aka length of the inbreeding loop
            comm_anc: nb of common ancestors, aka number of such loops"""
        p_coal = 2 * comm_anc * (1 / 2) ** nb_meiosis
        return p_coal

    def roh_density_pedigree(self, x, nb_meiosis:int, comm_anc:int=1):
        """Returns the expected ROH distribution within an individual, given its pedigree.
        Args:
            x: Length [in Morgan] where to evaluate the density. Can be a float or an array of float
            nb_meiosis: nb of Meiosis, aka length of the inbreeding loop
            comm_anc: nb of common ancestors, aka number of such loops"""
        p_coal = self.coalescence_prob_pedigree(nb_meiosis, comm_anc)
        pdf = self.block_density(x, nb_meiosis)
        return p_coal * pdf

    def ibd_decay(self, t:np.ndarray, admix:np.ndarray, bins:list, lengths_0:np.ndarray, nb_pairs_0:float):
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

    def log_likelihood_Ne(self, Ne: float, observed_length:np.ndarray, data_type:Litteral['IBD', 'ROH'], nb_observations:float, min_l: float) -> float:
        """Calculates the log-likelihood for a given Ne, based on observed IBD/ROH lengths.
        Computation is done assuming independence between segments, using a Poisson point process model.
        Args:
            Ne: effective population size.
            observed_length: array containing the length of the observed IBD/ROH segments.
            data_type: type of data, either 'IBD' or 'ROH'.
            nb_observations: number of observations considered.
            min_l: minimal length of the IBD/ROH segments.
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
        observed_length = observed_length[observed_length > min_l]

        pdf_vals = density_func(observed_length, Ne)
        return np.sum(np.log(pdf_vals + 1e-30)) - nb_observations * integrale_func(min_l, Ne)

    def estimate_Ne(self, observed_length:np.ndarray, data_type:Litteral['IBD', 'ROH'], nb_observations:float, min_l: float,
                Ne_bounds=(10, 10e6)
            ) -> (float, (float, float)):
        """Estimates Ne and a 95% confidence interval using the maximum log likelihood.
        Args:
            observed_length: array containing the length of the observed IBD/ROH segments.
            data_type: type of data, either 'IBD' or 'ROH'.
            nb_observations: number of observations considered.
            min_l: minimal length of the IBD/ROH segments.
        Returns:
            The optimal Ne and the 95% confidence interval."""

        res = minimize_scalar(lambda Ne: -self.log_likelihood_Ne(Ne, observed_length, data_type, nb_observations, min_l), method='bounded', bounds=Ne_bounds)

        # get 95% CI with Wilks' theorem
        def root_func(Ne):
            return res.fun + self.log_likelihood_Ne(Ne, observed_length, data_type, nb_observations, min_l) + 3.84/2
        ci_lower = brentq(root_func, Ne_bounds[0], res.x - 1e-5, xtol=1e-5)
        ci_upper = brentq(root_func, res.x + 1e-5, Ne_bounds[1], xtol=1e-5)

        return res.x, (ci_lower, ci_upper)

# TODO
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
