from IBDecay.utils import chromosome_lengthsM_human, DataIBD
from pandera.typing import DataFrame

import numpy as np

class Calculator_ROH:
    """Class that calculates expected ROH
    Note: all length must be morgans"""
    #### Length of all Chromosomes (between first and last 1240k SNP)

    def __init__(self, chr_lgts=chromosome_lengthsM_human):
        self.chr_lgts = chr_lgts

    def _roh_density_Ne_chr(self, x, Ne: float, chr_l:float):
        """Returns the expected RoH density for one chromosome, given an effective population size.
        Args:
            x: Length [in Morgan] where to evaluate the density. Can be a float or an array of float
            Ne: effective population size
            chr_l: length of the chromosome [in Morgan]"""
        a = 2*x + 1/(2*Ne)
        inside = 4*(chr_l-x)/ Ne / a**3
        edge = 2 / Ne / a**2
        return (inside + edge) * (0<x) * (x<chr_l)

    def roh_density_Ne(self, x, Ne:float):
        """"Returns the expected RoH density for the whole genome, given an effective population size.
        Args:
            x: Length [in Morgan] where to evaluate the density. Can be a float or an array of float
            Ne: effective population size"""
        pdfs = [self._roh_density_Ne_chr(x, Ne, chr_l) for chr_l in self.chr_lgts]
        pdf_full = np.sum(pdfs, axis=0)
        return pdf_full

    def _block_density_chr(self, x, chr_l:float, m:float):
        """Returns the expected block density after m meiosis, for one chromosome.
        Args:
            x: Length [in Morgan] where to evaluate the density. Can be a float or an array of float
            chr_l: length of the chromosome [in Morgan]
            m: nb of meiosis -> average nb of recombination per Morgan"""
        pdf = ((chr_l-x) * m**2 + m ) * np.exp(-m*x)
        return pdf * (0<x) * (x<chr_l)

    def block_density(self, x, m:float):
        """Returns the expected block density after m meiosis, for the whole genome.
        Args:
            x: Length [in Morgan] where to evaluate the density. Can be a float or an array of float
            m: nb of meiosis -> average nb of recombination per Morgan"""
        pdfs = [self._block_density_chr(x, chr_l, m) for chr_l in self.chr_lgts]
        pdf_full = np.sum(pdfs, axis=0)
        return pdf_full

    def roh_prob_pedigree(self, m:int, comm_anc:int=1) -> float:
        """Returns the RoH probability within an individual, given its pedigree.
        Args:
            m: nb of Meiosis, aka length of the inbreeding loop
            comm_anc: nb of common ancestors, aka nb of loops"""
        c_prob = 2 * comm_anc * (1 / 2) ** m
        return c_prob

    def roh_density_pedigree(self, x, m:int, comm_anc:int=1):
        """Returns the density of RoH blocks of length x [in Morgan], given a pedigree.
        Args:
            x: Length [in Morgan] where to evaluate the density. Can be a float or an array of float
            m: nb of meiosis -> average nb of recombination per Morgan
            comm_anc: nb of common ancestors, aka nb of loops"""
        p_roh = self.roh_prob_pedigree(m, comm_anc)
        pdf = self.block_density(x, m)
        return p_roh * pdf

class Calculator_IBD:
    """Class that calculates expected IBD amount over time"""

    def __init__(self, bins, df_0:DataFrame[DataIBD], nb_pairs_0:float=1):
        self.bins = bins
        self.bin_sizes = self.bins[1:] - self.bins[:-1]
        self.bin_mids = self.bins[:-1] + self.bin_sizes / 2

        self.x0 = np.histogram(df_0['lengthM'], bins=bins, density=False)[0] / nb_pairs_0

        # precomputation of cumulative sums
        self._cumsum1 = self.x0 * self.bin_sizes
        self._cumsum1 = np.cumsum(self._cumsum1[::-1])[::-1]
        self._cumsum2 = self.x0 * self.bin_mids * self.bin_sizes
        self._cumsum2 =  np.cumsum(self._cumsum2[::-1])[::-1]

    def ibd_decay_analytics(self, t_grid):
        """Computes the expected IBD amount for each bin, at different time points.
        Returns an array of shape (len(t_grid), len(bins)-1)."""
        t_grid = np.asarray(t_grid)
        return np.exp(-self.bin_mids * t_grid[:, np.newaxis]) * (self.x0 + (2*t_grid[:, np.newaxis] - t_grid[:, np.newaxis]**2 * self.bin_mids)*self._cumsum1 + t_grid[:, np.newaxis]**2 * self._cumsum2)

class Estimator_IBD:
    """Class that estimates the time since common ancestor, given observed IBD lengths."""

    def __init__(self, df_0:DataFrame[DataIBD], df_t:DataFrame[DataIBD], nb_pairs_0:float=1, nb_pairs_t:float=1,
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
