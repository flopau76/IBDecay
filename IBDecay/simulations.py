from IBDecay.utils import chromosome_lengthsM_human

import os
import msprime, tskit
import numpy as np
import pandas as pd
from tqdm.auto import trange
import warnings

class Simulator:
    def __init__(self, Ne=5000, max_t=1000, min_l=0.02, chr_lgts=chromosome_lengthsM_human):
        self.max_t = max_t
        self.min_l = min_l

        self.demography = msprime.Demography()
        self.demography.add_population(name="A", initial_size=Ne, default_sampling_time=0)

        self.chr_lgts = chr_lgts

        self.seed = 0

    def _get_roh_from_tree_seq_old(self, tree_sequence: tskit.TreeSequence):
        """Extract vector of all ROH from a tree sequence containing only two samples"""
        res = []

        for tree in tree_sequence.trees():
            try:
                t_mrca = tree.tmrca(0, 1)
                roh_size = tree.span
                if roh_size >= self.min_l:
                    res.append((roh_size, t_mrca))
            except ValueError: # no TMRCA found (nodes stop)
                pass
        res = pd.DataFrame(res, columns=["lengthM", "tmrca"])
        return res

    def _get_roh_from_tree_seq(self, tree_sequence: tskit.TreeSequence):
        """Extract vector of all ROH from a tree sequence containing only two samples"""
        ibd = tree_sequence.ibd_segments(min_span=self.min_l, store_segments=True).get((0, 1))
        if ibd is None:
            return pd.DataFrame(columns=["lengthM", "tmrca"])
        tmrca = tree_sequence.nodes_time[ibd.node]
        lengths = ibd.right - ibd.left
        return pd.DataFrame({"lengthM": lengths, "tmrca": tmrca})

    def simulate_roh(self, n_sim:int=10, save_path=None) -> DataFrame[DataROH]|None:
        """Run n_sim simulations of ROH within a diploïd individual"""

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        df_all = []
        for sim_id in trange(n_sim, desc="Simulating individuals", disable=None):
            for (chr_id, chr_l) in enumerate(self.chr_lgts):
                # Simulate the ancestry of a single diploid individual
                sim: tskit.TreeSequence = msprime.sim_ancestry(samples=1, demography=self.demography,
                                                ploidy=2, sequence_length=chr_l, discrete_genome=False,
                                                recombination_rate=1,
                                                end_time=self.max_t)

                # Process each simulation
                df = self._get_roh_from_tree_seq(sim)
                df["iid"] = sim_id
                df["chr"] = chr_id
                if save_path is not None:
                    df.to_csv(save_path, index=False, header=(sim_id == 0 and chr_id == 0), mode="a")
                else:
                    df_all.append(df)
        if save_path is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                df_all = pd.concat(df_all, ignore_index=True)
            return df_all # type: ignore
        else:
            return None

    def simulate_ibd_decay(self, t1:float=0, t2:float=0, n_sim:int=10, ploidy:int=2, samples=None, save_path=None) -> DataFrame[DataIBD]|None:
        """Run n_sim simulations of IBD between two individuals from different time points t1 and t2 (generations ago).
        In total: n_sim * ploidy^2 pairs are compared"""

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if samples is None:
            samples = [
                msprime.SampleSet(1, population="A", ploidy=ploidy, time=t1),
                msprime.SampleSet(1, population="A", ploidy=ploidy, time=t2),
            ]

        df_all = []
        for sim_id in trange(n_sim, desc=f"Simulating {n_sim} pairs", disable=None): # type: ignore
            for (chr_id, chr_l) in enumerate(self.chr_lgts):
                # Simulate the ancestry of two individuals from different times
                sim = msprime.sim_ancestry(samples=samples, demography=self.demography,
                                                    ploidy=ploidy, sequence_length=chr_l, discrete_genome=False,
                                                    recombination_rate=1,
                                                    end_time=self.max_t)

                # Process each pair of chromosomes
                for chrom_1 in range(ploidy):
                    for chrom_2 in range(ploidy):
                        tree_seq = sim.simplify(samples=(chrom_1, ploidy + chrom_2))
                        df = self._get_roh_from_tree_seq(tree_seq)
                        df["sim"] = sim_id
                        df["iid1"] = f"{sim_id}_1"
                        df["iid2"] = f"{sim_id}_2"
                        df["chr"] = chr_id
                        df["chrom_1"] = chrom_1
                        df["chrom_2"] = chrom_2
                        if save_path is not None:
                            df.to_csv(save_path, index=False, header=(sim_id == 0 and chr_id == 0 and chrom_1 == 0 and chrom_2 == 0), mode="a")
                        else:
                            df_all.append(df)
        if save_path is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                df_all = pd.concat(df_all, ignore_index=True)
            return df_all # type: ignore
        else:
            return None