from IBDecay.utils import chromosome_lengthsM_human

from collections.abc import Sequence

import os
import msprime, tskit
import pandas as pd
import warnings


def _resolve_demography(demography: msprime.Demography | None, Ne: int) -> msprime.Demography:
    """Build the default constant-size demography if none is given."""
    if demography is not None:
        return demography
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=Ne, default_sampling_time=0)
    return demography


def _prepare_save_path(save_path: str, overwrite: bool) -> None:
    """Create the parent directory for `save_path` and handle pre-existing files.

    Raises FileExistsError if the file already exists and `overwrite` is False;
    otherwise deletes any existing file so results start from a clean, single-header CSV.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        if not overwrite:
            raise FileExistsError(
                f"{save_path} already exists. Set overwrite=True to overwrite it."
            )
        os.remove(save_path)


def _get_roh_from_tree_seq(tree_sequence: tskit.TreeSequence, min_l: float) -> pd.DataFrame:
    """Extract IBD/ROH segments (length [Morgan] and TMRCA [generations]) between the two samples of a tree sequence."""
    ibd = tree_sequence.ibd_segments(min_span=min_l, store_segments=True).get((0, 1))
    if ibd is None:
        return pd.DataFrame(columns=["StartM", "EndM", "lengthM", "tmrca"])
    tmrca = tree_sequence.nodes_time[ibd.node]
    lengths = ibd.right - ibd.left
    return pd.DataFrame({"StartM":ibd.left, "EndM": ibd.right, "lengthM": lengths, "tmrca": tmrca})


def simulate_roh(
    n_sim: int = 10,
    min_l: float = 0.02,
    max_t: int = 1000,
    Ne: int = 5000,
    chr_lgts: Sequence[float] = chromosome_lengthsM_human,
    demography: msprime.Demography | None = None,
    seed: int | None = None,
    save_path: str | None = None,
    overwrite: bool = False,
) -> pd.DataFrame | None:
    """
    Simulate ROH within n_sim independent diploid individuals.

    Note: Even within an individual, chromosomes are simulated independently.

    If `save_path` is given, results are written incrementally to that CSV
    (raising FileExistsError if it already exists, unless `overwrite=True`)
    and this function returns None; otherwise a combined DataFrame of all
    results is returned.

    Args:
        n_sim: Number of independent diploid individuals to simulate.
        min_l: Minimum IBD segment length (Morgans) to report.
        max_t: Maximum time (generations) simulated back in the ancestry (msprime end_time).
            All segments which have not coalesced by then will be ignored, although they might pass the length filter.
        Ne: Effective population size for the default constant-size demography
            (ignored if `demography` is provided).
        chr_lgts: List of chromosome lengths (Morgans) to simulate.
        demography: Custom msprime.Demography; overrides `Ne` if given.
        seed: Base random seed (msprime requires seed >= 1).
        save_path: If given, path of the CSV file to write results to incrementally,
            instead of returning them in memory. Parent directories are created
            automatically if needed.
        overwrite: If `save_path` already exists, whether to overwrite it.
            Raises FileExistsError if the file exists and this is False.

    Returns:
        A DataFrame of all simulated ROH segments if `save_path` is None,
        otherwise None (results are written to disk instead).
    """
    demography = _resolve_demography(demography, Ne)

    if save_path is not None:
        _prepare_save_path(save_path, overwrite)

    df_all = []
    n_chr = len(chr_lgts)
    for chr_id, chr_l in enumerate(chr_lgts):
        print(f"Simulating chromosome {chr_id + 1}/{n_chr}", flush=True)
        # Simulate n_sim independent replicates of a single diploid individual's
        # ancestry for this chromosome in one call, instead of one call per replicate.
        replicates = msprime.sim_ancestry(
            samples=1, demography=demography,
            ploidy=2, sequence_length=chr_l, discrete_genome=False,
            recombination_rate=1,
            end_time=max_t,
            num_replicates=n_sim,
            random_seed=None if seed is None else seed + chr_id,
        )

        for sim_id, sim in enumerate(replicates):
            df = _get_roh_from_tree_seq(sim, min_l)
            df["iid"] = sim_id
            df["chr"] = chr_id
            if save_path is not None:
                df.to_csv(save_path, index=False, header=(sim_id == 0 and chr_id == 0), mode="a")
            else:
                df_all.append(df)

    if save_path is not None:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        return pd.concat(df_all, ignore_index=True)


def simulate_ibd(
    n_sim: int = 10,
    min_l: float = 0.02,
    max_t: int = 1000,
    Ne: int = 5000,
    chr_lgts: Sequence[float] = chromosome_lengthsM_human,
    t1: float = 0,
    t2: float = 0,
    pop1: str = "A",
    pop2: str = "A",
    ploidy: int = 2,
    demography: msprime.Demography | None = None,
    seed: int | None = None,
    save_path: str | None = None,
    overwrite: bool = False,
) -> pd.DataFrame | None:
    """
    Simulate IBD between n_sim independent pairs of individuals sampled at times
    t1 and t2 (generations ago).

    If `save_path` is given, results are written incrementally to that CSV
    (raising FileExistsError if it already exists, unless `overwrite=True`)
    and this function returns None; otherwise a combined DataFrame of all
    results is returned.

    Args:
        t1 and t2: Sampling time (generations ago) of the two individuals.
        pop1 and pop2: Populations of the two individuals. Only needed when using a custom demography.
        ploidy: Ploidy of both sampled individuals.
        min_l: Minimum IBD segment length (Morgans) to report.
        max_t: Maximum time (generations) simulated back in the ancestry (msprime end_time).
            All segments which have not coalesced by then will be ignored, although they might pass the length filter.
        Ne: Effective population size for the default constant-size demography
            (ignored if `demography` is provided).
        chr_lgts: List of chromosome lengths (Morgans) to simulate.
        demography: Custom msprime.Demography; overrides `Ne` if given.
        seed: Base random seed (msprime requires seed >= 1).
        save_path: If given, path of the CSV file to write results to incrementally,
            instead of returning them in memory. Parent directories are created
            automatically if needed.
        overwrite: If `save_path` already exists, whether to overwrite it.
            Raises FileExistsError if the file exists and this is False.

    Returns:
        A DataFrame of all simulated IBD segments if `save_path` is None,
        otherwise None (results are written to disk instead).
    """
    demography = _resolve_demography(demography, Ne)

    if save_path is not None:
        _prepare_save_path(save_path, overwrite)

    samples = (
            msprime.SampleSet(1, population=pop1, ploidy=ploidy, time=t1),
            msprime.SampleSet(1, population=pop2, ploidy=ploidy, time=t2),
        )

    df_all = []
    n_chr = len(chr_lgts)
    for chr_id, chr_l in enumerate(chr_lgts):
        print(f"Simulating {n_sim} pairs for chromosome {chr_id + 1}/{n_chr}", flush=True)
        replicates = msprime.sim_ancestry(
            samples=samples, demography=demography,
            ploidy=ploidy, sequence_length=chr_l, discrete_genome=False,
            recombination_rate=1,
            end_time=max_t,
            num_replicates=n_sim,
            random_seed=None if seed is None else seed + chr_id,
        )

        for sim_id, sim in enumerate(replicates):
            for chrom_1 in range(ploidy):
                for chrom_2 in range(ploidy):
                    tree_seq = sim.simplify(samples=(chrom_1, ploidy + chrom_2))
                    df = _get_roh_from_tree_seq(tree_seq, min_l)
                    df["sim"] = sim_id
                    df["iid1"] = f"{sim_id}_1"
                    df["iid2"] = f"{sim_id}_2"
                    df["chr"] = chr_id
                    df["chrom_1"] = chrom_1
                    df["chrom_2"] = chrom_2
                    if save_path is not None:
                        df.to_csv(
                            save_path, index=False,
                            header=(sim_id == 0 and chr_id == 0 and chrom_1 == 0 and chrom_2 == 0),
                            mode="a",
                        )
                    else:
                        df_all.append(df)

    if save_path is not None:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        return pd.concat(df_all, ignore_index=True)
