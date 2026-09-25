"""Verify that the module simulations creates ROH/IBD results with the correct format"""

import pandas as pd
import pytest
from IBDecay.simulations import (
    simulate_ibd,
    simulate_roh,
)

# Tiny parameters shared by the integration tests, to keep msprime simulations fast.
SMALL_CHR_LGTS = (2, 1)
SMALL_NE = 100
SMALL_MAX_T = 50
SMALL_N_SIM = 5


# ---------------------------------------------------------------------------
# simulate_roh (integration: exercises real msprime simulations)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSimulateRoh:
    REQUIRED_ROH_COLUMNS = frozenset(
        {"StartM", "EndM", "lengthM", "tmrca", "iid", "chr"}
    )

    def test_returns_dataframe_with_expected_columns(self):
        df = simulate_roh(
            n_sim=SMALL_N_SIM,
            chr_lgts=SMALL_CHR_LGTS,
            Ne=SMALL_NE,
            max_t=SMALL_MAX_T,
            seed=1,
        )
        assert set(df.columns) >= self.REQUIRED_ROH_COLUMNS

    def test_same_seed_is_reproducible(self):
        kwargs = {
            "n_sim": SMALL_N_SIM,
            "chr_lgts": SMALL_CHR_LGTS,
            "Ne": SMALL_NE,
            "max_t": SMALL_MAX_T,
            "seed": 42,
        }

        df1 = simulate_roh(**kwargs)
        df2 = simulate_roh(**kwargs)

        pd.testing.assert_frame_equal(df1, df2)

    def test_save_path_writes_csv(self, tmp_path):
        save_path = tmp_path / "roh.csv"

        result = simulate_roh(
            n_sim=2,
            chr_lgts=SMALL_CHR_LGTS,
            Ne=SMALL_NE,
            max_t=SMALL_MAX_T,
            seed=7,
            save_path=str(save_path),
        )

        assert result is None
        assert save_path.exists()
        df = pd.read_csv(save_path)
        assert set(df.columns) >= self.REQUIRED_ROH_COLUMNS


# ---------------------------------------------------------------------------
# simulate_ibd (integration: exercises real msprime simulations)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSimulateIbd:
    REQUIRED_IBD_COLUMNS = frozenset(
        {"StartM", "EndM", "lengthM", "tmrca", "iid1", "iid2", "chr"}
    )

    def test_returns_dataframe_with_expected_columns(self):
        n_sim = 2
        df = simulate_ibd(
            t1=0,
            t2=0,
            n_sim=n_sim,
            ploidy=2,
            chr_lgts=SMALL_CHR_LGTS,
            Ne=SMALL_NE,
            max_t=SMALL_MAX_T,
            seed=1,
        )
        assert set(df.columns) >= self.REQUIRED_IBD_COLUMNS

    def test_same_seed_is_reproducible(self):
        kwargs = {
            "t1": 0,
            "t2": 10,
            "n_sim": 2,
            "ploidy": 2,
            "chr_lgts": SMALL_CHR_LGTS,
            "Ne": SMALL_NE,
            "max_t": SMALL_MAX_T,
            "seed": 42,
        }

        df1 = simulate_ibd(**kwargs)
        df2 = simulate_ibd(**kwargs)

        pd.testing.assert_frame_equal(df1, df2)

    def test_save_path_writes_csv(self, tmp_path):
        save_path = tmp_path / "ibd.csv"

        result = simulate_ibd(
            t1=0,
            t2=0,
            n_sim=1,
            ploidy=2,
            chr_lgts=SMALL_CHR_LGTS,
            Ne=SMALL_NE,
            max_t=SMALL_MAX_T,
            seed=7,
            save_path=str(save_path),
        )

        assert result is None
        assert save_path.exists()
        df = pd.read_csv(save_path)
        assert set(df.columns) >= self.REQUIRED_IBD_COLUMNS
