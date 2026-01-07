#___________________________________________________
# Necessary imports
#___________________________________________________
import os
import pandas as pd
import numpy as np

# type hinting:
import pandera.pandas as pa
from pandera.typing import Series, DataFrame

chromosome_lengthsM_human = [2.8426, 2.688187, 2.232549, 2.14201, 2.040477, 1.917145, 1.871491, 1.680018, 
            1.661367, 1.8090949, 1.5821669, 1.745901, 1.2551429, 1.1859521, 1.413411, 
            1.340264, 1.2849959, 1.175495, 1.0772971, 1.082123, 0.636394, 0.724438]

chromosome_lengths_macaque = [ 223616716, 196192209, 185287830, 169962632, 
                187314383, 179082969, 169867068, 145676177, 
                134123906, 99508945, 133063178, 130042031, 
                108732418, 128052760, 113274796, 79622849, 
                95424395, 74473312, 58315217, 77134767]

chromosome_lengthsM_macaque = [1.0426104, 0.9114943, 0.88774445, 0.9821343,
                0.8610165399999999, 0.8323678, 0.9808, 0.6987766000000001,
                0.7982016000000001, 0.56821987, 0.7563325000000001, 0.60809906,
                0.63967335, 0.60983124, 0.6090937000000001, 0.48101673,
                0.6244356, 0.4388225, 1.0864966, 0.43828705]

#___________________________________________________
# Data Format
#___________________________________________________

class DataROH(pa.DataFrameModel):
    iid: Series
    StartM: Series[float]       # in Morgans
    EndM: Series[float]         # in Morgans
    length: Series[int]
    lengthM: Series[float]      # in Morgans
    ch: Series[int]
    StartBP: Series[int]
    EndBP: Series[int]

class DataIBD(pa.DataFrameModel):
    Start: Series[int]
    End: Series[int]
    StartM: Series[float]       # in Morgans
    EndM: Series[float]         # in Morgans
    length: Series[int]
    lengthM: Series[float]      # in Morgans
    ch: Series[int]
    StartBP: Series[int]
    EndBP: Series[int]
    iid1: Series
    iid2: Series
    SNP_Dens: Series[float]

class DataIBDInd(pa.DataFrameModel):
    iid1: Series
    iid2: Series
    max_IBD: Series[float]
    sum_IBD_8: Series[float]    # sum_IBD>8
    n_IBD_8: Series[int]        # n_IBD>8
    sum_IBD_12: Series[float]
    n_IBD_12: Series[int]
    sum_IBD_16: Series[float]
    n_IBD_16: Series[int]
    sum_IBD_20: Series[float]
    n_IBD_20: Series[int]

class DataMeta(pa.DataFrameModel):
    Master_ID: Series
    iid: Series
    frac_gp: Series[float]
    frac_missing: Series[float]
    frac_het: Series[float]
    n_cov_snp: Series[int]
    Archaeological_ID: Series
    Projects: Series
    Locality: Series
    Province: Series
    Country: Series
    Latitude: Series[float]
    Longitude: Series[float]
    date: Series
    date_type: Series
    imputation_type: Series

#___________________________________________________
# Data manipulation
#___________________________________________________

class DataHandler:
    def __init__(self, df_meta: pd.DataFrame, df_ibd: pd.DataFrame, df_ibd_ind: pd.DataFrame):
        self.df_meta = df_meta
        self.df_ibd = df_ibd
        self.df_ibd_ind = df_ibd_ind

    def get_iids_meta(self, iids) -> set[str]:
        """Return the list of iids present in the metadata."""
        return set(self.df_meta[self.df_meta['iid'].isin(iids)]['iid'])

    def filter_quality(self, iids, filters=[("frac_gp", 0.7, 1.0)]) -> set[str]:
        """Filter iids based on specified column ranges in df_meta."""
        filter = self.df_meta['iid'].isin(iids)
        for (col, min, max) in filters:
            filter &= self.df_meta[col].between(min, max)
        return set(self.df_meta[filter]['iid'])

    def filter_time(self, iids, min_date:int, max_date:int, strict:bool=False) -> set[str]:
        """Filter the iids whose dating correspond to the given time_range, based on the metadata.
        If strict, the mean date must be within the range.
        If not strict, the dating range must overlap with the given time_range."""
        dates = self.df_meta[['iid', 'date']][self.df_meta['iid'].isin(iids)]
        dates[['min', 'max']] = dates['date'].str.split(':', expand=True).astype(float)
        dates['mean'] = dates[['min', 'max']].mean(axis=1)

        if strict:
            # individuals whose mean datation is within the time_range
            valid = ( (dates['min'] <= max_date) & (dates['max'] >= min_date) )
        else:
            # individuals whose timespan overlaps with the given time_range
            valid = ( (dates['min'] >= min_date) & (dates['min'] <= max_date) ) | \
                    ( (dates['max'] >= min_date) & (dates['max'] <= max_date) ) | \
                    ( (dates['min'] <= min_date) & (dates['max'] >= max_date) )
        return set(dates['iid'][valid])

    def get_iids_site(self, site:str) -> set[str]:
        """Returns the iids corresponding to a given site, based on the metadata."""
        return set(self.df_meta[self.df_meta['iid'].str.startswith(site)]['iid'])
    
    def get_ibd_iids(self, iids1, iids2=None, filter_rel=("sum_IBD>12", 0, 100)) -> tuple[int, pd.DataFrame]:
        """Returns the IBD segments between two sets of iids, and the number of individual pairs concerned."""
        if iids2 is None:
            iids2 = iids1
        iids1 = self.get_iids_meta(iids1)
        iids2 = self.get_iids_meta(iids2)

        subset = self.df_ibd_ind[
                (self.df_ibd_ind['iid1'].isin(iids1) & self.df_ibd_ind['iid2'].isin(iids2)) |
                (self.df_ibd_ind['iid1'].isin(iids2) & self.df_ibd_ind['iid2'].isin(iids1))
            ]

        if filter_rel is not None:
            col, min, max = filter_rel
            pairs = subset[['iid1', 'iid2']][(subset[col].between(min, max))]
        else:
            pairs = subset[['iid1', 'iid2']]

        nb_removed = len(subset) - len(pairs)
        df_ibd_filtered = self.df_ibd.merge(pairs, on=['iid1', 'iid2'], how='inner')

        if iids1 == iids2:
            return len(iids1)*(len(iids1)-1)//2 - nb_removed, df_ibd_filtered
        else:
            return len(iids1)*len(iids2) - nb_removed, df_ibd_filtered
    
    def get_ibd_sites(self, site1: str, site2: str|None=None, filter_rel=("sum_IBD>12", 0, 100)) -> tuple[int, pd.DataFrame]:
        """Returns the IBD segments between two sites, and the number of individual pairs concerned."""
        iids1 = self.get_iids_site(site1)
        if site2 is None:
            iids2 = None
        else:
            iids2 = self.get_iids_site(site2)
        return self.get_ibd_iids(iids1, iids2, filter_rel)

#___________________________________________________
# Summary Stats:
#___________________________________________________
def get_stats(df:DataFrame[DataIBD], L=[8,12,16,20], data_type:Literal['IBD', 'ROH']='IBD', save:None|str=None) -> DataFrame[DataIBDInd]:
    """Compute summary stats for the pandas df_RG"""
    if data_type == 'IBD':
        id_col = ['iid1', 'iid2']
    elif data_type == 'ROH':
        id_col = ['iid']
    else:
        raise ValueError("data_type must be 'IBD' or 'ROH'")
    df_stats = pd.DataFrame()
    df_stats["max"] = df.groupby(id_col, observed=False)["lengthM"].max()
    for n in L:
        groups = df.loc[df['lengthM'] > 0.01*n].groupby(id_col, observed=False)
        n_roh = groups.size()
        sum_roh = groups["lengthM"].sum()
        df_stats[f"count_{data_type}>{n}"] = n_roh
        df_stats[f"sum_{data_type}>{n}"] = sum_roh
    df_stats = df_stats.fillna(0).reset_index()
    if save is not None:
        df_stats.to_csv(save, index=False)
    return df_stats.sort_values(by=f"sum_{data_type}>{L[0]}", ascending=False) # type: ignore