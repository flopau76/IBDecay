from IBDecay.expectations import Calculator
from IBDecay.utils import chromosome_lengthsM_human

from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patheffects as pe
from matplotlib import gridspec
from matplotlib.patches import Patch
from matplotlib.axes import Axes
from matplotlib.markers import MarkerStyle

class Plotter:
    def __init__(self, chr_lgts=chromosome_lengthsM_human):
        self.chr_lgts = chr_lgts

        # colors for expected lines
        self.Ne_colors = ["#fde725", "#5ec962", "#21918c", "#3b528b", "#440154", "k"]
        self.delta_t_colors = ["#fde725", "#5ec962", "#21918c", "#3b528b", "#440154", "k"]
        self.pedigree_colors = ['red', 'green', 'blue', 'purple', 'brown']

        # histogramm colors
        self.kwargs_histo_one_site = {"color": "sandybrown", "edgecolor": "gray", "alpha": 1}
        self.kwargs_histo_two_site = {"site_a" : {"color": "blue", "edgecolor": "gray", "alpha": 0.6},
                                "site_b" : {"color": "violet", "edgecolor": "gray", "alpha": 0.6},
                                "cross"  : {"color": "lime", "edgecolor": "gray", "alpha": 0.6}}

#___________________________________________________
# Histograms at population levels
#___________________________________________________
    def plot_histo(self, df_data:pd.DataFrame, nb_normalize: int=1, bins=np.arange(0.08, 0.30, 0.005),
            data_type:Literal['IBD', 'ROH']='IBD', Ne:list[int]=[1500, 3000, 5000],
            xlabel:str|None=None, ylabel:str|None=None
        ):
        """Plot data histogram.
        Args:
            df_data: Dataframe containing the data to plot. Must contain a column 'lengthM'.
            nb_normalize: Number of pairs (for IBD) or individuals (for ROH) to normalize the histogram.
            bins: Bins to use for the histogram.
            data_type: 'IBD' or 'ROH'.
            Ne: List of effective population sizes to plot the expected lines for."""
        fig, ax = plt.subplots()

        # Format the plot
        if xlabel is None:
            xlabel = f"{data_type} segment length (M)"
        if ylabel is None:
            ylabel = f"Average nb of {data_type} segments per {'pair' if data_type=='IBD' else 'individual'}"
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(bins[0], bins[-1]+(bins[1]-bins[0]))
        ax.set_yscale('log')

        # Plot the actual histogram
        ax.hist(df_data['lengthM'], bins=bins, weights=np.full(len(df_data), 1/nb_normalize), **self.kwargs_histo_one_site)

        # Plot the expected histogram for a constant Ne
        calculator = Calculator(self.chr_lgts)
        segment_density_func = calculator.ibd_density_Ne if data_type=='IBD' else calculator.roh_density_Ne
        x = np.linspace(bins[0], bins[-1], 1000)
        bin_width = bins[1] - bins[0]
        for N, c in zip(Ne, self.Ne_colors):
            y = bin_width * segment_density_func(x, N)
            ax.plot(x, y, color=c, linestyle='dashed', scaley=False)
            ax.text(x[100], y[100], f"Ne={int(N)}", color='black')

        return fig, ax

    # TODO
    def _plot_ibd_two_sites(self, df_ibd1:pd.DataFrame, df_ibd2:pd.DataFrame, df_ibd_cross:pd.DataFrame,
                            nb_pairs_1: int, nb_pairs_2: int, nb_pairs_cross: int, bins=np.arange(0.08, 0.30, 0.005),
                            Ne:list[int]=[1500, 3000, 5000], delta_t:list[int]=[],
                            name_site1:str="", name_site2:str=""
        ):
        """Plot IBD Histogram comparing two sites."""
        fig, ax = plt.subplots()

        # Plot the actual histogram
        ax.hist(df_ibd1['lengthM'], bins=bins, weights=np.full(len(df_ibd1), 1/nb_pairs_1), label=name_site1, **self.kwargs_histo_two_site["site_a"])
        ax.hist(df_ibd2['lengthM'], bins=bins, weights=np.full(len(df_ibd2), 1/nb_pairs_2), label=name_site2, **self.kwargs_histo_two_site["site_b"])
        ax.hist(df_ibd_cross['lengthM'], bins=bins, weights=np.full(len(df_ibd_cross), 1/nb_pairs_cross), label="between", **self.kwargs_histo_two_site["cross"])

        plt.legend(loc='upper right', title="IBD Sharing")
        ax.set_xlim(bins[0], bins[-1]+(bins[1]-bins[0]))
        self._format_ax(ax, 'IBD segment length (cM)', 'Average nb of IBD segments per pair')

        ### Expectations
        # IBD decay
        calculator = Calculator_IBD(bins=bins, df_0=df_ibd1)
        bin_mids = bins[:-1] + (bins[1:] - bins[:-1]) / 2
        bin_size = bins[1:] - bins[:-1]
        xdt = calculator_ibd.ibd_decay_analytics(delta_t) / nb_pairs_1
        for i, (dt, c) in enumerate(zip(delta_t, self.delta_t_colors)):
            ax.plot(bin_mids, xdt[i, :], color=c, linestyle='solid', scaley=False)
            ax.text(bin_mids[1], xdt[i, 1], f"dt={dt}", color='black', rotation=-40, rotation_mode='anchor')
        # Constant Ne
        calculator_roh = Calculator_ROH()
        for N, c in zip(Ne, self.Ne_colors):
            y = 4 * calculator_roh.roh_density_Ne(bin_mids, N) * bin_size
            ax.plot(bin_mids, y, color=c, linestyle='dashed', scaley=False)
            ax.text(bin_mids[0], y[0], f"Ne={N}", color='black')

        return fig
#___________________________________________________
# Summary stats with one bar per inividual
#___________________________________________________
    def plot_summary_stats(self, df_stats:pd.DataFrame, L=[8,12,16,20], L_colors=["#313695", "#abd9e9", "#fee090", "#d7191c"],
                legend:bool=True, x_ticks:None|str=None, y_ticks:bool=False, ax=None):
        """Plot the distribution of roh summary stats as a bar plot.
        Args:
            df_stats can be obtained by running create_stats(df_roh, L)
            L is the list of thresholds [in cM] considered (ie amount/length of roh >= x with x in L)
            legend: boolean, wether to plot legend or not
            x_ticks: if str, column of dataframe to use as x_ticks
            y_ticks: boolean wether to tick y axis or not"""

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        # Prepare data
        df_stats = df_stats.sort_values(f"sum_ROH>{L[0]}", ascending=False)
        data = 100 * df_stats[[f"sum_ROH>{n}" for n in L]].values
        x = np.arange(len(df_stats))
        bottom = np.zeros((len(df_stats), len(L)))
        for i in range(1,len(L)):
            bottom[:,i] = data[:,0]-data[:,i]

        # Make plot for each bin
        for i in range(len(L)):
            ax.bar(x, data[:,i], bottom=bottom[:,i], width=0.8, color=L_colors[i], edgecolor="black", label=f"{L[i]}-{L[i+1]} cM" if i<len(L)-1 else f">{L[i]} cM")

        if legend:
            ax.legend(title="Sum of ROH in")
        if x_ticks:
            ax.set_xticks(x)
            ax.set_xticklabels(df_stats[x_ticks], rotation=270)
        else:
            ax.tick_params(labelbottom = False)
        ax.tick_params(axis='x', which='both', bottom=False, top=False)

        if y_ticks:
            ax.set_ylabel("Cumulative length of ROH [cM]")
        else:
            ax.tick_params(left = False, labelleft = False)
        ax.set_xlim(-1, len(df_stats))

        return fig, ax

    def plot_summary_stats_panel(self, df_stats:List[pd.DataFrame], L=[8,12,16,20], L_colors=["#313695", "#abd9e9", "#fee090", "#d7191c"], titles:List[str]=[],
                        figsize=None, x_ticks:None|str=None):
        """Plot the distribution of multiple roh summary stats into different panels.
        Args:
            df_stats: list of dataframes containing the stats to plot. Each dataframe can be obtained by running create_stats(df_roh, L) on a different subset of data (eg different subspecies
            titles: list of titles for each panel
            For other arguments, refer to plot_summary_stats()"""
        if figsize is None:
            figsize = (6*len(df_stats),6)
        fig, axes = plt.subplots(1, len(df_stats), figsize=figsize, sharey=True, width_ratios=[len(df) for df in df_stats])
        for i, df in enumerate(df_stats):
            title = titles[i] if i < len(titles) else ""
            self.plot_summary_stats(df, L=L, L_colors=L_colors, legend=(i==len(df_stats)-1), x_ticks=x_ticks, y_ticks=(i==0), ax=axes[i])
            axes[i].set_title(title)
        return fig, axes

#___________________________________________________
# Chromosome details at individual level
#___________________________________________________
    def _plot_single_chromosome(self, ax:Axes, pos_x:float, chrom_length:float, df_RG, df_RG2, unit:Literal['BP', 'Morgans']='Morgans'):
        """Plot a Chromosome of length l on ax"""
        # Plot settings
        width = 0.8
        c1 = "maroon"
        c2 = "saddlebrown"

        start_col = "StartBP" if unit=="BP" else "StartM"
        end_col = "EndBP" if unit=="BP" else "EndM"

        # Convert width in axis coordinate to linewith in figure coordinate (nb of points)
        fig = ax.get_figure()
        length = fig.bbox_inches.width * ax.get_position().width * 72    # 72=nb of points/inch
        lw = width * length / np.diff(ax.get_xlim())[0]

        ### Plot chromosome outline
        ax.plot([pos_x, pos_x], [0, chrom_length], lw = lw, color="lightgray",
                    solid_capstyle = 'round', zorder=0,
                    path_effects=[pe.Stroke(linewidth=lw+3, foreground='k'), pe.Normal()])

        ### Plot the dataframe if only one is given
        if df_RG2 is None:
            ax.vlines(x=np.full(len(df_RG), pos_x), ymin=df_RG[start_col], ymax=df_RG[end_col], lw=lw, color=c1)

        ### Otherwise plot both dataframes next to each other
        else:
            ax.vlines(x=np.full(len(df_RG), pos_x-0.25*width), ymin=df_RG[start_col], ymax=df_RG[end_col], lw=lw*0.45, color=c1)
            ax.vlines(x=np.full(len(df_RG2), pos_x+0.25*width), ymin=df_RG2[start_col], ymax=df_RG2[end_col], lw=lw*0.45, color=c2)

    def plot_all_chromosomes(self, df_roh:pd.DataFrame, df_roh2:pd.DataFrame|None=None,
                            unit:Literal['BP', 'Morgans']='Morgans',
                            legend:tuple|None=None,
                            ax=None):
        """Plot ROH in a genome. If given two dataframes, plot them side by side for comparison."""

        # Plot settings
        c1 = "maroon"
        c2 = "saddlebrown"

        if ax is None:
            fig, ax = plt.subplots(layout="constrained")
        else:
            fig = ax.get_figure()

        ### Set the ax limits (necessary to do this first to get the right line width in the _plot_single_chromosome function)
        ax.set_xlim(0, len(self.chr_lgts) + 1)
        ax.set_ylim(-0.05 * np.max(self.chr_lgts), 1.05 * np.max(self.chr_lgts))

        ### Format the axis
        ax.set_xticks([i for i in range(1, len(self.chr_lgts)+1)])
        ax.tick_params(axis='x', which='both', bottom=False, top=False)
        ax.set_xlabel("Chromosome")
        ax.set_ylabel("Position (bp)" if unit=="BP" else "Position (Morgan)")

        ### Plot the chromosomes
        for ch, ch_len in enumerate(self.chr_lgts, start=1):
            df_ch = df_roh[df_roh['ch'] == ch]
            if df_roh2 is not None:
                df_ch2 = df_roh2[df_roh2['ch'] == ch]
            else:
                df_ch2 = None
            self._plot_single_chromosome(ax, pos_x=ch, chrom_length=ch_len, df_RG=df_ch, df_RG2=df_ch2, unit=unit)

        if df_roh2 is not None and legend is not None:
            legend_elements = [Patch(facecolor=c1, edgecolor=c1, label=legend[0]),
                            Patch(facecolor=c2, edgecolor=c2, label=legend[1])]
            ax.legend(handles=legend_elements, bbox_to_anchor=(0.5, 1.03), loc="upper center", ncols=2)

        return fig, ax
#___________________________________________________
# Detail of results for one individual on one chromosome
#___________________________________________________
    def plot_chromosome_detail(self, df_roh:pd.DataFrame, df_roh_2:pd.DataFrame|None=None, df_het:pd.DataFrame|None=None, unit:Litteral['M', 'BP']='BP',
                                chrom:int|List|None=None, chrom_length:List|int|None=None, max_points:int|None=1e4,
                                figsize:tuple|None=None, fig=None):
        """Plot ROH segments and heterozygosity along one or several chromosomes.
        Args:
            df_roh: dataframe containing ROH segments to plot. Must contain columns 'ch', 'StartBP', 'EndBP', 'StartM', 'EndM'
            df_roh_2: optional second dataframe to plot for comparison
            df_het: optional dataframe containing heterozygosity information along the chromosome. Must contain columns 'ch', 'posBP', 'posM', 'het'
            unit: whether to plot positions in base pairs ('BP') or Morgans ('M')
            chrom: which chromosome(s) to plot. If None, plot all chromosomes present in df_roh.
            chrom_length: length of the chromosome(s) to plot. If None, use the maximum end position in df_roh.
            max_points: maximum number of points to plot for heterozygosity (if df_het is provided). If there are more points than this, a random subset will be plotted."""
        # Plot settings
        kwargs_roh_1 = {'color':'maroon', 'alpha':1, 'linewidth':6}
        kwargs_roh_2 = {'color':'saddlebrown', 'alpha': 1, 'linewidth':6}
        kwargs_het = {'color':'blue', 'alpha' : 0.01, 's' : 3}
        m = MarkerStyle('o', fillstyle='none')

        if chrom is None:
            chrom = df_roh['ch'].unique()
        elif isinstance(chrom, int):
            chrom = [chrom]
        if chrom is None:
            chrom = df_roh['ch'].unique()
        elif isinstance(chrom, int):
            chrom = [chrom]

        fig, axes = plt.subplots(len(chrom), figsize=figsize, layout="constrained")
        fig, axes = plt.subplots(len(chrom), figsize=figsize, layout="constrained")

        for i, chr in enumerate(chrom):
            df_roh_ch = df_roh[df_roh['ch']==chr]
            if df_roh_2 is not None:
                df_roh_2_ch = df_roh_2[df_roh_2['ch']==chr]
            if df_het is not None:
                df_het_ch = df_het[df_het['ch']==chr]

            ax = axes[i] if len(chrom) > 1 else axes
            # Plot SNP heterozygosity if provided
            if df_het is not None and len(df_het) > 0:
                if max_points is not None:
                    rng = np.random.default_rng()
                    subset = rng.choice(df_het_ch.index, size=int(min(len(df_het_ch.index), max_points)), replace=False)
                    df_het_ch = df_het_ch.loc[subset]
                ax.scatter(df_het_ch[f'pos{unit}'], df_het_ch['het'].astype(int), marker=m, **kwargs_het)

            # Plot ROH segments
            cmap = plt.get_cmap('Dark2').colors
            color_list = [cmap[i % len(cmap)] for i in range(len(df_roh_ch))]
            ax.hlines(xmin=df_roh_ch[f'Start{unit}'], xmax=df_roh_ch[f'End{unit}'],
                        y=[1.2]*len(df_roh_ch.index),**kwargs_roh_1)    # , color=color_list, linewidth=6
            if df_roh_2 is not None:
                ax.hlines(xmin=df_roh_2_ch[f'Start{unit}'], xmax=df_roh_2_ch[f'End{unit}'], y=[1.4]*len(df_roh_2_ch.index),**kwargs_roh_2)

            # Adjust x_axis
            if chrom_length is not None:
                if isinstance(chrom_length, list):
                    xmax = chrom_length[chr-1]
                else:
                    xmax = chrom_length
                ax.set_xlim(0, xmax)

            ax.set_title(f"Chromosome {chr}")
            ax.set_yticks([0, 1])
            ax.tick_params(axis='y', which='minor', left=False, right=False)

        ax.set_xlabel(f"Genetic position ({unit})")

        return fig, axes

#___________________________________________________
# Heatmap of log-likelihood
#___________________________________________________
    def plot_ll_heatmap(self, ll, t_grid, admix_grid, true_admix=None, true_time=None, gamma=10, ax=None):
        """Plot log-likelihood heatmap."""
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        norm = colors.PowerNorm(gamma=gamma)
        c = ax.imshow(ll, origin='lower', aspect='auto', extent=(admix_grid[0], admix_grid[-1], t_grid[0], t_grid[-1]), cmap='viridis', norm=norm)
        cbar = fig.colorbar(c, ax=ax, label='Log-Likelihood')
        ticks = np.linspace(0,1, num=7)
        ticks = norm.inverse(ticks)
        ticks_r = np.round(ticks, 0)
        if len(np.unique(ticks_r)) < 7:
            ticks_r = np.round(ticks, 1)
        cbar.set_ticks(ticks_r) # type: ignore

        # MLE point
        time_opt, admix_opt = np.unravel_index(np.argmax(ll, axis=None), ll.shape)
        time_opt = t_grid[time_opt]
        admix_opt = admix_grid[admix_opt]
        ax.scatter(admix_opt, time_opt, marker='+', color='red', s=100, label=f'MLE: time={time_opt:.0f}, admix={admix_opt:.2f}')
        # True optimal point
        if true_time is not None:
            ax.scatter(true_admix, true_time, marker='x', color='firebrick', s=100, label=f'True: time={true_time}, admix={true_admix:.2f}')

        # Confidence interval contour
        threshold_2d = np.max(ll) - 5.991/2     # # chi^2(2, 0.95)/2
        ax.contour(admix_grid, t_grid, ll, levels=[threshold_2d], colors='red', linewidths=2)

        ax.set_xlabel('Admixture Proportion')
        ax.set_ylabel('Time passed (generations)')
        ax.set_title('Log-likelihood colormap')
        ax.legend()
        return fig, ax