import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from minervachem.plotting import parity_plot


def load_data_per_solvent(pkl_fn, solvent='acetone'):
    df = pd.read_pickle(pkl_fn)
    df = df.drop(df[df.mae == 0.00].index).reset_index(drop=True)
    dfg = df.groupby('property')
    d = dfg.get_group(solvent)
    return d

def load_result_data(pkl_fn, task_name=None, verbose=False):
    df = pd.read_pickle(pkl_fn)
    if verbose:
        print(f"There are {len(df[df.mae == 0.00].index)} errored entries in {pkl_fn}")
    df = df.drop(df[df.mae == 0.00].index).reset_index(drop=True)
    if task_name:
        dfg = df.groupby('property')
        d = dfg.get_group(task_name)
        return d
    else:
        return df


def random_seed_check_with_errorbars(filelocation, solv_name, figsize=(10.5, 3.5), dpi=500):
    datadf = load_data_per_solvent(pkl_fn=filelocation, solvent=solv_name)
    dfg = datadf.groupby('max_subgraph_size')
    ssizes = list(dfg.groups.keys())
    dark_colors = ['#00FFFF', '#4831D4', "#0d88e6", '#EA738D', "#b33dc6", "#1984c5"]
    light_colors = ['#FF69B4', '#CCF381', "#7c1158", '#89ABE3', "#ea5545", "#c23728"]
    fig, axes = plt.subplots(1, len(ssizes), figsize=figsize, dpi=dpi, sharey='row')
    for i, size in enumerate(ssizes):
        d = dfg.get_group(size)
        x_axis = [str(element) for element in np.unique(d.n_shots.values[:])]
        a = d.groupby('notes')
        meta = []
        non_meta = []
        for seed in a.groups.keys():
            layer_situation = seed.split('_')[0]
            aa = a.get_group(seed).sort_values(['n_shots'])
            meta.append(list(aa.rmse.values[:]))
            non_meta.append(list(aa.nonmeta_rmse.values[:]))
        meta = np.array(meta)
        non_meta = np.array(non_meta)
        meta_mean = np.mean(meta, axis=0)
        meta_std = np.std(meta, axis=0) / np.sqrt(len(meta))
        non_meta_mean = np.mean(non_meta, axis=0)
        non_meta_std = np.std(non_meta, axis=0) / np.sqrt(len(non_meta))
        if len(ssizes) == 1:
            axes.set_title(f'max_size={size}', fontsize=14)
            axes.errorbar(x_axis, meta_mean, yerr=meta_std, marker='X', ls='-', ms=6, lw=3,
                             color=dark_colors[-1], label='meta', capsize=6, elinewidth=2)
            axes.errorbar(x_axis, non_meta_mean, yerr=non_meta_std, marker='x', ls='--', ms=6, lw=3,
                             color=light_colors[-1], label='non-meta', capsize=6, elinewidth=2)
            axes.set_xlabel('N shots')
            axes.set_ylabel('RMSE')
            axes.legend()
        else:
            axes[i].set_title(f'max_size={size}', fontsize=14)
            axes[i].errorbar(x_axis, meta_mean, yerr=meta_std, marker='X', ls='-', ms=6, lw=3,
                             color=dark_colors[-1], label='meta', capsize=6, elinewidth=2)
            axes[i].errorbar(x_axis, non_meta_mean, yerr=non_meta_std, marker='x', ls='--', ms=6, lw=3,
                             color=light_colors[-1], label='non-meta', capsize=6, elinewidth=2)
            axes[i].set_xlabel('N shots')
            if i == 0:
                axes[i].set_ylabel('RMSE')
                axes[i].legend()

    fig.suptitle(f"{solv_name.capitalize()}", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{filelocation.replace('.pkl', '.png')}")
    plt.clf()
    return


def rs_check_with_errorbars_r2(filelocation, solv_name, figsize=(10.5, 3.5), dpi=500):
    datadf = load_data_per_solvent(pkl_fn=filelocation, solvent=solv_name)
    dfg = datadf.groupby('max_subgraph_size')
    ssizes = list(dfg.groups.keys())
    dark_colors = ['#00FFFF', '#4831D4', "#0d88e6", '#EA738D', "#b33dc6", "#1984c5"]
    light_colors = ['#FF69B4', '#CCF381', "#7c1158", '#89ABE3', "#ea5545", "#c23728"]
    fig, axes = plt.subplots(1, len(ssizes), figsize=figsize, dpi=dpi, sharey='row')
    for i, size in enumerate(ssizes):
        d = dfg.get_group(size)
        x_axis = [str(element) for element in np.unique(d.n_shots.values[:])]
        a = d.groupby('notes')
        meta = []
        non_meta = []
        for seed in a.groups.keys():
            layer_situation = seed.split('_')[0]
            aa = a.get_group(seed).sort_values(['n_shots'])
            meta.append(list(aa.r_2.values[:]))
            non_meta.append(list(aa.nonmeta_r_2.values[:]))
        meta = np.array(meta)
        non_meta = np.array(non_meta)
        meta_mean = np.mean(meta, axis=0)
        meta_std = np.std(meta, axis=0) / np.sqrt(len(meta))
        non_meta_mean = np.mean(non_meta, axis=0)
        non_meta_std = np.std(non_meta, axis=0) / np.sqrt(len(non_meta))
        if len(ssizes) == 1:
            axes.set_title(f'max_size={size}', fontsize=14)
            axes.errorbar(x_axis, meta_mean, yerr=meta_std, marker='X', ls='-', ms=6, lw=3,
                             color=dark_colors[-1], label='meta', capsize=6, elinewidth=2)
            axes.errorbar(x_axis, non_meta_mean, yerr=non_meta_std, marker='x', ls='--', ms=6, lw=3,
                             color=light_colors[-1], label='non-meta', capsize=6, elinewidth=2)
            axes.set_xlabel('N shots')
            axes.axhline(y=0, color='indianred')
            axes.set_ylabel(r'$R^2$')
            axes.legend()
        else:
            axes[i].set_title(f'max_size={size}', fontsize=14)
            axes[i].errorbar(x_axis, meta_mean, yerr=meta_std, marker='X', ls='-', ms=6, lw=3,
                             color=dark_colors[-1], label='meta', capsize=6, elinewidth=2)
            axes[i].errorbar(x_axis, non_meta_mean, yerr=non_meta_std, marker='x', ls='--', ms=6, lw=3,
                             color=light_colors[-1], label='non-meta', capsize=6, elinewidth=2)
            axes[i].set_xlabel('N shots')
            axes[i].axhline(y=0, color='indianred')
            if i == 0:
                axes[i].set_ylabel(r'$R^2$')
                axes[i].legend()

    fig.suptitle(f"{solv_name.capitalize()}", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"r2{filelocation.replace('.pkl', '.png')}")
    plt.clf()
    return


def plot_metrics_with_errorbars(filelocation, task_name, metric='rmse', msize_range=(3,), ns_range=(2,),
                                title=None, figname=None, yticks_range=None, figsize=(4.5, 3.5), dpi=500, fsize=14,
                                single_color=None, title_up=True, axes=None, realxvalues=False):
    """
    This function uses the provided file location and task name to load the results data, extract
    the required metric values for both meta and non-meta approaches, and then plots these values
    with error bars representing the standard error of the mean.
    :param filelocation: str
        Path to the file containing the results data.
    :param task_name: str
        Identifier for the task, used for filtering and labeling the plot.
    :param metric: str, optional, default='rmse'
        Performance metric to plot. Options include 'rmse', 'r_2', or 'mae'.
    :param msize_range: tuple, optional, default=(3,)
        Tuple containing maximal subgraph sizes to iterate over.
    :param ns_range: tuple, optional, default=(2,)
        Tuple containing the number of shots values to iterate over.
    :param title: str, optional
        Title for the plot. If None, task_name is used as the title.
    :param figname: str, optional
        The name of the file where the plot will be saved.
    :param yticks_range: iterable, optional
        Custom tick positions for the y-axis.
    :param figsize: tuple, optional, default=(4.5, 3.5)
        The size of the figure.
    :param dpi: int, optional, default=500
        The resolution of the figure.
    :param title_up: bool, optional, default=True
        If True, capitalizes the title.
    :return: None
        Displays the plot or saves it to a file if figname is provided.
    """
    meta_res = []
    non_meta_res = []
    x_values = [str(el) for el in ns_range] if len(msize_range) == 1 else [str(el) for el in msize_range]
    if realxvalues:
        x_values = ns_range if len(msize_range) == 1 else msize_range

    for n in x_values:
        temp = get_result_for_task_witherrorbar(task_name, filelocation,
                                                ns=int(n) if len(msize_range) == 1 else ns_range[0],
                                                maxssize=msize_range[0] if len(msize_range) == 1 else int(n),
                                                err=metric)
        if temp['meta'][0] is None or temp['nonmeta'][0] is None:
            print(f"Invalid data for ns={n}, max_size={msize_range}, metric={metric}")
        meta_res.append(temp['meta'])
        non_meta_res.append(temp['nonmeta'])

    meta_res = np.array(meta_res)
    non_meta_res = np.array(non_meta_res)

    # dark_colors = ["#18B8B8", '#4831D4', "#0d88e6", '#EA738D', "#b33dc6", "#1984c5"]
    # light_colors = ["#C92A79", '#CCF381', "#7c1158", '#89ABE3', "#ea5545", "#c23728"]

    dark_color = "#1984c5"
    light_color = "#c23728"
    if single_color:
        dark_color = single_color
        light_color = single_color

    if title is None:
        title = task_name

    if axes is None:
        fig, axes = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    if title_up:
        axes.set_title(f'{title.capitalize()}', fontsize=fsize)
    else:
        axes.set_title(f'{title}', fontsize=fsize)
    axes.errorbar(x_values, meta_res[:, 0], yerr=meta_res[:, 1], marker='X', ls='-', ms=8, lw=5,
                  color=dark_color, label='meta', capsize=6, elinewidth=2)
    axes.errorbar(x_values, non_meta_res[:, 0], yerr=non_meta_res[:, 1], marker='x', ls=':', # ls='--', 
                  ms=8, lw=5, color=light_color, label='non-meta', capsize=6, elinewidth=2)
    axes.set_xlabel('N shots', fontsize=fsize, weight='bold', labelpad=5) if len(msize_range) == 1 \
        else axes.set_xlabel('max size',
                             fontsize=fsize,
                             weight='bold',
                             labelpad=5)
    if metric == 'r_2':
        axes.set_ylabel(r'$\mathbf{R^2}$', fontsize=fsize-2, weight='bold', rotation='horizontal', labelpad=10)
    else:
        axes.set_ylabel(f'{metric.upper()}', fontsize=fsize-2, weight='bold', labelpad=10)
    axes.legend(fontsize=fsize-2)
    axes.grid(False)
    for spine in axes.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.0)
    if yticks_range is not None:
        axes.set_yticks(yticks_range)
    axes.xaxis.set_tick_params(labelsize=fsize-2)
    axes.yaxis.set_tick_params(labelsize=fsize-2)
    axes.tick_params(axis='both', direction='out', length=4, bottom=True, left=True)
    plt.tight_layout()
    if figname is not None:
        plt.savefig(figname)
        plt.close()
    # else:
    #     plt.show()

    return axes


def plot_mega_grid(filelocation, task_name, metrics=('rmse', 'r_2', 'mae'),
                   msize_range=(3, 5, 7), ns_range=(2, 5, 10), figsize=(15, 12), dpi=500):
    """
    Creates a 3x3 grid of subplots where:
    - Each row corresponds to a different error metric.
    - Each column corresponds to a different maximal substructure size.

    :param filelocation: str
        Path to the file containing the results data.
    :param task_name: str
        Identifier for the task.
    :param metrics: tuple, optional, default=('rmse', 'r_2', 'mae')
        Tuple containing the three error metrics to plot.
    :param msize_range: tuple, optional, default=(3, 5, 7)
        Tuple containing three maximal subgraph sizes.
    :param ns_range: tuple, optional, default=(2, 5, 10)
        Tuple containing the number of shots values.
    :param figsize: tuple, optional, default=(15, 12)
        Figure size.
    :param dpi: int, optional, default=500
        The resolution of the figure.
    :return: None
        Displays the mega plot.
    """

    # fig, axs = plt.subplots(3, 3, figsize=figsize, dpi=dpi, sharex=True)
    fig, axs = plt.subplots(len(metrics), len(msize_range), figsize=figsize, dpi=dpi, sharex=True)

    # Compute y-axis limits for each metric row
    y_limits = {}
    for metric in metrics:
        all_values = []
        for max_size in msize_range:
            meta_res = []
            non_meta_res = []
            for ns in ns_range:
                temp = get_result_for_task_witherrorbar(task_name, filelocation, ns=ns, maxssize=max_size, err=metric)
                meta_res.append(temp['meta'][0])  # Mean values
                non_meta_res.append(temp['nonmeta'][0])  # Mean values
            all_values.extend(meta_res + non_meta_res)
        y_limits[metric] = (min(all_values), max(all_values))  # Store limits per metric

    # Generate subplots
    for row, metric in enumerate(metrics):
        for col, max_size in enumerate(msize_range):
            print(f"Trying metric={metric}, max_size={max_size}...")
            ax = axs[row, col]
            plot_metrics_with_errorbars(filelocation, task_name, metric=metric,
                                        msize_range=(max_size,), ns_range=ns_range,
                                        title=None, figname=None, yticks_range=None,
                                        figsize=(5, 4), dpi=dpi, title_up=False, axes=ax)

            ax.set_ylim(y_limits[metric])  # Set row-specific y-axis limits
            # Super labels
            if row == 0:
                ax.set_title(f'Max Subgraph Size = {max_size}', fontsize=18, weight='bold')
            else:
                ax.set_title('')
            if col == 0:
                if metric == 'r_2':
                    ax.set_ylabel(r'$\mathbf{R^2}$', fontsize=18, weight='bold', rotation='horizontal', labelpad=10)
                else:
                    ax.set_ylabel(f'{metric.upper()}', fontsize=18, weight='bold', labelpad=10)
            else:
                ax.set_ylabel('')
            if row == len(metrics) - 1:
                ax.set_xlabel('N shots', fontsize=18, weight='bold', labelpad=5)
            else:
                ax.set_xlabel('')
            ax.xaxis.set_tick_params(labelsize=16)
            ax.yaxis.set_tick_params(labelsize=16)
            ax.legend(fontsize=18)

    plt.tight_layout()
    # plt.show()
    plt.savefig(filelocation.replace('.pkl', '_megagridtest.png'))
    plt.clf()


def random_seed_check_r2_errorbars(pkl_fn,
                                   figsize=(10.5, 3.5), dpi=500):

    df = pd.read_pickle(pkl_fn)
    df = df.drop(df[df.mae==0.00].index).reset_index(drop=True)
    dfg = df.groupby('max_subgraph_size')
    rss = list(dfg.groups.keys())
    dark_colors = ['#00FFFF', '#4831D4', "#0d88e6", '#EA738D', "#b33dc6", "#1984c5"]
    light_colors = ['#FF69B4', '#CCF381', "#7c1158", '#89ABE3', "#ea5545", "#c23728"]
    fig, axes = plt.subplots(1, len(rss), figsize=figsize, dpi=dpi, sharey='row')
    for i in range(len(rss)):
        d = dfg.get_group(rss[i])
        x_axis = [str(element) for element in np.unique(d.n_shots.values[:])]
        a = d.groupby('notes')
        meta = []
        non_meta = []
        for seed in a.groups.keys():
            aa = a.get_group(seed).sort_values(['n_shots'])
            meta.append(list(aa.r_2.values[:]))
            non_meta.append(list(aa.nonmeta_r_2.values[:]))
        meta = np.array(meta)
        non_meta = np.array(non_meta)
        meta_mean = np.mean(meta, axis=0)
        meta_std = np.std(meta, axis=0) / np.sqrt(len(meta))
        non_meta_mean = np.mean(non_meta, axis=0)
        non_meta_std = np.std(non_meta, axis=0) / np.sqrt(len(non_meta))
        axes[i].set_title(f'max_size={rss[i]}', fontsize=14)
        axes[i].errorbar(x_axis, meta_mean, yerr=meta_std, marker='X', ls='-', ms=6, lw=3,
                         color=dark_colors[-1], label='meta', capsize=6, elinewidth=2)
        axes[i].errorbar(x_axis, non_meta_mean, yerr=non_meta_std, marker='x', ls='--', ms=6, lw=3,
                         color=light_colors[-1], label='non-meta', capsize=6, elinewidth=2)
        axes[i].set_xlabel('N shots')
        axes[i].axhline(y=0, color='indianred')
        if i == 0:
            axes[i].set_ylabel(r'$R^2$')
            axes[i].legend()

    fig.suptitle(f"{pkl_fn.split('_')[-4].capitalize()}", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"/Users/ypimonova/Documents/mdspace/bsdb_over20rs_bias_sem/{pkl_fn.split('_')[-4]}"
                f"/r2_errorbars_{pkl_fn.split('_')[-4]}_"
                f"{pkl_fn.split('_')[-1].replace('.pkl', '')}.png")
    plt.clf()
    return


def alpha_perp_par_plot(pkl_fn, figname='untitled.png', figsize=(17.5, 7.5), dpi=500, suffix=''):

    df = pd.read_pickle(pkl_fn)
    dfg = df.groupby('property')
    properties = list(dfg.groups.keys())
    dark_colors = ['firebrick', 'darkorange', 'darkgreen', 'steelblue', 'navy', 'indigo']
    # light_colors = ['lightcoral', 'burlywood', 'limegreen', 'skyblue', 'royalblue', 'mediumpurple']
    for i in range(len(properties)):
        d = dfg.get_group(properties[i])
        x_axis = [str(element) for element in np.unique(d.n_shots.values[:])]
        depth_range = np.unique(d.max_subgraph_size.values[:])
        fig, axes = plt.subplots(1, len(depth_range), figsize=figsize, dpi=dpi, sharey='row', sharex='col')
        for dep in range(len(depth_range)):
            d_fixed = d[d.max_subgraph_size == depth_range[dep]].reset_index(drop=True)
            axes[dep].set_title(f'max_size={depth_range[dep]}', fontsize=14)
            alpha_ratios = (d_fixed['alpha_perp'].to_numpy())/(d_fixed['alpha_par'].to_numpy())
            axes[dep].plot(x_axis, alpha_ratios, 'o', ls='--', ms=4, color=dark_colors[i])
            axes[dep].set_xlabel('N shots')
            axes[dep].set_yscale('log')
            if dep == 0:
                axes[dep].set_ylabel(r'$\frac{\alpha_{\perp}}{\alpha_{\parallel}}$',
                                     rotation='horizontal', fontsize=20)
        fig.suptitle(f'{properties[i].capitalize()}', fontsize=16)
        plt.tight_layout()
        plt.savefig(figname, dpi=300)
        plt.clf()
    return


def get_result_for_task_witherrorbar(task_name, filelocation, ns=20, maxssize=5, err='rmse', verbose=False):
    """
    This function loads result data from the specified file and filters the dataset based on
    the provided number of shots (ns) and/or maximum subgraph size (maxssize). It then extracts the
    performance metrics for both meta and non-meta approaches, calculates the mean and standard error,
    and returns these values in a dictionary.
    :param task_name: str
        Identifier for the task.
    :param filelocation: str
        Path to the file containing the result data.
    :param ns: int, optional, default: 20
        Number of shots to filter the results.
    :param maxssize: int, optional, default: 5
        Maximum subgraph size to filter the results.
    :param err: str, optional, default: rmse
        Performance metric to compute. Options are 'rmse', 'r_2', or 'mae'.
    :return: dict
        A dictionary with keys 'meta' and 'nonmeta'. Each key maps to a list containing:
          [mean, standard_error] computed over the selected data.
    """
    res_data = load_result_data(pkl_fn=filelocation, task_name=task_name, verbose=verbose)
    d_temp = res_data[(res_data.max_subgraph_size == maxssize) & (res_data.n_shots == ns)].reset_index(drop=True)
    allowed_metrics = ['rmse', 'r_2', 'mae']
    if err in allowed_metrics:
        meta_colname = err
        nonmeta_colname = 'nonmeta_' + err
    else:
        raise ValueError(f"Unknown error metric: {err}. Allowed metrics are {allowed_metrics}.")

    meta = d_temp[meta_colname].to_numpy()
    non_meta = d_temp[nonmeta_colname].to_numpy()
    meta_mean = np.mean(meta, axis=0)
    meta_std = np.std(meta, axis=0) / np.sqrt(len(meta))
    non_meta_mean = np.mean(non_meta, axis=0)
    non_meta_std = np.std(non_meta, axis=0) / np.sqrt(len(non_meta))
    results = {'meta': [meta_mean, meta_std],
               'nonmeta': [non_meta_mean, non_meta_std]}
    return results


def get_relative_improvement_witherrorbar(
    task_name, filelocation, ns=20, maxssize=5, err='rmse', verbose=False
):
    """
    Loads result data, filters by number of shots and max subgraph size, and computes
    the mean and standard error of the relative improvement between meta and non-meta approaches.

    :param task_name: str
    :param filelocation: str
    :param ns: int
    :param maxssize: int
    :param err: str, one of 'rmse', 'r_2', 'mae'
    :return: tuple (improvement_mean, improvement_std_error)
    """
    res_data = load_result_data(pkl_fn=filelocation, task_name=task_name, verbose=verbose)
    d_temp = res_data[(res_data.max_subgraph_size == maxssize) & (res_data.n_shots == ns)].reset_index(drop=True)
    allowed_metrics = ['rmse', 'r_2', 'mae']
    if err not in allowed_metrics:
        raise ValueError(f"Unknown error metric: {err}. Allowed metrics are {allowed_metrics}.")

    meta = d_temp[err].to_numpy()
    nonmeta = d_temp['nonmeta_' + err].to_numpy()

    if len(meta) != len(nonmeta):
        raise ValueError(f"Meta and non-meta arrays have different lengths: {len(meta)} vs {len(nonmeta)}")

    if len(meta) == 0:
        raise ValueError("No data found for the given filters.")

    if err == 'r_2':
        # Higher is better
        # improvement = (meta - nonmeta) / meta * 100
        improvement = (meta - nonmeta) / (1 - nonmeta) #R2 star (range from 0 to 1)
    else:
        # Lower is better
        improvement = (nonmeta - meta) / meta * 100

    improvement_mean = np.mean(improvement)
    improvement_std = np.std(improvement, ddof=1) / np.sqrt(len(improvement))

    return improvement_mean, improvement_std

def plot_relative_improvement(filelocation, task_name, metrics=('rmse', 'r_2', 'mae'),
                              msize_range=(3, 5, 7), ns_range=(2, 5, 10), figsize=(5, 15), dpi=500,
                              title=None, figname=None, title_up=True):
    """
    Plots relative improvement for each metric as a line plot with markers and error bars.
    All three substructure sizes are on the same plot with different colors.
    The grid is 3 by 1 (one plot per metric).
    """

    fig, axs = plt.subplots(len(metrics), 1, figsize=figsize, dpi=dpi, sharex=True)
    if len(metrics) == 1:
        axs = [axs]  # Make it iterable if only one metric

    colors = ['#FFDB58', '#fb7d07', '#004577']
    if title is None:
        title = task_name
    x_values = [str(el) for el in ns_range]

    for row, metric in enumerate(metrics):
        ax = axs[row]
        for col, max_size in enumerate(msize_range):
            rel_improvements = []
            rel_errors = []
            for ns in ns_range:
                ri, ri_std = get_relative_improvement_witherrorbar(task_name, filelocation, ns=ns, maxssize=max_size,
                                                                   err=metric)
                rel_improvements.append(ri)
                rel_errors.append(ri_std)
            ax.errorbar(x_values, rel_improvements, yerr=rel_errors, label=f'Max Subgraph Size = {max_size}',
                        marker='X', linestyle='-', ms=6, lw=3, color=colors[col], capsize=6, elinewidth=2)
            # ax.plot(ns_range, rel_improvements, label=f'Max Subgraph Size = {max_size}',
            #         marker='o', linestyle='-', ms=6, lw=3, color=colors[col])

        if metric == 'r_2':
            ax.set_ylabel(r'$\mathbf{R^2}$ Relative Improvement (%)', fontsize=14, weight='bold', labelpad=10)
        else:
            ax.set_ylabel(f'{metric.upper()} Relative Improvement (%)', fontsize=14, weight='bold', labelpad=10)

    for ax in axs:
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.0)
        ax.tick_params(axis='both', direction='out', length=4, bottom=True, left=True)
        ax.xaxis.set_tick_params(labelsize=14)
        ax.yaxis.set_tick_params(labelsize=14)

    axs[0].legend(fontsize=14)
    axs[-1].set_xlabel('N shots', fontsize=14, weight='bold', labelpad=5)

    if title_up:
        axs[0].set_title(f'{title.capitalize()}', fontsize=16, weight='bold')
    else:
        axs[0].set_title(f'{title}', fontsize=16, weight='bold')

    plt.tight_layout()
    if figname is None:
        figname = filelocation.replace('.pkl', '_relative_improvement.png')
    plt.savefig(figname)
    plt.clf()


def get_result_for_task_witherrorbar_multilayer(solv_name, filelocation, ns=50, maxssize=5, err='rmse',
                                                eps_par=1.0, generation=''):
    solvdata = load_data_per_solvent(pkl_fn=filelocation, solvent=solv_name)
    solvdata['generation'] = np.array([solvdata.notes.values[i].split('_')[0] for i in range(len(solvdata))])
    d = solvdata[(solvdata.max_subgraph_size == maxssize) & (solvdata.n_shots == ns)
                 & (solvdata.epsilon_par == eps_par) & (solvdata.generation == generation)].reset_index(drop=True)
    if (err == 'rmse') or (err == 'r_2'):
        meta_colname = err
        nonmeta_colname = 'nonmeta_' + err
    else:
        print(f'Unknown value to look for: {err}. Aborting.')
        return

    meta = d[meta_colname].to_numpy()
    non_meta = d[nonmeta_colname].to_numpy()
    meta_mean = np.mean(meta, axis=0)
    meta_std = np.std(meta, axis=0) / np.sqrt(len(meta))
    non_meta_mean = np.mean(non_meta, axis=0)
    non_meta_std = np.std(non_meta, axis=0) / np.sqrt(len(non_meta))
    results = {'meta': [meta_mean, meta_std],
               'nonmeta': [non_meta_mean, non_meta_std]}
    return results


def construct_filename(taskname, value_type='logmolar', maxsize=5, nlayers=None):
    if nlayers==0:
        fn = f'/llns100_13solvents_maxsize{maxsize}/' \
             f'bigsoldb_{value_type}_20rs_{taskname}{maxsize}_{taskname}.pkl'
    elif nlayers==2:
        fn = f'/llns100_13solvents_maxsize{maxsize}/' \
             f'bigsoldb_{value_type}_20rs_{taskname}{maxsize}_twolayertest_{taskname}.pkl'
    elif nlayers==5:
        fn = f'/llns100_13solvents_maxsize{maxsize}/' \
             f'bigsoldb_{value_type}_20rs_{taskname}{maxsize}_fivelayertest_{taskname}.pkl'
    else:
        fn = f'/llns100_13solvents_maxsize{maxsize}/' \
             f'bigsoldb_{value_type}_20rs_{taskname}{maxsize}_layertest_{taskname}.pkl'

    return fn


def meta_nonmeta_parity(tasks_list=(), nshots=50, maxsize=5, nlayers=0, err='rmse', figsize=(6, 6)):
    plot_data = {}
    for task in tasks_list:
        task_fn = construct_filename(task, value_type='logmolar', maxsize=maxsize, nlayers=nlayers)
        plot_data[task] = get_result_for_task_witherrorbar(task, task_fn, ns=nshots, maxssize=maxsize, err=err)

    names = list(plot_data.keys())
    x = [plot_data[nam]['nonmeta'][0] for nam in names]
    y = [plot_data[nam]['meta'][0] for nam in names]
    x_err = [plot_data[nam]['nonmeta'][1] for nam in names]  # Error for x values
    y_err = [plot_data[nam]['meta'][1] for nam in names]  # Error for y values

    f, ax = plt.subplots(figsize=figsize)
    ax.errorbar(x, y, xerr=x_err, yerr=y_err, fmt='o', ms=4, capsize=5, color='darkorange')
    for i, name in enumerate(names):
        ax.annotate(name, (x[i], y[i]), textcoords="offset points", xytext=(5, 5), ha='center', fontsize=8)
    ax.plot([0, 1], [0, 1], 'b--', transform=ax.transAxes)
    if err=='rmse':
        ax.set_xlabel('RMSE, non-meta')
        ax.set_ylabel('RMSE, meta')
        ax.set_ylim(bottom=0.8, top=1.7)
        ax.set_xlim(left=0.8, right=1.7)
    elif err=='r_2':
        ax.set_xlabel(r'$R^2$, non-meta')
        ax.set_ylabel(r'$R^2$, meta')
        ax.axhline(y=0, color='indianred', alpha=0.75)
        ax.axvline(x=0, color='indianred', alpha=0.75)
        ax.set_ylim(bottom=-1.0, top=0.5)
        ax.set_xlim(left=-1.0, right=0.5)
    ax.set_title(f'{nlayers} gen, {nshots} shots, max subgraph size={maxsize}', fontsize=12)
    figure_name = f'/llns100_13solvents_maxsize{maxsize}/' \
                  f'{err}_meta_nonmeta_parity_{nlayers}gen_{nshots}shots.png'
    plt.tight_layout()
    plt.savefig(figure_name)


def meta_nonmeta_parity_allgenoptions(tasks_list=(), nshots=50, maxsize=5, err='rmse', figsize=(6, 6)):
    data0 = {}
    data2 = {}
    data5 = {}
    for task in tasks_list:
        task_fn0 = construct_filename(task, value_type='logmolar', maxsize=maxsize, nlayers=0)
        task_fn2 = construct_filename(task, value_type='logmolar', maxsize=maxsize, nlayers=2)
        task_fn5 = construct_filename(task, value_type='logmolar', maxsize=maxsize, nlayers=5)
        data0[task] = get_result_for_task_witherrorbar(task, task_fn0, ns=nshots, maxssize=maxsize, err=err)
        data2[task] = get_result_for_task_witherrorbar(task, task_fn2, ns=nshots, maxssize=maxsize, err=err)
        data5[task] = get_result_for_task_witherrorbar(task, task_fn5, ns=nshots, maxssize=maxsize, err=err)

    names = list(data0.keys())
    x0 = [data0[nam]['nonmeta'][0] for nam in names]
    y0 = [data0[nam]['meta'][0] for nam in names]
    x0_err = [data0[nam]['nonmeta'][1] for nam in names]  # Error for x values
    y0_err = [data0[nam]['meta'][1] for nam in names]  # Error for y values
    x2 = [data2[nam]['nonmeta'][0] for nam in names]
    y2 = [data2[nam]['meta'][0] for nam in names]
    x2_err = [data2[nam]['nonmeta'][1] for nam in names]  # Error for x values
    y2_err = [data2[nam]['meta'][1] for nam in names]  # Error for y values
    x5 = [data5[nam]['nonmeta'][0] for nam in names]
    y5 = [data5[nam]['meta'][0] for nam in names]
    x5_err = [data5[nam]['nonmeta'][1] for nam in names]  # Error for x values
    y5_err = [data5[nam]['meta'][1] for nam in names]  # Error for y values

    f, ax = plt.subplots(figsize=figsize, dpi=300)
    ax.errorbar(x0, y0, xerr=x0_err, yerr=y0_err, fmt='o', ms=4, capsize=5, color='darkorange', label='0gen')
    ax.errorbar(x2, y2, xerr=x2_err, yerr=y2_err, fmt='o', ms=4, capsize=5, color='seagreen', label='2gen')
    ax.errorbar(x5, y5, xerr=x5_err, yerr=y5_err, fmt='o', ms=4, capsize=5, color='firebrick', label='5gen')
    for i, name in enumerate(names):
        ax.annotate(name, (x0[i], y0[i]), textcoords="offset points", xytext=(5, 5), ha='center', fontsize=7)
    ax.plot([0, 1], [0, 1], 'b--', transform=ax.transAxes)
    if err=='rmse':
        ax.set_xlabel('RMSE, non-meta')
        ax.set_ylabel('RMSE, meta')
        ax.set_ylim(bottom=0.8, top=1.7)
        ax.set_xlim(left=0.8, right=1.7)
    elif err=='r_2':
        ax.set_xlabel(r'$R^2$, non-meta')
        ax.set_ylabel(r'$R^2$, meta')
        ax.axhline(y=0, color='indianred', alpha=0.75)
        ax.axvline(x=0, color='indianred', alpha=0.75)
        ax.set_ylim(bottom=-1.0, top=0.5)
        ax.set_xlim(left=-1.0, right=0.5)
    ax.set_title(f'{nshots} shots, max subgraph size={maxsize}', fontsize=12)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, labels, loc='best')
    figure_name = f'/llns100_13solvents_maxsize{maxsize}/' \
                  f'{err}_meta_nonmeta_allgens_{nshots}shots.png'
    plt.tight_layout()
    plt.savefig(figure_name)


def meta_nonmeta_parity_allgens(tasks_list=(), nshots=50, maxsize=5, eps=0.2, err='rmse', figsize=(6, 6)):
    # This fucntion is updated for using arbitrary number of layers and epsilon

    for j, task in enumerate(tasks_list):
        task_fn = construct_filename(task, value_type='logmolar', maxsize=maxsize, nlayers=None)
        df = pd.read_pickle(task_fn)
        a = np.array([df.notes.values[i].split('_')[0] for i in range(len(df))])
        gens = sorted(np.unique(a), key=lambda x: int(x[1:]))
        if j==0:
            results = [{} for _ in range(len(gens))]
        for jg, gen in enumerate(gens):
            results[jg][task] = get_result_for_task_witherrorbar_multilayer(task, task_fn, ns=nshots, maxssize=maxsize,
                                                                            err=err, eps_par=eps, generation=gen)

    names = list(results[0].keys())
    x = []
    y = []
    x_err = []
    y_err = []
    for i, gen in enumerate(gens):
        x.append([results[i][nam]['nonmeta'][0] for nam in names])
        y.append([results[i][nam]['meta'][0] for nam in names])
        x_err.append([results[i][nam]['nonmeta'][1] for nam in names])
        y_err.append([results[i][nam]['meta'][1] for nam in names])

    f, ax = plt.subplots(figsize=figsize, dpi=300)
    color_list = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6',
                  '#bfef45', '#fabed4', '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000', '#aaffc3',
                  '#808000', '#ffd8b1', '#000075', '#a9a9a9']
    for i, gen in enumerate(gens):
        ax.errorbar(x[i], y[i], xerr=x_err[i], yerr=y_err[i], fmt='o', ms=4, capsize=5, color=color_list[i], label=gen)
        if i==0:
            for i2, name2 in enumerate(names):
                ax.annotate(name2, (x[i][i2], y[i][i2]), textcoords="offset points",
                            xytext=(5, 5), ha='center', fontsize=7)

    ax.plot([0, 1], [0, 1], 'b--', transform=ax.transAxes)
    if err=='rmse':
        ax.set_xlabel('RMSE, non-meta')
        ax.set_ylabel('RMSE, meta')
        ax.set_ylim(bottom=0.8, top=1.7)
        ax.set_xlim(left=0.8, right=1.7)
    elif err=='r_2':
        ax.set_xlabel(r'$R^2$, non-meta')
        ax.set_ylabel(r'$R^2$, meta')
        ax.axhline(y=0, color='indianred', alpha=0.75)
        ax.axvline(x=0, color='indianred', alpha=0.75)
        ax.set_ylim(bottom=-1.0, top=0.5)
        ax.set_xlim(left=-1.0, right=0.5)
    ax.set_title(f'{nshots} shots, max size={maxsize}, eps={eps}', fontsize=12)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, labels, loc='best')
    figure_name = f"/llns100_13solvents_maxsize{maxsize}/" \
                  f"{err}_meta_nonmeta_allgens_{nshots}shots_eps{str(eps).replace('.', '')}.png"
    plt.tight_layout()
    plt.savefig(figure_name)


def meta_nonmeta_bars(tasks_list=(), nshots=50, maxsize=5, err='rmse', figsize=(12, 6)):
    data0 = {}
    data2 = {}
    data5 = {}
    for task in tasks_list:
        task_fn0 = construct_filename(task, value_type='logmolar', maxsize=maxsize, nlayers=0)
        task_fn2 = construct_filename(task, value_type='logmolar', maxsize=maxsize, nlayers=2)
        task_fn5 = construct_filename(task, value_type='logmolar', maxsize=maxsize, nlayers=5)
        data0[task] = get_result_for_task_witherrorbar(task, task_fn0, ns=nshots, maxssize=maxsize, err=err)
        data2[task] = get_result_for_task_witherrorbar(task, task_fn2, ns=nshots, maxssize=maxsize, err=err)
        data5[task] = get_result_for_task_witherrorbar(task, task_fn5, ns=nshots, maxssize=maxsize, err=err)

    names = list(data0.keys())
    x0 = [data0[nam]['nonmeta'][0] for nam in names]
    y0 = [data0[nam]['meta'][0] for nam in names]

    x2 = [data2[nam]['nonmeta'][0] for nam in names]
    y2 = [data2[nam]['meta'][0] for nam in names]

    x5 = [data5[nam]['nonmeta'][0] for nam in names]
    y5 = [data5[nam]['meta'][0] for nam in names]

    b0 = np.array(x0) / np.array(y0)
    b2 = np.array(x2) / np.array(y2)
    b5 = np.array(x5) / np.array(y5)

    x = np.arange(len(names))  # the label locations
    width = 0.25  # the width of the bars

    f, ax = plt.subplots(figsize=figsize, dpi=300, layout='constrained')

    for i in x:
        ax.bar(i, b0[i], width, color='darkorange', label='0gen')
        ax.bar(i + width, b2[i], width, color='seagreen', label='2gen')
        ax.bar(i + width*2, b5[i], width, color='firebrick', label='5gen')
        if i==0:
            handles, labels = plt.gca().get_legend_handles_labels()

    ax.set_ylabel('Length (mm)')
    ax.set_title('Penguin attributes by species')
    ax.set_xticks(x + width, names)
    # ax.legend(loc='upper left', ncols=3)

    if err=='rmse':
        ax.set_ylabel(r'$\frac{\mathrm{RMSE,\ non-meta}}{\mathrm{RMSE,\ meta}}$')
        ax.axhline(y=1e0, color='indianred', alpha=0.75)
        # ax.set_ylim(bottom=0.8, top=1.7)
        # ax.set_xlim(left=0.8, right=1.7)
    elif err=='r_2':
        ax.set_ylabel(r'$\frac{\mathrm{R^2,\ non-meta}}{\mathrm{R^2,\ meta}}$')
        ax.axhline(y=1e0, color='indianred', alpha=0.75)
        ax.axvline(x=0, color='indianred', alpha=0.75)
        # ax.set_ylim(bottom=-1.0, top=0.5)
        # ax.set_xlim(left=-1.0, right=0.5)
    ax.set_title(f'{nshots} shots, max subgraph size={maxsize}', fontsize=12)

    plt.legend(handles, labels, loc='best', ncols=3)
    figure_name = f'/llns100_13solvents_maxsize{maxsize}/' \
                  f'{err}_BARS_meta_nonmeta_allgens_{nshots}shots.png'
    plt.tight_layout()
    plt.savefig(figure_name)


def values_distributions(filename, make_log=False, fig_suffix='', figsize=(6, 6), xleft = 0.0, xright=10.0):
    df = pd.read_csv(filename)
    a = np.array([])
    for col in df.columns:
        if 'solub_' in col:
            a = np.append(a, df[col].to_numpy())
    a = a[~np.isnan(a)]
    if make_log:
        a = np.log10(a)

    plt.figure(figsize=figsize)
    plt.hist(a, bins=50, density=True, alpha=0.7, color='blue')
    # Fit a normal distribution to the data
    mu, std = norm.fit(a)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2, label=fr'Fit results: $\mu$={mu:.2f}, $\sigma$={std:.2f}')
    # Set labels and title
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    # plt.title('Histogram with Fitted Normal Distribution')
    plt.xlim(xleft, xright)
    plt.legend()
    # Show plot
    plt.tight_layout()
    plt.savefig(f'/value_freqs_llns100_for_{fig_suffix}.png')


def two_parity_plots(x_values_2, y_values_2, model, name1='train', name2='test',
                     figsize=(8, 3), dpi=None, title=None, **kwargs):
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    for i, (subset, x, y) in enumerate(zip([name1, name2], x_values_2, y_values_2)):
        pred = model.predict(x)
        parity_plot(y, pred, ax=axes[i], title=f'{subset.title()} (n={y.shape[0]:,d})', **kwargs)
        if title:
            plt.suptitle(title)
    return axes


def plot_layer_metrics(layer_metrics, orientation='horizontal', metrics_to_plot=None):
    """
    Plot performance metrics across layers.

    Args:
        layer_metrics (dict): Dictionary containing lists of metrics per layer
        orientation (str): 'horizontal' or 'vertical' layout of subplots
        metrics_to_plot (list): List of metrics to plot. If None, plots all metrics
    """
    all_metrics = ['MAE', 'RMSE', 'R_2']
    metrics_to_plot = metrics_to_plot or all_metrics

    for metric in metrics_to_plot:
        if metric not in all_metrics:
            raise ValueError(f"Invalid metric: {metric}. Must be one of {all_metrics}")

    n_plots = len(metrics_to_plot)
    n_layers = len(layer_metrics[metrics_to_plot[0]])
    layers = range(n_layers)

    if orientation == 'horizontal':
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    else:  # vertical
        fig, axes = plt.subplots(n_plots, 1, figsize=(8, 4 * n_plots))

    if n_plots == 1:
        axes = [axes]

    # Style settings
    colors = ['deeppink', 'hotpink', 'orchid']  # Default matplotlib colors
    marker_style = 'o'
    line_style = '-'

    # Create plots
    for idx, (ax, metric) in enumerate(zip(axes, metrics_to_plot)):
        values = layer_metrics[metric]

        # Plot line with markers
        line = ax.plot(layers, values,
                       f'{marker_style}{line_style}',
                       color=colors[idx],
                       markersize=8,
                       label=metric)

        # Add value annotations
        for x, y in zip(layers, values):
            ax.annotate(f'{y:.3f}',
                        (x, y),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center')

        # Customize plot
        ax.set_xlabel('Layer')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric}')
        if metric == 'R_2':
            ax.set_ylabel(r'$R^2$')
            ax.set_title(r'$R^2$')
        # ax.grid(True, linestyle='--', alpha=0.7)

        # Set x-axis ticks to be integers
        ax.set_xticks(layers)

        # Add some padding to y-axis for annotations
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.15 * y_range)

    plt.tight_layout()
    return fig, axes
