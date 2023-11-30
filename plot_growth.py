# %%
from networkx.algorithms.bipartite.basic import color
from networkx.readwrite import edgelist
from numpy.random import standard_cauchy
from graph_functions import NetworkGraph, SubGraph, RhoGrid, set_accessibility
# from analysis_functions import *
from analysis_functions import radial_profile, xy_kurtosis, spatial_binning, \
    xy_smoothness, trim_array, find_city_radius, find_city_scaling, \
    remove_nan_inf, network_to_gdf, join_networkpnt_poppoly
import networkx as nx
import numpy as np
import pickle
import geopandas as gpd
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from copy import copy, deepcopy
from numba import jit
from scipy.stats import gaussian_kde
import matplotlib.gridspec as gridspec
from matplotlib import cm
import matplotlib.cm as cmx
import matplotlib.lines as mlines
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker
import matplotlib.colors as colors
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)
import bisect
from SALib.plotting.bar import plot as sal_barplot
from sklearn.metrics import r2_score
from network_growth_main import graph_to_array

plt.ioff()
# %%


def shiftedColorMap(cmap, start, midpoint, stop, name='shifted'):
    '''
    NOW EASIER TO USE matplotlib.colors.TwoSlopeNorm
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range. Defaults to 0.0
          (no lower offset). Should be between 0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 0.5 (no shift).
          Should be between 0.0 and 1.0. In general, this should be
          1 - vmax / (vmax + abs(vmin)). For example if your data range from
          -15.0 to +5.0 and you want the center of the colormap at 0.0,
          `midpoint` should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range. Defaults to 1.0
          (no upper offset). Should be between `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap

def thresholdColorMap(cmap, data, threshold, colour, datamax=None,
                      name='threshold'):
    """
    Creates a colormap with a solid colour below (above) a threshold, and a
    pre-existing colormap used above (below) the threshold.

    :param cmap: str, matplotlib.colors.ListedColormap
        Matplotlib named colormap.
    :param threshold: float
        Threshold (from data) below which colormap is "colour".
    :param data: numpy.ndarray
        Data for which the colormap is created
    :param colour: str, tuple
        Matplotlib named colour or (R, G, B) tuple
    :param datamax: float, int
        Maximum value for when data is being compared to another dataset with a
        larger maximum.
    :param name: str
        Name for new colormap
    :return:
    """
    if type(cmap) == str:
        cmap = cmx.get_cmap(cmap)
    if type(colour) == str:
        colour = colors.to_rgb(colour)

    data = np.sort(data.flatten())

    # if datamin is None:
    #     datamin = data.min()
    if datamax is None:
        datamax = data.max()

    cdict = {'red': [], 'green': [], 'blue': [], 'alpha': []}

    # regular index to map colours
    reg_index = np.linspace(0, 1, len(data), endpoint=True)

    index_len = len(data)
    thresh_idx = bisect.bisect_right(a=np.arange(index_len), x=threshold) - 1

    shift_index = np.hstack([np.zeros(shape=thresh_idx),
        np.linspace(0, 1.0, index_len - thresh_idx, endpoint=True)])

    for ri, si, dt in zip(reg_index, shift_index, data):
        if dt <= threshold:
            r, g, b = colour
            cdict['red'].append((ri, r, r))
            cdict['green'].append((ri, g, g))
            cdict['blue'].append((ri, b, b))
            cdict['alpha'].append((ri, 1, 1))

        else:
            r, g, b, a = cmap(si)
            cdict['red'].append((ri, r, r))
            cdict['green'].append((ri, g, g))
            cdict['blue'].append((ri, b, b))
            cdict['alpha'].append((ri, a, a))

    newcmap = LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap

def get_xlim_ylim(axes):
    xmin, ymin, xmax, ymax = None, None, None, None

    for i, ax in enumerate(axes):
        if i == 0:
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()

        else:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            if xlim[0] < xmin:
                xmin = xlim[0]
            if xlim[1] > xmax:
                xmax = xlim[1]
            if ylim[0] < ymin:
                ymin = ylim[0]
            if ylim[1] > ymax:
                ymax = ylim[1]
    return (xmin, xmax), (ymin, ymax)

def basic_pcolor(data, title=None, cmap=None, norm=None, savepath=None,
                 show=False):
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('w')
    img = ax.pcolor(data, cmap=cmap, norm=norm)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%')
    cbar = fig.colorbar(img, cax=cax)
    if title is not None:
        ax.set_title(title)
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
    if show:
        plt.show()

def plot_pop_network(year, networks_dict, pop_gdf, save_path=None):
    """
    Plot network on top of population density
    """
    ntwk = networks_dict[year]
    ntwk_coords = {k:[v[0] * 1000, v[1] * 1000] for k,v in ntwk.coords.items()}
    fig, ax = plt.subplots()
    pop_gdf.plot(ax=ax, column=f'{year}_pop', cmap='binary')

    nx.draw_networkx_nodes(ntwk.subgraph, pos=ntwk_coords, ax=ax,
                           node_color=ntwk.wbc_cols, node_size=3, cmap='jet')
    ax.set_xlim([496000, 561000])
    ax.set_ylim([155000, 207000])
    ax.set_title(f'{year}')

    sm1 = plt.cm.ScalarMappable(cmap='jet',
                                norm=plt.Normalize(vmin=min(ntwk.wbc_cols),
                                                   vmax=max(ntwk.wbc_cols)))
    sm1._A = []
    plt.colorbar(sm1, label='WBC')
    sm2 = plt.cm.ScalarMappable(cmap='binary',
                                norm=plt.Normalize(
                                    vmin=min(pop_grid[f'{year}_pop'] / 1000),
                                    vmax=max(pop_grid[f'{year}_pop'] / 1000)))
    sm2._A = []
    plt.colorbar(sm2, label='$\\rho_{pop}$ (1000s)')
    if save_path:
        plt.savefig(save_path)
    plt.tight_layout()

def plot_wbc_popdens(year, network_dict, pop_grid, savepath=None):
    """
    Plot the WBC of station on the network against population density of the
    grid cell.
    """
    ntwkgdf = network_to_gdf(network=network_dict[year])

    ntwrkyr_pop = \
        join_networkpnt_poppoly(network_gdf=ntwkgdf, pop_gdf=pop_grid,
                                year=year)

    null_filt = ~ntwrkyr_pop[f'{yr}_pop'].isnull() * \
                ~ntwrkyr_pop[f'WBC'].isnull()
    x = ntwrkyr_pop['WBC'].values[null_filt]
    y = ntwrkyr_pop[f'{yr}_pop'].values[null_filt]
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    fig = plt.figure(constrained_layout=True)

    spec = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)
    ax1 = fig.add_subplot(spec[0, 0:2])
    ax2 = fig.add_subplot(spec[1:, 0:2])
    ax3 = fig.add_subplot(spec[1:, -1])

    ax2.scatter(x, y, c=z)
    ax2.set_xlabel('WBC')
    ax2.set_ylabel('$\\rho_{pop}$ (cap km$^{-2}$)')
    # ax2.set_title(f'{yr}')
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    ax1.hist(x, bins=ax2.get_xticks(minor=True))
    # ax1.set_xlabel('WBC', fontsize='small')

    ax3.hist(y, orientation='horizontal', bins=ax2.get_yticks(minor=True))
    # ax3.set_ylabel('$\\rho_{pop}$ (cap km$^{-1}$)', fontsize='small')

    for ax in [ax1, ax2]:
        ax.set_xscale('log')
    for ax in [ax2, ax3]:
        ax.set_yscale('log')

    ax2.set_ylim([0, 62000])
    ax2.set_xlim([3.6e-05, 0.764])
    ax1.set_xlim(ax2.get_xlim())
    ax3.set_ylim(ax2.get_ylim())
    ax1.set_xticklabels([])
    ax3.set_yticklabels([])

    plt.figtext(x=0.9, y=0.9, s=f'{yr}', )
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)
    plt.show()


def plot_evolution(rho, xv, yv, year, network=None, save_path=None, show=True,
                   observed_rho=None, borders=None):
    """

    :param rho:
    :param xv:
    :param yv:
    :param year:
    :param network:
    :param save_path:
    :param show:
    :param observed_rho:
    :return:
    """
    if borders is not None:
        rho = trim_array(rho, borders)
        xv = trim_array(xv, borders)
        yv = trim_array(yv, borders)
        if observed_rho is not None:
            observed_rho = trim_array(observed_rho, borders)

    fig = plt.figure(constrained_layout=True)
    fig.patch.set_facecolor('w')
    spec = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)
    ax1 = fig.add_subplot(spec[0, 0:2])
    ax2 = fig.add_subplot(spec[1:, 0:2])
    ax3 = fig.add_subplot(spec[1:, -1])

    if observed_rho is not None:
        ax1.plot(xv[0, :], np.nanmean(observed_rho, axis=0), label='Obs.',
                 color='b')
    ax1.plot(xv[0, :], np.nanmean(rho, axis=0), label='Sim.', color='r')
    ax1.set_xlim(xv[0,0], xv[0, -1])
    ax1.set_title(r'Mean $\rho_{pop}$ along x-axis', fontsize='small')
    ax1.legend(fontsize='small')

    ax2.pcolor(xv, yv, rho, cmap='viridis', shading='auto')
    if network is not None:
        ax2.scatter(network['lon'], network['lat'], s=network['WBC'] * 20,
                    fc='none', ec='r')
    ax2.set_xlim(xv[0, 0], xv[0, -1])
    ax2.set_ylim(yv[0, 0], yv[-1, 0])
    ax2.set_title(rf'Modelled {year} $\rho_{{pop}}$ and London rail network',
                  fontsize='small')

    if observed_rho is not None:
        ax3.plot(np.nanmean(observed_rho, axis=1), yv[:, 0], color='b')
    ax3.plot(np.nanmean(rho, axis=1), yv[:, 0], color='r')
    ax3.set_ylim(yv[0,0], yv[-1,0])
    ax3.set_title(r'Mean $\rho_{pop}$ along y-axis', fontsize='small')

    sm2 = plt.cm.ScalarMappable(cmap='viridis',
                                norm=plt.Normalize(vmin=np.min(rho / 1000),
                                                   vmax=np.max(rho / 1000)))
    sm2._A = []
    fig.colorbar(sm2, label='$\\rho_{pop}$ (1000s)', ax=ax2, location='bottom')

    # plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


def plot_obs_sim(rho_sim, rho_obs, xv, yv, year, cmap="viridis",
                 cmap_diff="bwr", perc_diff=False, threshold_rho=None,
                 threshold_perc=None, borders=None, under_col_rho=None,
                 over_col_perc=None, network=None, save_path=None, show=True):
    """
    Plot observed and simulated population densities, as well as the difference
    between both.

    :param rho_sim: numpy.ndarray
        Simulated population density
    :param rho_obs: numpy.ndarray
        Observed population density
    :param xv:
        Meshgrid in x-direction
    :param yv:
        Meshgrid in y-direction
    :param year: int
        Year of data
    :param cmap: matplotlib.colors.ListedColormap
        Colormap for all plots
    :param cmap_diff: str
        Colormap for difference plot
    :param perc_diff: bool
        True: show difference as a percentage, False: difference as magnitude
    :param threshold_rho: numpy.ndarray

    :param threshold_perc: numpy.ndarray
        Threshold of percentage differece colormap
    :param network:
    :param save_path: str
        Save path
    :param show:
    :return:
    """
    if borders is not None:
        rho_sim = trim_array(rho_sim, borders)
        rho_obs = trim_array(rho_obs, borders)
        xv = trim_array(xv, borders)
        yv = trim_array(yv, borders)

    max_rho = max(rho_sim.max(), rho_obs.max())
    # min_rho = min(rho_sim.min(), rho_obs.min())
    simobs_diff = (rho_sim - rho_obs) / 1000

    if perc_diff:
        simobs_diff /= (rho_obs / 1000)
        simobs_diff *= 100  # for percentage, not fraction

    # setting xv and yv to origin
    xv = xv - xv.min()
    yv = yv - yv.min()

    if perc_diff:
        cm_diff_norm = \
            colors.TwoSlopeNorm(vmin=simobs_diff.min(), vmax=threshold_perc,
                                vcenter=0.)
    else:
        if simobs_diff.min() >= 0:
            cm_diff_norm = None
        else:
            cm_diff_norm = \
                colors.TwoSlopeNorm(vmin=simobs_diff.min(),
                                    vmax=simobs_diff.max(), vcenter=0.)

    fig = plt.figure(constrained_layout=True)
    fig.patch.set_facecolor('w')
    # spec = gridspec.GridSpec(ncols=4, nrows=2, figure=fig)
    # ax1 = fig.add_subplot(spec[0, 0:2])
    # ax2 = fig.add_subplot(spec[0, 2:])
    # ax3 = fig.add_subplot(spec[-1, 1:3])

    subfigs = fig.subfigures(2, 1)

    spec1 = gridspec.GridSpec(ncols=4, nrows=1, figure=subfigs[0])
    ax1 = subfigs[0].add_subplot(spec1[0:2])
    ax2 = subfigs[0].add_subplot(spec1[2:4])

    spec2 = gridspec.GridSpec(ncols=12, nrows=1, figure=subfigs[1])
    ax3 = subfigs[1].add_subplot(spec2[3:9])

    caxrho = subfigs[1].add_subplot(spec2[1:2])
    caxdiff = subfigs[1].add_subplot(spec2[10:11])

    # ax4 = subfigs[1].add_subplot(spec2[3:])

    # subfigs[0].subplots_adjust(bottom=0.8)
    if type(cmap) == str:
        cmap = cmx.get_cmap(cmap).copy()
    if under_col_rho is None:
        under_col_rho = 'w'
    cmap.set_under(under_col_rho)

    if type(cmap_diff) == str:
        cmap_diff = cmx.get_cmap(cmap_diff).copy()

    if perc_diff:
        if over_col_perc is None:
            over_col_perc = 'k'
        cmap_diff.set_over(over_col_perc)

    # ax1.pcolor(xv, yv, rho_sim, vmax=max_rho, cmap=cm_sim)
    ax1.pcolor(xv, yv, rho_sim, vmin=threshold_rho, vmax=max_rho, cmap=cmap,
               shading='auto')
    if network is not None:
        ax1.scatter(network['lon'], network['lat'], s=network['WBC'] * 20,
                    fc='none', ec='r')
    ax1.set_xlim(xv[0, 0], xv[0, -1])
    ax1.set_ylim(yv[0, 0], yv[-1, 0])
    ax1.set_title(rf'Modelled {year} $\rho$ (1000s)',
                  fontsize='small')
    ax1.tick_params(axis='both', labelsize=6)
    ax1.set_aspect('equal')

    # ax2.pcolor(xv, yv, rho_obs, vmax=max_rho, cmap=cm_obs)
    ax2.pcolor(xv, yv, rho_obs, vmin=threshold_rho, vmax=max_rho, cmap=cmap,
               shading='auto')
    if network is not None:
        ax2.scatter(network['lon'], network['lat'], s=network['WBC'] * 20,
                    fc='none', ec='r')
    ax2.set_xlim(xv[0, 0], xv[0, -1])
    ax2.set_ylim(yv[0, 0], yv[-1, 0])
    ax2.set_title(rf'Observed {year} $\rho$',
                  fontsize='small')
    ax2.tick_params(axis='both', labelsize=6)
    ax2.set_aspect('equal')

    ax3.pcolor(xv, yv, simobs_diff, cmap=cmap_diff, norm=cm_diff_norm,
               shading='auto', vmax=threshold_perc)
    if network is not None:
        ax3.scatter(network['lon'], network['lat'], s=network['WBC'] * 20,
                    fc='none', ec='r')
    ax3.set_xlim(xv[0, 0], xv[0, -1])
    ax3.set_ylim(yv[0, 0], yv[-1, 0])
    ax3.set_title(rf'{year} $\rho_{{sim}} - \rho_{{obs}}$',
                  fontsize='small')
    ax3.tick_params(axis='both', labelsize=6)
    ax3.set_aspect('equal')

    # cb_rho = plt.cm.ScalarMappable(cmap=cm_sim,
    #                                norm=plt.Normalize(vmin=rho_sim.min()/1000,
    #                                                   vmax=rho_sim.max()/1000))
    cb_rho = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=plt.Normalize(vmin=rho_sim.min()/1000,
                                                      vmax=rho_sim.max()/1000))
    cb_rho._A = []
    # cb_rho_fig = fig.colorbar(cb_rho, ax=(ax1, ax2), location='bottom',
    #                           extend='min', shrink=0.5)
    cb_rho_fig = subfigs[0].colorbar(cb_rho, cax=caxrho,
                                     # ax=(ax1, ax2), location='bottom',
                                     extend='min', shrink=0.6)
    cb_rho_fig.set_ticks(
        np.sort(np.append(cb_rho_fig.get_ticks(), threshold_rho / 1000)))
    cb_rho_fig.ax.yaxis.set_ticklabels(
        [str(int(l)) if int(l) == l else f'{l:.3f}'
         for l in cb_rho_fig.get_ticks()])
    cb_rho_fig.ax.tick_params(labelsize=6)
    cb_rho_fig.ax.set_ylabel(ylabel=r'$\rho$ (1000 cap km$^{-2}$)',
                             fontdict={'size': 6})

    cb_diff = plt.cm.ScalarMappable(cmap=cmap_diff, norm=cm_diff_norm)
    cb_diff._A = []
    # fig.colorbar(cb_diff, label=r'$\rho_{sim} - \rho_{obs}$ (1000s)', ax=ax3,
    #              location='right')
    # cb_diff_fig = fig.colorbar(cb_diff, ax=ax3, location='right')

    if perc_diff:
        cb_diff_fig = subfigs[1].colorbar(cb_diff, cax=caxdiff, shrink=0.6,
                                          extend='max')
    else:
        cb_diff_fig = subfigs[1].colorbar(cb_diff, cax=caxdiff, shrink=0.6)

    cb_diff_fig.ax.tick_params(labelsize=6)

    if perc_diff:
        cb_diff_fig.ax.set_ylabel(
            ylabel=r'$\frac{\rho_{sim} - \rho_{obs}}{\rho_{obs}}$ (%)',
            fontdict={'size': 6})
    else:
        cb_diff_fig.ax.set_ylabel(
            ylabel=r'$\rho_{sim} - \rho_{obs}$ (1000 cap km$^{-2}$)',
            fontdict={'size': 6})
    # plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()

def plot_var(var, var_name, xv, yv, year, network=None, save_path=None,
             save_var_name=None, show=True, observed_rho=None, borders=None):

    if borders is not None:
        var = trim_array(var, borders)
        xv = trim_array(xv, borders)
        yv = trim_array(yv, borders)

    fig = plt.figure(constrained_layout=True)
    fig.patch.set_facecolor('w')
    spec = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)
    ax1 = fig.add_subplot(spec[0, 0:2])
    twinax1 = ax1.twinx()
    ax2 = fig.add_subplot(spec[1:, 0:2])
    ax3 = fig.add_subplot(spec[1:, -1])
    twinax3 = ax3.twiny()

    if observed_rho is not None:
        if borders is not None:
            observed_rho = trim_array(observed_rho, borders)
        ax1.plot(xv[0, :], np.nanmean(observed_rho, axis=0),
                 label=r'Obs. $\rho_{{pop}}$', color='b')
    twinax1.plot(xv[0, :], np.nanmean(var, axis=0), label=f'Sim. {var_name}',
                 color='r')
    ax1.set_xlim(xv[0, 0], xv[0, -1])
    ax1.set_title(rf'Mean {var_name} and $\rho_{{pop}}$ along x-axis',
                  fontsize='small')
    ax1.tick_params(axis='y', colors='b')
    twinax1.tick_params(axis='y', colors='r')
    lines = [mlines.Line2D([], [], color=color, label=label)
             for label, color in
             zip([r'Obs. $\rho_{{pop}}$', f'Sim. {var_name}'], ['b', 'r'])]

    leg = ax1.legend(lines, [r'Obs. $\rho_{{pop}}$', f'Sim. {var_name}'],
                     loc='upper right', frameon=False, prop=dict(size='small'))
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)

    # ax1.legend(fontsize='small')

    ax2.pcolor(xv, yv, var, cmap='binary', shading='auto')
    if network is not None:
        ax2.scatter(network['lon'], network['lat'], s=network['WBC'] * 20,
                    fc='none', ec='r')
    ax2.set_xlim(xv[0, 0], xv[0, -1])
    ax2.set_ylim(yv[0, 0], yv[-1, 0])
    ax2.set_title(rf'Modelled {year} {var_name} and London rail network',
                  fontsize='small')

    if observed_rho is not None:
        ax3.plot(np.nanmean(observed_rho, axis=1), yv[:, 0], color='b')
    twinax3.plot(np.nanmean(var, axis=1), yv[:, 0], color='r')
    ax3.set_ylim(yv[0, 0], yv[-1, 0])
    ax3.set_title(f'Mean {var_name} and $\\rho_{{pop}}$ \n along y-axis',
                  fontsize='small')
    ax3.tick_params(axis='x', colors='b')
    twinax3.tick_params(axis='x', colors='r')

    sm2 = plt.cm.ScalarMappable(cmap='binary',
                                norm=plt.Normalize(vmin=np.nanmin(var),
                                                   vmax=np.nanmax(var)))
    sm2._A = []
    cbar = fig.colorbar(sm2, label=f'{var_name}', ax=ax2, location='bottom')

    # plt.tight_layout()
    if save_path:
        plt.savefig(f'{save_path}/{year}_{save_var_name}'
                    f'_growth.png')
    if show:
        plt.show()
    plt.close()

def plot_obssim_multiple(rho_sim, rho_obs, xv, yv, years,
                         cmap="viridis", cmap_diff="bwr", perc_diff=False,
                         threshold_rho=None, threshold_perc=None, borders=None,
                         under_col_rho=None, over_col_perc=None, network=None,
                         save_path=None, cbar_savepath=None, show=True):
    """
    Plot simulated population densities for multiple years, and the difference
    between observed and simulated.

    :param rho_sim: numpy.ndarray
        Simulated population density
    :param rho_obs: numpy.ndarray
        Observed population density
    :param xv:
        Meshgrid in x-direction
    :param yv:
        Meshgrid in y-direction
    :param years: int
        Years of data to be plotted
    :param cmap: matplotlib.colors.ListedColormap
        Colormap for all plots
    :param cmap_diff: str
        Colormap for difference plot
    :param perc_diff: bool
        True: show difference as a percentage, False: difference as magnitude
    :param threshold_rho: numpy.ndarray

    :param threshold_perc: numpy.ndarray
        Threshold of percentage differece colormap
    :param network:
    :param save_path: str
        Save path
    :param show:
    :return:
    """
    if borders is not None:
        rho_sim = {yr: trim_array(rho_sim[yr], borders) for yr in years}
        rho_obs = {yr: trim_array(rho_obs[yr], borders) for yr in years}
        xv = trim_array(xv, borders)
        yv = trim_array(yv, borders)

    max_rho = np.max([np.nanmax(rho_sim[yr]) for yr in years])
    min_rho = np.min([np.nanmin(rho_sim[yr]) for yr in years])

    simobs_diff = {yr: (rho_sim[yr] - rho_obs[yr]) / 1000
                   for yr in years}

    if perc_diff:
        for yr in years:
            simobs_diff[yr] /= (rho_obs[yr] / 1000) * 100

    # setting xv and yv to origin
    xv = xv - xv.min()
    yv = yv - yv.min()

    if perc_diff:
        simobs_diffmin = np.min([np.nanmin(simobs_diff[yr])
                                 for yr in years])
        cm_diff_norm = \
            colors.TwoSlopeNorm(vmin=simobs_diffmin, vmax=threshold_perc,
                                vcenter=0.)
    else:
        simobs_diffmin = np.min([np.nanmin(simobs_diff[yr])
                                 for yr in years])
        simobs_diffmax = np.max([np.nanmax(simobs_diff[yr])
                                 for yr in years])
        if simobs_diffmin >= 0:
            cm_diff_norm = None
        else:

            cm_diff_norm = \
                colors.TwoSlopeNorm(vmin=simobs_diffmin,
                                    vmax=simobs_diffmax, vcenter=0.)

    # fig, axes = plt.figure(constrained_layout=True)

    # spec = gridspec.GridSpec(ncols=4, nrows=2, figure=fig)
    # ax1 = fig.add_subplot(spec[0, 0:2])
    # ax2 = fig.add_subplot(spec[0, 2:])
    # ax3 = fig.add_subplot(spec[-1, 1:3])

    fig, axes = plt.subplots(nrows=len(years), ncols=2, figsize=(6, 10))
    fig.patch.set_facecolor('w')

    for i, yr in enumerate(years):

        if type(cmap) == str:
            cmap = cmx.get_cmap(cmap).copy()
        if under_col_rho is None:
            under_col_rho = 'w'
        cmap.set_under(under_col_rho)

        if type(cmap_diff) == str:
            cmap_diff = cmx.get_cmap(cmap_diff).copy()

        if perc_diff:
            if over_col_perc is None:
                over_col_perc = 'k'
            cmap_diff.set_over(over_col_perc)

        # ax1.pcolor(xv, yv, rho_sim, vmax=max_rho, cmap=cm_sim)
        axes[i, 0].pcolor(xv, yv, rho_sim[yr], vmin=threshold_rho, vmax=max_rho,
                          cmap=cmap, shading='auto')
        if network is not None:
            axes[i, 0].scatter(network['lon'], network['lat'],
                               s=network['WBC'] * 20,
                               fc='none', ec='r')
        axes[i, 0].set_xlim(xv[0, 0], xv[0, -1])
        axes[i, 0].set_ylim(yv[0, 0], yv[-1, 0])
        # axes[0].set_title(rf'Modelled {year} $\rho$ (1000s)',
        #               fontsize='small')
        axes[i, 0].tick_params(axis='both', labelsize=6)
        axes[i, 0].set_aspect('equal')

        axes[i, 0].set_ylabel(r"$\bf{" + str(yr) + "}$")

        axes[i, 1].pcolor(xv, yv, simobs_diff[yr], cmap=cmap_diff,
                          norm=cm_diff_norm, shading='auto',
                          vmax=threshold_perc)
        # if network is not None:
        #     axes[1].scatter(network['lon'], network['lat'], s=network['WBC'] * 20,
        #                 fc='none', ec='r')
        axes[i, 1].set_xlim(xv[0, 0], xv[0, -1])
        axes[i, 1].set_ylim(yv[0, 0], yv[-1, 0])
        # axes[1].set_title(rf'{year} $\rho_{{sim}} - \rho_{{obs}}$',
        #               fontsize='small')
        axes[i, 1].tick_params(axis='both', labelsize=6)
        axes[i, 1].set_aspect('equal')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.show()

    if threshold_rho is not None:
        min_rho = threshold_rho
    cbfig = plt.figure(figsize=(fig.get_size_inches()[0], 1.5))
    cbfig.patch.set_facecolor('w')
    subfigs = cbfig.subfigures(1, 2)
    ax1 = subfigs[0].add_axes([0.05, 0.80, 0.9, 0.1])
    cb1 = mpl.colorbar.ColorbarBase(
        ax1, orientation='horizontal', cmap=cmap,
        norm=mpl.colors.Normalize(vmin=min_rho / 1000, vmax=max_rho / 1000),
        extend='neither', label=r'$\rho$ (1000 cap km$^{-2}$)')
    cb1.ax.tick_params(labelsize='x-small')
    cb1.ax.set_xlabel(r'$\rho$ (1000 cap km$^{-2}$)',
                      fontsize='x-small')
    # cb1.ax.set_xticks(np.sort(np.append(cb1.ax.get_xticks(), max_rho)))
    # cb1.ax.xaxis.set_ticklabels([str(int(l)) if int(l) == l else f'{l:.3f}'
    #      for l in cb1.get_ticks()])

    ax2 = subfigs[1].add_axes([0.05, 0.80, 0.9, 0.1])
    cb2 = mpl.colorbar.ColorbarBase(ax2, orientation='horizontal',
                                    cmap=cmap_diff, norm=cm_diff_norm)
    cb2.ax.tick_params(labelsize='x-small')
    cb2.ax.set_xlabel(r'$\rho_{{sim}} - \rho_{{obs}}$ (1000 cap km$^{-2}$)',
                      fontsize='x-small')

    if cbar_savepath is not None:
        plt.savefig(cbar_savepath, dpi=300)
    plt.show()

    # fig.colorbar(cb_diff, label=r'$\rho_{sim} - \rho_{obs}$ (1000s)', ax=ax3,
    #              location='right')
    # cb_diff_fig = fig.colorbar(cb_diff, ax=ax3, location='right')

def plot_obssim3_multiple(rho_sim, rho_obs, xv, yv, years,
                          cmap="viridis", cmap_diff="bwr", perc_diff=False,
                          threshold_rho=None, threshold_perc=None, borders=None,
                          under_col_rho=None, over_col_perc=None, network=None,
                          save_path=None, cbar_savepath=None, show=True):
    """
    Plot simulated population densities for multiple years, and the difference
    between observed and simulated.

    :param rho_sim: numpy.ndarray
        Simulated population density
    :param rho_obs: numpy.ndarray
        Observed population density
    :param xv:
        Meshgrid in x-direction
    :param yv:
        Meshgrid in y-direction
    :param years: int
        Years of data to be plotted
    :param cmap: matplotlib.colors.ListedColormap
        Colormap for all plots
    :param cmap_diff: str
        Colormap for difference plot
    :param perc_diff: bool
        True: show difference as a percentage, False: difference as magnitude
    :param threshold_rho: numpy.ndarray

    :param threshold_perc: numpy.ndarray
        Threshold of percentage differece colormap
    :param network:
    :param save_path: str
        Save path
    :param show:
    :return:
    """
    if borders is not None:
        rho_sim = {yr: trim_array(rho_sim[yr], borders) for yr in years}
        rho_obs = {yr: trim_array(rho_obs[yr], borders) for yr in years}
        xv = trim_array(xv, borders)
        yv = trim_array(yv, borders)

    obs_max_rho = np.max([np.nanmax(rho_obs[yr]) for yr in years])
    obs_min_rho = np.min([np.nanmin(rho_obs[yr]) for yr in years])
    sim_max_rho = np.max([np.nanmax(rho_sim[yr]) for yr in years])
    sim_min_rho = np.min([np.nanmin(rho_sim[yr]) for yr in years])

    max_rho = np.max([obs_max_rho, sim_max_rho])
    min_rho = np.min([obs_min_rho, sim_min_rho])

    simobs_diff = {yr: (rho_sim[yr] - rho_obs[yr]) / 1000
                   for yr in years}

    if perc_diff:
        for yr in years:
            simobs_diff[yr] /= (rho_obs[yr] / 1000) * 100

    # setting xv and yv to origin
    xv = xv - xv.min()
    yv = yv - yv.min()

    if perc_diff:
        simobs_diffmin = np.min([np.nanmin(simobs_diff[yr])
                                 for yr in years])
        cm_diff_norm = \
            colors.TwoSlopeNorm(vmin=simobs_diffmin, vmax=threshold_perc,
                                vcenter=0.)
    else:
        simobs_diffmin = np.min([np.nanmin(simobs_diff[yr])
                                 for yr in years])
        simobs_diffmax = np.max([np.nanmax(simobs_diff[yr])
                                 for yr in years])
        if simobs_diffmin >= 0:
            cm_diff_norm = None
        else:

            cm_diff_norm = \
                colors.TwoSlopeNorm(vmin=simobs_diffmin,
                                    vmax=simobs_diffmax, vcenter=0.)

    fig, axes = plt.subplots(nrows=len(years), ncols=3, figsize=(9, 9))
    # fig, axes = plt.subplots(nrows=len(years), ncols=3)
    fig.patch.set_facecolor('w')

    for i, yr in enumerate(years):

        if type(cmap) == str:
            cmap = cmx.get_cmap(cmap).copy()
        if under_col_rho is None:
            under_col_rho = 'w'
        cmap.set_under(under_col_rho)

        if type(cmap_diff) == str:
            cmap_diff = cmx.get_cmap(cmap_diff).copy()

        if perc_diff:
            if over_col_perc is None:
                over_col_perc = 'k'
            cmap_diff.set_over(over_col_perc)

        # ax1.pcolor(xv, yv, rho_sim, vmax=max_rho, cmap=cm_sim)
        axes[i, 0].pcolor(xv, yv, rho_sim[yr], vmin=threshold_rho, vmax=max_rho,
                          cmap=cmap, shading='auto')
        if network is not None:
            axes[i, 0].scatter(network['lon'], network['lat'],
                               s=network['WBC'] * 20,
                               fc='none', ec='r')
        axes[i, 0].set_xlim(xv[0, 0], xv[0, -1])
        axes[i, 0].set_ylim(yv[0, 0], yv[-1, 0])
        # axes[0].set_title(rf'Modelled {year} $\rho$ (1000s)',
        #               fontsize='small')
        axes[i, 0].tick_params(axis='both', labelsize=7)
        axes[i, 0].set_aspect('equal')
        axes[i, 0].set_ylabel(r"$\bf{" + str(yr) + "}$")

        axes[i, 1].pcolor(xv, yv, rho_obs[yr], vmin=threshold_rho, vmax=max_rho,
                          cmap=cmap, shading='auto')
        if network is not None:  # IF THIS IS USED NEED TO CHANGE TO AV
            axes[i, 1].scatter(
                network['lon'], network['lat'], s=network['WBC'] * 20,
                fc='none', ec='r')
        axes[i, 1].set_xlim(xv[0, 0], xv[0, -1])
        axes[i, 1].set_ylim(yv[0, 0], yv[-1, 0])
        axes[i, 1].tick_params(axis='both', labelsize=7)
        axes[i, 1].set_aspect('equal')

        axes[i, 2].pcolor(xv, yv, simobs_diff[yr], cmap=cmap_diff,
                          norm=cm_diff_norm, shading='auto',
                          vmax=threshold_perc)
        # if network is not None:
        #     axes[1].scatter(network['lon'], network['lat'], s=network['WBC'] * 20,
        #                 fc='none', ec='r')
        axes[i, 2].set_xlim(xv[0, 0], xv[0, -1])
        axes[i, 2].set_ylim(yv[0, 0], yv[-1, 0])
        # axes[1].set_title(rf'{year} $\rho_{{sim}} - \rho_{{obs}}$',
        #               fontsize='small')
        axes[i, 2].tick_params(axis='both', labelsize=7)
        axes[i, 2].set_aspect('equal')
    plt.tight_layout(h_pad=1)
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.show()

    if threshold_rho is not None:
        min_rho = threshold_rho
    cbfig = plt.figure(figsize=(fig.get_size_inches()[0], 1.5))
    cbfig.patch.set_facecolor('w')
    subfigs = cbfig.subfigures(1, 2)
    ax1 = subfigs[0].add_axes([0.05, 0.80, 0.9, 0.1])
    cb1 = mpl.colorbar.ColorbarBase(
        ax1, orientation='horizontal', cmap=cmap,
        norm=mpl.colors.Normalize(vmin=min_rho / 1000, vmax=max_rho / 1000),
        extend='neither', label=r'$\rho$ (1000 cap km$^{-2}$)')
    cb1.ax.tick_params(labelsize='x-small')
    cb1.ax.set_xlabel(r'$\rho$ (1000 cap km$^{-2}$)',
                      fontsize='x-small')
    # cb1.ax.set_xticks(np.sort(np.append(cb1.ax.get_xticks(), max_rho)))
    # cb1.ax.xaxis.set_ticklabels([str(int(l)) if int(l) == l else f'{l:.3f}'
    #      for l in cb1.get_ticks()])

    ax2 = subfigs[1].add_axes([0.05, 0.80, 0.9, 0.1])
    cb2 = mpl.colorbar.ColorbarBase(ax2, orientation='horizontal',
                                    cmap=cmap_diff, norm=cm_diff_norm)
    cb2.ax.tick_params(labelsize='x-small')
    cb2.ax.set_xlabel(r'$\rho_{{sim}} - \rho_{{obs}}$ (1000 cap km$^{-2}$)',
                      fontsize='x-small')

    if cbar_savepath is not None:
        plt.savefig(cbar_savepath, dpi=300)
    plt.show()

    # fig.colorbar(cb_diff, label=r'$\rho_{sim} - \rho_{obs}$ (1000s)', ax=ax3,
    #              location='right')
    # cb_diff_fig = fig.colorbar(cb_diff, ax=ax3, location='right')


def plot_obssim4_multiple(rho_sim, rho_obs, xv, yv, years, cmap="viridis",
                          cmap_diff="bwr", perc_diff=False,
                          threshold_rho=None, threshold_perc=None, borders=None,
                          under_col_rho=None, over_col_perc=None, network=None,
                          save_path=None, cbar_savepath=None, show=True):
    """
    Plot simulated population densities for multiple years, and the difference
    between observed and simulated.

    :param rho_sim: numpy.ndarray
        Simulated population density
    :param rho_obs: numpy.ndarray
        Observed population density
    :param xv:
        Meshgrid in x-direction
    :param yv:
        Meshgrid in y-direction
    :param years: int
        Years of data to be plotted
    :param cmap: matplotlib.colors.ListedColormap
        Colormap for all plots
    :param cmap_diff: str
        Colormap for difference plot
    :param perc_diff: bool
        True: show difference as a percentage, False: difference as magnitude
    :param threshold_rho: numpy.ndarray

    :param threshold_perc: numpy.ndarray
        Threshold of percentage differece colormap
    :param network:
    :param save_path: str
        Save path
    :param show:
    :return:
    """
    if borders is not None:
        rho_sim = {yr: trim_array(rho_sim[yr], borders) for yr in years}
        rho_obs = {yr: trim_array(rho_obs[yr], borders) for yr in years}
        xv = trim_array(xv, borders)
        yv = trim_array(yv, borders)

    obs_max_rho = np.max([np.nanmax(rho_obs[yr]) for yr in years])
    obs_min_rho = np.min([np.nanmin(rho_obs[yr]) for yr in years])
    sim_max_rho = np.max([np.nanmax(rho_sim[yr]) for yr in years])
    sim_min_rho = np.min([np.nanmin(rho_sim[yr]) for yr in years])

    max_rho = np.max([obs_max_rho, sim_max_rho])
    min_rho = np.min([obs_min_rho, sim_min_rho])

    simobs_diff = {yr: (rho_sim[yr] - rho_obs[yr]) / 1000
                   for yr in years}

    if perc_diff:
        for yr in years:
            simobs_diff[yr] /= (rho_obs[yr] / 1000) * 100

    # setting xv and yv to origin
    xv = xv - xv.min()
    yv = yv - yv.min()

    if perc_diff:
        simobs_diffmin = np.min([np.nanmin(simobs_diff[yr])
                                 for yr in years])
        cm_diff_norm = \
            colors.TwoSlopeNorm(vmin=simobs_diffmin, vmax=threshold_perc,
                                vcenter=0.)
    else:
        simobs_diffmin = np.min([np.nanmin(simobs_diff[yr])
                                 for yr in years])
        simobs_diffmax = np.max([np.nanmax(simobs_diff[yr])
                                 for yr in years])
        if simobs_diffmin >= 0:
            cm_diff_norm = None
        else:

            cm_diff_norm = \
                colors.TwoSlopeNorm(vmin=simobs_diffmin,
                                    vmax=simobs_diffmax, vcenter=0.)

    fig, axes = plt.subplots(nrows=len(years), ncols=4, figsize=(9, 9),
                             gridspec_kw={'width_ratios': [3, 3, 3, 2],
                                          'height_ratios': [1, 1, 1, 1]})
    fig.patch.set_facecolor('w')

    for i, yr in enumerate(years):

        if type(cmap) == str:
            cmap = cmx.get_cmap(cmap).copy()
        if under_col_rho is None:
            under_col_rho = 'w'
        cmap.set_under(under_col_rho)

        if type(cmap_diff) == str:
            cmap_diff = cmx.get_cmap(cmap_diff).copy()

        if perc_diff:
            if over_col_perc is None:
                over_col_perc = 'k'
            cmap_diff.set_over(over_col_perc)

        axes[i, 0].pcolor(xv, yv, rho_sim[yr], vmin=threshold_rho, vmax=max_rho,
                          cmap=cmap, shading='auto')
        if network is not None:
            axes[i, 0].scatter(network['lon'], network['lat'],
                               s=network['WBC'] * 20,
                               fc='none', ec='r')
        axes[i, 0].set_xlim(xv[0, 0], xv[0, -1])
        axes[i, 0].set_ylim(yv[0, 0], yv[-1, 0])
        axes[i, 0].tick_params(axis='both', labelsize=7)
        axes[i, 0].set_aspect('equal')
        axes[i, 0].set_ylabel(r"$\bf{" + str(yr) + "}$")

        axes[i, 1].pcolor(xv, yv, rho_obs[yr], vmin=threshold_rho, vmax=max_rho,
                          cmap=cmap, shading='auto')
        if network is not None:  # IF THIS IS USED NEED TO CHANGE TO AV
            axes[i, 1].scatter(
                network['lon'], network['lat'], s=network['WBC'] * 20,
                fc='none', ec='r')
        axes[i, 1].set_xlim(xv[0, 0], xv[0, -1])
        axes[i, 1].set_ylim(yv[0, 0], yv[-1, 0])
        axes[i, 1].tick_params(axis='both', labelsize=7)
        axes[i, 1].set_aspect('equal')

        axes[i, 2].pcolor(xv, yv, simobs_diff[yr], cmap=cmap_diff,
                          norm=cm_diff_norm, shading='auto',
                          vmax=threshold_perc)
        axes[i, 2].set_xlim(xv[0, 0], xv[0, -1])
        axes[i, 2].set_ylim(yv[0, 0], yv[-1, 0])
        axes[i, 2].tick_params(axis='both', labelsize=7)
        axes[i, 2].set_aspect('equal')

        plot_accuracy_scatter(obs_data=rho_obs[yr], sim_data=rho_sim[yr],
                              varobs=r'Obs. $\rho$ (1000s)',
                              varsim=r'Sim. $\rho$ (1000s)',
                              ax=axes[i, 3], log=True)
    plt.tight_layout(h_pad=1)
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.show()

    if threshold_rho is not None:
        min_rho = threshold_rho
    cbfig = plt.figure(figsize=(fig.get_size_inches()[0], 1.5))
    cbfig.patch.set_facecolor('w')
    subfigs = cbfig.subfigures(1, 2)
    ax1 = subfigs[0].add_axes([0.05, 0.80, 0.9, 0.1])
    cb1 = mpl.colorbar.ColorbarBase(
        ax1, orientation='horizontal', cmap=cmap,
        norm=mpl.colors.Normalize(vmin=min_rho / 1000, vmax=max_rho / 1000),
        extend='neither', label=r'$\rho$ (1000 cap km$^{-2}$)')
    cb1.ax.tick_params(labelsize='x-small')
    cb1.ax.set_xlabel(r'$\rho$ (1000 cap km$^{-2}$)',
                      fontsize='x-small')

    ax2 = subfigs[1].add_axes([0.05, 0.80, 0.9, 0.1])
    cb2 = mpl.colorbar.ColorbarBase(ax2, orientation='horizontal',
                                    cmap=cmap_diff, norm=cm_diff_norm)
    cb2.ax.tick_params(labelsize='x-small')
    cb2.ax.set_xlabel(r'$\rho_{{sim}} - \rho_{{obs}}$ (1000 cap km$^{-2}$)',
                      fontsize='x-small')

    if cbar_savepath is not None:
        plt.savefig(cbar_savepath, dpi=300)
    plt.show()


def plot_evolution_population(rho, xv, yv, year, network=None, save_path=None):
    fig = plt.figure(constrained_layout=True)

    spec = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)
    ax1 = fig.add_subplot(spec[0, 0:2])
    ax2 = fig.add_subplot(spec[1:, 0:2])
    ax3 = fig.add_subplot(spec[1:, -1])

    ax1.plot(xv[0, :], np.mean(rho, axis=0))
    ax1.set_xlim(xv[0, 0], xv[0, -1])
    ax1.set_title(r'Mean $\rho_{pop}$ along x-axis', fontsize='small')

    ax2.pcolor(xv, yv, rho, cmap='jet')
    if network is not None:
        ax2.scatter(network['lon'], network['lat'], cmap='jet',
                    c=network['WBC'] / network['WBC'].max(), s=4)
    ax2.set_xlim(xv[0, 0], xv[0, -1])
    ax2.set_ylim(yv[0, 0], yv[-1, 0])
    ax2.set_title(rf'Modelled {year} $\rho_{{pop}}$ and London rail network',
                  fontsize='small')

    ax3.plot(np.mean(rho, axis=1), yv[:, 0])
    ax3.set_ylim(yv[0, 0], yv[-1, 0])
    ax3.set_title(r'Mean $\rho_{pop}$ along y-axis', fontsize='small')

    sm2 = plt.cm.ScalarMappable(cmap='jet',
                                norm=plt.Normalize(vmin=np.min(rho / 1000),
                                                   vmax=np.max(rho / 1000)))
    sm2._A = []
    fig.colorbar(sm2, label='$\\rho_{pop}$ (1000s)', ax=ax2, location='left')

    # plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_cc_wbc(networkdf, cmap, var, var_label, col_midpoint=None,
                save_path=None):
    # TODO fix this
    cmap = plt.get_cmap(cmap)

    colours = networkdf[var] / networkdf[var].max()
    if col_midpoint:
        cmap = shiftedColorMap(cmap=cmap, start=colours.min(),
                               midpoint=col_midpoint, stop=colours.max())
    fig, ax = plt.subplots()
    ax.scatter(x=networkdf['lon'], y=networkdf['lat'], cmap=cmap, c=colours,
               s=4)
    sm2 = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=np.min(networkdf[var]),
                                                   vmax=np.max(networkdf[var])))
    sm2._A = []
    cb = plt.colorbar(sm2, label=var_label, ax=ax)
    if col_midpoint:
        cb.ax.plot(np.mean(cb.ax.get_xlim()), np.median(networkdf[var]),
                   marker='*', c='k')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_beta_comparison(eta, beta_values, save_path=None):
    """
    Compares the impact of different values of beta on distribution array eta
    :return:
    """

    if len(beta_values) != 4:
        raise ValueError("There must be exactly 4 values for comparison")

    beta_exp1 = np.exp(beta_values[0] * eta)
    beta_exp1 /= beta_exp1.sum()

    beta_exp2 = np.exp(beta_values[1] * eta)
    beta_exp2 /= beta_exp2.sum()

    beta_exp3 = np.exp(beta_values[2] * eta)
    beta_exp3 /= beta_exp3.sum()

    beta_exp4 = np.exp(beta_values[3] * eta)
    beta_exp4 /= beta_exp4.sum()

    sum_div = eta / eta.sum()
    sfmt = ticker.ScalarFormatter(useMathText=True)
    sfmt.set_powerlimits((0, 0))

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=[16.98, 8.44])

    beta1 = axes[0, 0].pcolor(beta_exp1)
    b1cb = fig.colorbar(beta1, ax=axes[0, 0], format=sfmt)
    axes[0, 0].set_title(rf'$\beta$ = {beta_values[0]}')

    beta2 = axes[0, 1].pcolor(beta_exp2)
    b2cb = fig.colorbar(beta2, ax=axes[0, 1], format=sfmt)
    axes[0, 1].set_title(rf'$\beta$ = {beta_values[1]}')

    beta3 = axes[0, 2].pcolor(beta_exp3)
    b3cb = fig.colorbar(beta3, ax=axes[0, 2], format=sfmt)
    axes[0, 2].set_title(rf'$\beta$ = {beta_values[2]}')

    beta4 = axes[1, 0].pcolor(beta_exp4)
    b4cb = fig.colorbar(beta4, ax=axes[1, 0], format=sfmt)
    axes[1, 0].set_title(rf'$\beta$ = {beta_values[3]}')

    sum_ex = axes[1, 1].pcolor(sum_div)
    sumcb = fig.colorbar(sum_ex, ax=axes[1, 1], format=sfmt)
    axes[1, 1].set_title(r'Sum only')
    axes[1, 2].plot(sorted(sum_div.flatten()), label='Sum ref.')
    axes[1, 2].plot(sorted(beta_exp1.flatten()),
                    label=rf'$\beta$ = {beta_values[0]}')
    axes[1, 2].plot(sorted(beta_exp2.flatten()),
                    label=rf'$\beta$ = {beta_values[1]}')
    axes[1, 2].plot(sorted(beta_exp3.flatten()),
                    label=rf'$\beta$ = {beta_values[2]}')
    axes[1, 2].plot(sorted(beta_exp4.flatten()),
                    label=rf'$\beta$ = {beta_values[3]}')
    axes[1, 2].legend()
    axes[1, 2].set_ylabel(r'$\eta (i)$')
    axes[1, 2].yaxis.set_major_formatter(sfmt)
    axes[1, 2].set(xticklabels=[])

    for ax in [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]]:
        ax.set(yticklabels=[])
        ax.set(xticklabels=[])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
# %%
def softmax(arr, beta):
    kexp = np.exp(beta * arr)
    kexp /= kexp.sum()
    return kexp

def bar_beta_softmax(save_path):
    arrs = np.zeros(shape=(5, 5))
    arr = np.arange(1, 6)
    # sum_div
    arrs[0, :] = arr / arr.sum()
    # beta_exp 0.5
    arrs[1, :] = softmax(arr, 0.5)
    # beta_exp 1
    arrs[2, :] = softmax(arr, 1)
    # beta_exp 2
    arrs[3, :] = softmax(arr, 2)
    # beta_exp 10
    arrs[4, :] = softmax(arr, 10)

    colrs = [plt.cm.viridis(x) for x in np.linspace(0, 1, len(arr))]

    N = 5
    bar_width = .5
    xloc = np.arange(N)

    scheme = ['', 'Sum only', '$\\beta$ = 0.5', '$\\beta$ = 1', '$\\beta$ = 2',
              '$\\beta$ = 10']

    fig, ax = plt.subplots()
    ax.bar(xloc, arrs[:, 0], width=bar_width, color=colrs[0], label='1')
    ax.bar(xloc, arrs[:, 1], bottom=arrs[:, 0], width=bar_width, color=colrs[1],
           label='2')
    ax.bar(xloc, arrs[:, 2], bottom=arrs[:, 0] + arrs[:, 1], width=bar_width,
           color=colrs[2], label='3')
    ax.bar(xloc, arrs[:, 3], bottom=arrs[:, 0] + arrs[:, 1] + arrs[:, 2],
           width=bar_width, color=colrs[3], label='4')
    ax.bar(xloc, arrs[:, 4], bottom=arrs[:, 0] + arrs[:, 1] + arrs[:, 2] +
                                    arrs[:, 3],
           width=bar_width, color=colrs[4], label='5')

    ax.set_title('Comparative softmax weighting of different\n'
                 '$\\beta$ values for array [1,2,3,4,5]')
    ax.set_xticklabels(scheme)
    plt.legend()

    if save_path:
        plt.savefig(save_path)

def plot_max_gi(max_nbc, years, save_path=None):
    """
    Plots the maximum g(i) across the years

    :param max_nbc: list
        Maximum average node betweenness centrality in domain
    :param years: list
        Years to plot
    :param save_path: str
        Save path
    :return:
    """
    fig, ax = plt.subplots()
    ax.plot(max_nbc, color='r')
    ax.set_title('Maximum weighted node betweenness centrality (NBC)')
    ax.set_xticks(np.arange(len(max_nbc)))
    ax.set_xticklabels(years, rotation=90)
    ax.set_xlabel('Year')
    ax.set_ylabel('Max NBC')
    ax.set_ylim(0.1, 0.6)
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()

def plot_obs_sim_pop(observed_N, simulated_N, r, save_path=None):
    """
    Plots observed and simulated total domain populations
    :param observed_N: numpy.ndarray, list
    :param simulated_N: numpy.ndarray, list
    :param r: float
        Growth rate
    :param save_path: str
        Save path
    :return:
    """
    fig, ax = plt.subplots()

    ax.plot(np.arange(0, 190, 10), observed_N, color='b', label='Obs.')
    ax.plot(simulated_N, color='r', label='Sim.')
    ax.set_title('Simulated and observed total\npopulations across domain')
    ax.set_xticks(np.arange(len(simulated_N), step=10))
    ax.set_xticklabels(np.arange(1831, 2012, 10), rotation=90)
    ax.set_xlabel('Year')
    ax.set_ylabel('Population (1e7)')
    ax.text(0.85, 0.1, f'$r$={r}', transform=ax.transAxes)
    plt.legend()

    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()

def plot_gi_ct(gmgi, years, gm_eq=None, save_path=None):
    """
    Plots boxplots and scatter plots of normalised gm - g(i) under different
    scenarios to investigate spread
    :param gmgi:
    :param years:
    :param gm_eq:
    :param save_path: str
        Save path
    :return:
    """
    fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(14, 14))

    twinaxes = np.empty_like(axes)
    for i in np.arange(0, len(axes[:, 0])):
        for j in np.arange(0, len(axes[0, :])):
            twinaxes[i, j] = axes[i, j].twiny()

    min_ct = np.min([gmgi[yr]['gmgi'].min()
                     for yr in years
                     if yr in gmgi.keys()])

    for i in np.arange(0, len(axes[:, 0])):
        for j in np.arange(0, len(axes[0, :])):
            if (i * 4 + j) < len(years):
                yr = years[i * 4 + j]
                if yr in gmgi.keys():
                    norm_gmgi = gmgi[yr]['gmgi'] / gmgi[yr]['gm'].values[0]
                    twinaxes[i, j].boxplot(norm_gmgi, showfliers=False)

                    axes[i, j].scatter(gmgi[yr]['dist'], norm_gmgi, s=2,
                                       alpha=0.5)
                    axes[i, j].text(0.8, 0.1, f'{yr}',
                                    transform=axes[i, j].transAxes,
                                    size=8)

            for pos in ['top', 'right']:
                axes[i, j].spines[pos].set_visible(False)
                twinaxes[i, j].spines[pos].set_visible(False)

            axes[i, j].tick_params(axis='x', which='both', top=False)
            twinaxes[i, j].tick_params(axis='x', which='both', top=False)

            axes[i, j].set_ylim(bottom=0, top=1)

            if i < 4 and j < 2:
                axes[i, j].set_xticklabels("")
            if i < 3 and j >= 2:
                axes[i, j].set_xticklabels("")

            if i == 4 and j in [2, 3]:
                for pos in ['bottom', 'left']:
                    axes[i, j].spines[pos].set_visible(False)
                    twinaxes[i, j].spines[pos].set_visible(False)
                axes[i, j].tick_params(axis='both', which='both', bottom=False,
                                       left=False)
                twinaxes[i, j].tick_params(axis='both', which='both',
                                           bottom=False, left=False)
                axes[i, j].set_yticklabels("")
                axes[i, j].set_xticklabels("")

            twinaxes[i, j].set_xticklabels("")

            axes[i, j].tick_params(labelsize=8)

    axes[2, 0].set_ylabel('$g_{m} - \\bar g(i)$', fontsize=10)
    axes[3, 2].set_xlabel('Distance from centre (km)', fontsize=10)

    if gm_eq:
        gm = gm_eq
    else:
        gm = gmgi[years[0]]['gm'].values[0]

    axes[4, 3].text(0.5, 0.5, f'$g_{{m}}$ = {gm}',
                    transform=axes[4, 3].transAxes,
                    size=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)

    plt.show()
    plt.close()


def plot_radial(y, yvar, title, centre=None, nan_val=0.0, savepath=None, \
                borders=None, show=False, ylim=None):
    """
    Plots the radial profile of 2D data.

    :param rho_sim: numpy.ndarray
        Simulated population density array
    :param rho_obs: numpy.ndarray
        Observed population density array
    :param centre: tuple
        Centre point of both arrays
    :return:
        np.ndarray
    """

    y_rad = radial_profile(y, centre, nan_val=nan_val)[0]

    fig, ax = plt.subplots(figsize=(5.5, 2))
    fig.patch.set_facecolor('w')
    ax.plot(y_rad, zorder=3, color='r')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # ax.set_aspect(0.4)
    ax.set_xlim(xmin=0, xmax=len(y_rad))
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    ax.set_xlabel('Distance from centre (km)', fontdict={'size': 8})
    ax.set_ylabel(yvar, fontdict={'size': 8})
    ax.set_title(title)
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
    if show:
        plt.show()
    plt.close()

def plot_radial_obs_sim(rho_sim, rho_obs=None, centre=None, nan_val=0.0,
                        savepath=None, borders=None, show=False):
    """
    Plots the radial profiles of simulated and observed population density, and
    the difference between each.

    :param rho_sim: numpy.ndarray
        Simulated population density array
    :param rho_obs: numpy.ndarray
        Observed population density array
    :param centre: tuple
        Centre point of both arrays
    :param nan_val: float
        Fill values for NaNs in radial profiles
    :return:
        np.ndarray
    """
    if borders is not None:
        rho_sim = trim_array(rho_sim, borders)
        if rho_obs is not None:
            rho_obs = trim_array(rho_obs, borders)

    sim_rad = radial_profile(rho_sim/1000, centre, nan_val=nan_val)[0]
    if rho_obs is not None:
        obs_rad = radial_profile(rho_obs/1000, centre, nan_val=nan_val)[0]
        sim_obs_diff = radial_profile((rho_sim - rho_obs)/1000, centre,
                                      nan_val=nan_val)[0]

    fig, ax = plt.subplots(figsize=(5.5,2))
    fig.patch.set_facecolor('w')
    ax.plot(sim_rad, zorder=3, color='r', label=r"$\rho_{sim}$")
    if rho_obs is not None:
        ax.plot(obs_rad, zorder=2, color='b', label=r"$\rho_{obs}$")
        ax.plot(sim_obs_diff, color='grey', ls=':', zorder=1,
                label=r"$\rho_{sim} - \rho_{obs}$")
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend()
    # ax.set_aspect(0.4)
    ax.set_xlim(xmin=0, xmax=len(sim_rad))
    if rho_obs is not None:
        ax.set_title(
            'Radial profiles of simulated and observed population density',
            fontdict={'size': 10})
    ax.set_xlabel('Distance from centre (km)', fontdict={'size': 8})
    ax.set_ylabel(r'Mean $\rho$ (1000 cap km$^{-2}$)', fontdict={'size': 8})
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
    if show:
        plt.show()
    plt.close()

def plot_radial_years(rho_dict, years, title, xvar, ymax=None, cmap=None,
                      centre=None, nan_val=0.0, savepath=None, borders=None):
    """
    Plots the radial profiles of simulated and observed population density, and
    the difference between each.

    :param rho_dict: dict
        Simulated population density array
    :param centre: tuple
        Centre point of both arrays
    :param nan_val: float
        Fill values for NaNs in radial profile
    :return:
        np.ndarray
    """
    if cmap is None:
        cmap = 'winter'
    clrs = [cmx.get_cmap(cmap)(c) for c in np.linspace(0, 1, len(years))]

    fig, ax = plt.subplots(figsize=(5.5, 2))
    fig.patch.set_facecolor('w')
    for i, yr in enumerate(years):
        arr = rho_dict[yr]
        if borders is not None:
            arr = trim_array(arr, borders)
        rad = radial_profile(arr / 1000, centre, nan_val=nan_val)[0]
        ax.plot(rad, color=clrs[i])

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_facecolor('lightgrey')
    lines = [mlines.Line2D([], [], color=colour, label=label)
             for label, colour in
             zip([years[0], years[-1]], [clrs[0], clrs[-1]])]
    leg = ax.legend(lines, [years[0], years[-1]],
                    loc='upper right', frameon=False)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)
    # ax.set_aspect(0.4)
    ax.set_xlim(xmin=0, xmax=len(rad))
    ax.set_title(title, fontdict={'size': 10})
    if ymax is not None:
        ax.set_ylim(ymax=ymax)

    ax.set_xlabel('Distance from centre (km)', fontdict={'size': 8})
    ax.set_ylabel(xvar, fontdict={'size': 8})
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
        plt.show()
        plt.close()
    else:
        plt.show()

def plot_two_radial_years(rho_dict1, rho_dict2, years, title=None, cmap=None,
                          centre=None, savepath=None, borders=None,
                          lr_titles=None, nan_val=0.0):
    """
    Plots the radial profiles of simulated and observed population density, and
    the difference between each.

    :param rho_dict1: dict
        Population density array, typically simulated
    :param rho_dict2: dict
        Population density array, typically observed
    :param years: list, numpy.nd.array
        Years to plot from rho_dict1, rho_dict2
    :param title: str
        Figure title
    :param cmap: str
        Colormap
    :param centre: tuple
        Centre point of both arrays
    :param borders: list, numpy.ndarray, tuple
        x, y border to be removed (grid cell size)
    :param lr_titles: list, numpy.ndarray, tuple
        Names of the left and right columns
    :param savepath: str
        Figure save path
    :param centre: tuple
        Centre point for radial profile if using r2 scores
    :param nan_val: float
        Fill values for NaNs in radial profile if using r2 scores
    :return:
        np.ndarray
    """
    if cmap is None:
        cmap = 'winter'
    clrs = [cmx.get_cmap(cmap)(c) for c in np.linspace(0, 1, len(years))]

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(11, 2))
    fig.patch.set_facecolor('w')
    for i, yr in enumerate(years):
        arr1 = rho_dict1[yr]
        arr2 = rho_dict2[yr]
        if borders is not None:
            arr1 = trim_array(arr1, borders)
            arr2 = trim_array(arr2, borders)
        rad1 = radial_profile(arr1 / 1000, centre, nan_val=nan_val)[0]
        rad2 = radial_profile(arr2 / 1000, centre, nan_val=nan_val)[0]
        ax1.plot(rad1, color=clrs[i])
        ax2.plot(rad2, color=clrs[i])

    for ax in (ax1, ax2):
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_facecolor('lightgrey')
        ax.set_xlim(xmin=0, xmax=len(rad1))
        ax.set_ylim(ymax=np.max([ax1.get_ylim()[1], ax2.get_ylim()[1]]))
        ax.set_xlabel('Distance from centre (km)', fontdict={'size': 8})
        ax.set_ylabel(r'Mean $\rho$ (1000 cap km$^{-2}$)', fontdict={'size': 8})
    lines = [mlines.Line2D([], [], color=colour, label=label)
             for label, colour in
             zip([years[0], years[-1]], [clrs[0], clrs[-1]])]
    leg = ax2.legend(lines, [years[0], years[-1]],
                     loc='upper right', frameon=False)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)
    # ax.set_aspect(0.4)
    if title is not None:
        fig.suptitle(title, fontdict={'size': 8})

    if lr_titles is not None:
        if len(lr_titles) == 2:
            ax1.set_title(lr_titles[0], fontdict={'fontsize': 'small'})
            ax2.set_title(lr_titles[1], fontdict={'fontsize': 'small'})

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()
    plt.close()

def plot_xy_mean_years(rho_dict, years, xv, yv, ymax=None, cmaps=None,
                       savepath=None, borders=None):
    """
    Plots means along the x and y axes

    :param rho_dict: dict
        Dictionary of population arrays over years
    :param years: numpy.ndarray
        Years to plot (must be keys in rho_dict
    :param xv: numpy.ndarray

    :param yv: numpy.nd.array

    :param cmap: tuples
        Colormap(s) for x, y
    :param savepath:
        Filepath for saving the figure
    :return:
        np.ndarray
    """
    if cmaps is None:
        cmaps = ('winter', 'autumn')
    clrs_x = [cmx.get_cmap(cmaps[0])(c) for c in np.linspace(0, 1, len(years))]
    clrs_y = [cmx.get_cmap(cmaps[-1])(c) for c in np.linspace(0, 1, len(years))]

    if borders is not None:
        xv = trim_array(xv, borders)
        yv = trim_array(yv, borders)
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(5.5, 4))
    for i, yr in enumerate(years):
        arr = rho_dict[yr]
        if borders is not None:
            arr = trim_array(arr, borders)
        ax1.plot(xv[0, :], np.nanmean(arr / 1000, axis=0),
                 color=clrs_x[i])

    ax1.set_xlim(xv[0, 0], xv[0, -1])
    ax1.set_title(r'Mean $\rho_{pop}$ along x-axis', fontsize='small')
    ax1.legend(fontsize='small')

    for i, yr in enumerate(years):
        arr = rho_dict[yr]
        if borders is not None:
            arr = trim_array(arr, borders)
        ax2.plot(yv[:, 0], np.nanmean(arr / 1000, axis=1),
                 color=clrs_y[i])
    ax2.set_xlim(yv[0, 0], yv[-1, 0])
    ax2.set_title(r'Mean $\rho_{pop}$ along y-axis', fontsize='small')

    for ax in (ax1, ax2):
        ax.set_ylabel(r'$\rho_{pop}$ (1000 cap km$^{-2}$)',
                      fontdict={'size': 8})
        ax.set_facecolor('lightgrey')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if ymax is not None:
            ax.set_ylim(ymax=ymax)

    lines = [mlines.Line2D([], [], color=colour, label=label)
             for label, colour in
             zip([years[0], years[-1]], [clrs_x[0], clrs_x[-1]])]
    leg = ax1.legend(lines, [years[0], years[-1]],
                     loc='upper right', frameon=False)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)

    lines = [mlines.Line2D([], [], color=colour, label=label)
             for label, colour in
             zip([years[0], years[-1]], [clrs_y[0], clrs_y[-1]])]
    leg = ax2.legend(lines, [years[0], years[-1]],
                     loc='upper right', frameon=False)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)

    len_diff = abs(len(xv[0, :]) - len(yv[:, 0]))

    # only works for current shape as x > y
    ax1.set_xlim(xv[0, :].min(), xv[0, :].max())
    ax2.set_xlim(yv[:, 0].min() - len_diff // 2, yv[:, 0].max() + len_diff // 2)

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()
    plt.close()

def plot_two_xy_mean_years(rho_dict1, rho_dict2, years, xv, yv, title=None,
                           cmaps=None, savepath=None, borders=None,
                           lr_titles=None):
    """
    Plots means along the x and y axes

    :param rho_dict1: dict
        Dictionary of population arrays over years, typically simulated
    :param rho_dict2: dict
        Dictionary of population arrays over years, typically observed
    :param years: numpy.ndarray
        Years to plot (must be keys in rho_dict
    :param xv: numpy.ndarray

    :param yv: numpy.nd.array

    :param cmap: tuples
        Colormap(s) for x, y
    :param lr_titles: list, numpy.ndarray, tuple
        Names of the left and right columns
    :param savepath:
        Filepath for saving the figure
    :return:
        np.ndarray
    """
    if borders is not None:
        xv = trim_array(xv, borders)
        yv = trim_array(yv, borders)

    if cmaps is None:
        cmaps = ('winter', 'autumn')
    clrs_x = [cmx.get_cmap(cmaps[0])(c) for c in np.linspace(0, 1, len(years))]
    clrs_y = [cmx.get_cmap(cmaps[-1])(c) for c in np.linspace(0, 1, len(years))]

    fig, ((ax11, ax21), (ax12, ax22)) = \
        plt.subplots(nrows=2, ncols=2, figsize=(8, 4))
    fig.patch.set_facecolor('w')
    for i, yr in enumerate(years):
        arr1 = rho_dict1[yr]
        arr2 = rho_dict2[yr]
        if borders is not None:
            arr1 = trim_array(arr1, borders)
            arr2 = trim_array(arr2, borders)
        ax11.plot(xv[0, :], np.nanmean(arr1 / 1000, axis=0),
                  color=clrs_x[i])
        ax21.plot(xv[0, :], np.nanmean(arr2 / 1000, axis=0),
                  color=clrs_x[i])

    for ax in (ax11, ax21):
        ax.set_xlim(xv[0, 0], xv[0, -1])
        ax.set_xlabel(r'x-direction (km)', fontsize='small')
        ax.set_ylim(ymax=np.max([ax11.get_ylim()[1], ax21.get_ylim()[1]]))
        # ax.legend(fontsize='small')

    for i, yr in enumerate(years):
        arr1 = rho_dict1[yr]
        arr2 = rho_dict2[yr]
        if borders is not None:
            arr1 = trim_array(arr1, borders)
            arr2 = trim_array(arr2, borders)
        ax12.plot(yv[:, 0], np.nanmean(arr1 / 1000, axis=1),
                  color=clrs_y[i])
        ax22.plot(yv[:, 0], np.nanmean(arr2 / 1000, axis=1),
                  color=clrs_y[i])

    for ax in (ax12, ax22):
        ax.set_xlim(yv[0, 0], yv[-1, 0])
        ax.set_xlabel(r'y-direction (km)', fontsize='small')
        ax.set_ylim(ymax=np.max([ax12.get_ylim()[1], ax22.get_ylim()[1]]))

    for ax in (ax11, ax21, ax12, ax22):
        ax.set_ylabel(r'Mean $\rho$ (10$^{3}$ cap km$^{-2}$)',
                      fontdict={'size': 8})
        ax.set_facecolor('lightgrey')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    lines = [mlines.Line2D([], [], color=colour, label=label)
             for label, colour in
             zip([years[0], years[-1]], [clrs_x[0], clrs_x[-1]])]
    leg = ax21.legend(lines, [years[0], years[-1]],
                      loc='upper right', frameon=False)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)

    lines = [mlines.Line2D([], [], color=colour, label=label)
             for label, colour in
             zip([years[0], years[-1]], [clrs_y[0], clrs_y[-1]])]
    leg = ax22.legend(lines, [years[0], years[-1]],
                      loc='upper right', frameon=False)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)

    len_diff = abs(len(xv[0, :]) - len(yv[:, 0]))

    # only works for current shape as x > y
    ax11.set_xlim(xv[0, :].min(), xv[0, :].max())
    ax21.set_xlim(xv[0, :].min(), xv[0, :].max())
    ax12.set_xlim(yv[:, 0].min() - len_diff // 2,
                  yv[:, 0].max() + len_diff // 2)
    ax22.set_xlim(yv[:, 0].min() - len_diff // 2,
                  yv[:, 0].max() + len_diff // 2)

    if title is not None:
        fig.suptitle(title)

    if lr_titles is not None:
        if len(lr_titles) == 2:
            ax11.set_title(lr_titles[0], fontdict={'fontsize': 'small'})
            ax21.set_title(lr_titles[1], fontdict={'fontsize': 'small'})

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()
    plt.close()

def plot_two_xy_mean_years_together(rho_dict1, rho_dict2, years, xv, yv,
                                    title=None, cmaps=None, savepath=None,
                                    borders=None):
    """
    Plots means along the x and y axes

    :param rho_dict1: dict
        Dictionary of population arrays over years, typically simulated
    :param rho_dict2: dict
        Dictionary of population arrays over years, typically observed
    :param years: numpy.ndarray
        Years to plot (must be keys in rho_dict
    :param xv: numpy.ndarray

    :param yv: numpy.nd.array

    :param cmap: tuples
        Colormap(s) for x, y
    :param savepath:
        Filepath for saving the figure
    :return:
        np.ndarray
    """
    if borders is not None:
        xv = trim_array(xv, borders)
        yv = trim_array(yv, borders)

    if cmaps is None:
        cmaps = ('winter', 'autumn')
    clrs_x = [cmx.get_cmap(cmaps[0])(c) for c in np.linspace(0, 1, len(years))]
    clrs_y = [cmx.get_cmap(cmaps[-1])(c) for c in np.linspace(0, 1, len(years))]

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(5.5, 4))
    fig.patch.set_facecolor('w')
    for i, yr in enumerate(years):
        arr1 = rho_dict1[yr]
        arr2 = rho_dict2[yr]
        if borders is not None:
            arr1 = trim_array(arr1, borders)
            arr2 = trim_array(arr2, borders)
        ax1.plot(xv[0, :], np.nanmean(arr1 / 1000, axis=0),
                 color=clrs_x[i])
        ax1.plot(xv[0, :], np.nanmean(arr2 / 1000, axis=0), ls='dashed',
                 color=clrs_x[i])

        ax2.plot(yv[:, 0], np.nanmean(arr1 / 1000, axis=1),
                 color=clrs_y[i])
        ax2.plot(yv[:, 0], np.nanmean(arr2 / 1000, axis=1), ls='dashed',
                 color=clrs_y[i])

    ax1.set_xlim(xv[0, 0], xv[0, -1])
    ax1.set_xlabel(r'x-direction (km)', fontsize='small')

    ax2.set_xlim(yv[0, 0], yv[-1, 0])
    ax2.set_xlabel(r'y-direction (km)', fontsize='small')

    for ax in (ax1, ax2):
        ax.set_ylabel(r'Mean $\rho$ (10$^{3}$ cap km$^{-2}$)',
                      fontdict={'size': 8})
        ax.set_facecolor('lightgrey')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    lines = [mlines.Line2D([], [], color=colour, label=label)
             for label, colour in
             zip([years[0], years[-1]], [clrs_x[0], clrs_x[-1]])]
    leg = ax1.legend(lines, [years[::2][0], years[::2][-1]],
                     loc='upper right', frameon=False)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)

    lines = [mlines.Line2D([], [], color=colour, label=label)
             for label, colour in
             zip([years[0], years[-1]], [clrs_y[0], clrs_y[-1]])]
    leg = ax2.legend(lines, [years[::2][0], years[::2][-1]],
                     loc='upper right', frameon=False)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)

    len_diff = abs(len(xv[0, :]) - len(yv[:, 0]))

    # only works for current shape as x > y
    ax1.set_xlim(xv[0, :].min(), xv[0, :].max())
    ax2.set_xlim(yv[:, 0].min() - len_diff // 2,
                 yv[:, 0].max() + len_diff // 2)

    if title is not None:
        fig.suptitle(title)

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()

def plot_xy_mean_obssim(rho_dict1, rho_dict2, year, xv, yv, spatial_dict=None,
                        ymax=None, norm=False, savepath=None):
    """
    Plots means along the x and y axes

    :param rho_dict1: dict
        Dictionary of population arrays over years, typically simulated
    :param rho_dict2: dict
        Dictionary of population arrays over years, typically observed
    :param year: numpy.ndarray
        Year to plot (must be key in rho_dict)
    :param xv: numpy.ndarray

    :param yv: numpy.nd.array

    :param cmap: tuples
        Colormap(s) for x, y
    :param savepath: str
        Filepath for saving the figure
    :return:
        np.ndarray
    """
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(5.5, 4))

    if spatial_dict is not None and year in spatial_dict.keys():
        twinax1 = ax1.twinx()
        twinax2 = ax2.twinx()

        ax1.patch.set_visible(False)
        # Set axtwin's patch visible and colour it
        twinax1.patch.set_visible(True)
        twinax1.patch.set_facecolor('lightgrey')
        ax1.set_zorder(twinax1.get_zorder() + 1)

        ax2.patch.set_visible(False)
        # Set axtwin's patch visible and colour it
        twinax2.patch.set_visible(True)
        twinax2.patch.set_facecolor('lightgrey')
        ax2.set_zorder(twinax2.get_zorder() + 1)

    else:
        ax1.set_facecolor('lightgrey')
        ax2.set_facecolor('lightgrey')

    xv -= xv.min()
    yv -= yv.min()

    x1 = np.nanmean(rho_dict1[year] / 1000, axis=0)
    x2 = np.nanmean(rho_dict2[year] / 1000, axis=0)
    y1 = np.nanmean(rho_dict1[year] / 1000, axis=1)
    y2 = np.nanmean(rho_dict2[year] / 1000, axis=1)

    norm_title = ''
    y_label = r'$\rho_{pop}$ (1000 cap km$^{-2}$)'
    if norm:
        max_ob_x = np.nanmax(x2)
        x1 /= max_ob_x
        x2 /= max_ob_x
        max_ob_y = np.nanmax(y2)
        y1 /= max_ob_y
        y2 /= max_ob_y

        norm_title = '(normalised)'
        y_label = r'Normalised $\rho_{pop}$'

    ax1.plot(xv[0, :], x2, color='b', label='Obs.')
    ax1.plot(xv[0, :], x1, color='r', label='Sim')
    if spatial_dict is not None and year in spatial_dict.keys():
        hght, bins = spatial_binning(array=spatial_dict[year], dir='x',
                                     bin_width=5)
        twinax1.bar(bins[:-1] + 2.5, height=hght, width=np.diff(bins),
                    color='grey', edgecolor='k')
        twinax1.set_ylabel('No. of stations')
    ax1.set_xlim(xv[0, 0], xv[0, -1])
    ax1.set_title(rf'Mean $\rho_{{pop}}$ along x-axis {norm_title}',
                  fontsize='small')
    ax1.legend(fontsize='small')

    ax2.plot(yv[:, 0], y2, color='b')
    ax2.plot(yv[:, 0], y1, color='r')
    if spatial_dict is not None and year in spatial_dict.keys():
        hght, bins = spatial_binning(array=spatial_dict[year], dir='y',
                                     bin_width=5)
        twinax2.bar(bins[:-1] + 2.5, height=hght, width=np.diff(bins),
                    color='grey', edgecolor='k')
        twinax2.set_ylabel('No. of stations')
    ax2.set_xlim(yv[0, 0], yv[-1, 0])
    ax2.set_title(rf'Mean $\rho_{{pop}}$ along y-axis {norm_title}',
                  fontsize='small')

    for ax in (ax1, ax2):
        ax.set_ylabel(y_label, fontdict={'size': 8})
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if ymax is not None:
            ax.set_ylim(ymax=ymax)

    ty = ax1.text(x=0.05, y=0.85, s=f'{year}', transform=ax1.transAxes)
    ty.set_bbox(dict(facecolor='w', alpha=0.5, edgecolor='w'))

    ax1.text(x=0.05, y=0.5, horizontalalignment='left', s='W',
             transform=ax1.transAxes)
    ax1.text(x=0.95, y=0.5, horizontalalignment='right', s='E',
             transform=ax1.transAxes)
    ax2.text(x=0.05, y=0.5, horizontalalignment='left', s='S',
             transform=ax2.transAxes)
    ax2.text(x=0.95, y=0.5, horizontalalignment='right', s='N',
             transform=ax2.transAxes)

    len_diff = abs(len(xv[0, :]) - len(yv[:, 0]))
    # only works for current shape as x > y
    ax1.set_xlim(xv[0, :].min(), xv[0, :].max())
    ax2.set_xlim(yv[:, 0].min() - len_diff // 2, yv[:, 0].max() + len_diff // 2)

    ax2xticks = np.array([[i, x] for i, x in enumerate(ax2.get_xticks())
                          if x >= 0 and x <= len(y2)])
    ax2.set_xticks(ax2xticks[:, 1])

    ax1.minorticks_on()
    ax2.minorticks_on()

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()
    if savepath is not None:
        plt.close()


def over_density_hatch(results, eta, xv, yv, K, r):
    """
    Hatching plotted when local density above domain density

    :param results:
    :param eta:
    :param xv:
    :param yv:
    :param K:
    :param r:
    :return:
    """
    for yr in results.keys():
        print(yr)
        yr_res = results[yr]
        domain_area = yr_res.shape[0] * yr_res.shape[1]
        P = np.around(np.nansum(yr_res) / domain_area, 4)
        over = np.where(results[yr] > P, 1, np.nan)

        fig, ax = plt.subplots()
        ax.pcolor(results[yr])
        ax.contourf(over, hatches=['//'], alpha=0)
        ax.set_title(
            f"{yr}, $P$: {P}, $N$ where $\\rho > P$: {np.nansum(over)}")
        plt.savefig(f'./plots/roughSA/locdom/{yr}_densitydiff.png')
        # estimate effective growth domain vs local
        q_local = (eta[yr] * yr_res * r * (1 - (P / K))) / 1000
        q_domain = (eta[yr] * P * r * (1 - (P / K))) / 1000

        q_local = np.where(np.isnan(q_local), 0, q_local)
        q_domain = np.where(np.isnan(q_domain), 0, q_domain)

        min_q = min(np.nanmin(q_local), np.nanmin(q_domain))
        max_q = max(np.nanmax(q_local), np.nanmax(q_domain))
        print(max_q)
        q_diff = (q_local - q_domain)

        cm_diff_norm = \
            colors.TwoSlopeNorm(vmin=q_diff.min(), vmax=q_diff.max(),
                                vcenter=0.)

        fig = plt.figure(constrained_layout=True)

        subfigs = fig.subfigures(2, 1)

        spec1 = gridspec.GridSpec(ncols=4, nrows=1, figure=subfigs[0])
        ax1 = subfigs[0].add_subplot(spec1[0:2])
        ax2 = subfigs[0].add_subplot(spec1[2:4])
        spec2 = gridspec.GridSpec(ncols=12, nrows=1, figure=subfigs[1])
        ax3 = subfigs[1].add_subplot(spec2[3:9])
        caxrho = subfigs[1].add_subplot(spec2[1:2])
        caxdiff = subfigs[1].add_subplot(spec2[10:11])

        ax1.pcolor(xv, yv, q_local, vmin=min_q, vmax=max_q)
        ax1.set_xlim(xv[0, 0], xv[0, -1])
        ax1.set_ylim(yv[0, 0], yv[-1, 0])
        ax1.set_title(f"Local: $dP$ = "
                      f"{np.around(np.nansum(q_local) / domain_area, 4)}")
        ax1.tick_params(axis='both', labelsize=6)
        ax1.set_aspect('equal')

        ax2.pcolor(xv, yv, q_domain, vmin=min_q, vmax=max_q)
        ax2.set_xlim(xv[0, 0], xv[0, -1])
        ax2.set_ylim(yv[0, 0], yv[-1, 0])
        ax2.set_title(f"Domain: $dP$ = "
                      f"{np.around(np.nansum(q_domain) / domain_area, 4)}")
        ax2.tick_params(axis='both', labelsize=6)
        ax2.set_aspect('equal')

        subfigs[0].suptitle('Growth using local ($\\rho$) and domain ($P$) '
                            'logistic function')

        ax3.pcolor(xv, yv, q_diff, cmap='bwr', norm=cm_diff_norm)
        ax3.set_xlim(xv[0, 0], xv[0, -1])
        ax3.set_ylim(yv[0, 0], yv[-1, 0])
        ax3.set_title(rf'{yr} $q_{{\rho}} - q_{{P}}$ (1000 cap km$^{{-2}}$)',
                      fontsize='small')
        ax3.tick_params(axis='both', labelsize=6)
        ax3.set_aspect('equal')

        cb_q = plt.cm.ScalarMappable(cmap='viridis',
                                     norm=plt.Normalize(vmin=min_q,
                                                        vmax=max_q))
        cb_q._A = []
        cb_rho_fig = subfigs[0].colorbar(cb_q, cax=caxrho,
                                         extend='min', shrink=0.6)
        cb_rho_fig.ax.tick_params(labelsize=6)
        cb_rho_fig.ax.set_ylabel(ylabel=r'$d\rho$ (1000 cap km$^{-2}$)',
                                 fontdict={'size': 6})

        cb_diff = plt.cm.ScalarMappable(cmap='bwr', norm=cm_diff_norm)
        cb_diff._A = []

        cb_diff_fig = subfigs[1].colorbar(cb_diff, cax=caxdiff, shrink=0.6)
        cb_diff_fig.ax.tick_params(labelsize=6)
        cb_diff_fig.ax.set_ylabel(ylabel=r'$q_{\rho} - q_{P}$ '
                                         r'(1000 cap km$^{-2}$)',
                                  fontdict={'size': 6})
        # plt.tight_layout()
        plt.savefig(f'./plots/roughSA/locdom/{yr}_growthdiff.png')
        plt.show()

def plot_kurtosis(rhodict1, rhodict2=None, labels=None, savepath=None):
    """
    Plot kurtosis of mean x and y profiles

    :param rhodict1: dict
         Dictionary of population arrays over years
    :param rho_dict1: dict
         Dictionary of population arrays over years
    :return:
    """
    years = sorted(list(rhodict1.keys()))
    xk_1, yk_1 = xy_kurtosis(arr_dict=rhodict1, years=years)

    if rhodict2 is not None:
        xk_2, yk_2 = xy_kurtosis(arr_dict=rhodict2, years=years)

    total_pop = []

    for yr in years:
        total_pop.append(rhodict1[yr].sum() / 1e7)

    fig, ax = plt.subplots()
    # ax.set_facecolor('lightgrey')
    twinax = ax.twinx()

    ax.patch.set_visible(False)
    # Set axtwin's patch visible and colorise it in grey
    twinax.patch.set_visible(True)
    twinax.patch.set_facecolor('lightgrey')
    ax.set_zorder(twinax.get_zorder() + 1)

    ax.axhline(0, ls='--', c='grey', zorder=0)
    twinax.plot(total_pop, color='k', zorder=1)
    ax.plot(xk_1, color='b', zorder=2)
    ax.plot(yk_1, color='r', zorder=2)

    if rhodict2 is not None:
        ax.plot(xk_2, color='b', zorder=2, marker='^')
        ax.plot(yk_2, color='r', zorder=2, marker='^')

    ax.set_xticks(np.arange(len(years)))

    ax.set_xticklabels(years, rotation=90)
    twinax.set_ylabel('Total population (1e7 cap)')
    ax.set_ylabel('Kurtosis')
    ax.set_xlabel('Year')

    if rhodict2 is None:
        labels = ['N', 'x kurtosis', 'y kurtosis']
        lineparams = zip(labels, ['k', 'b', 'r'], ['', '', ''])
    else:
        if labels is None:
            labels = ['N', 'x kurtosis obs.', 'y kurtosis obs.',
                      'x kurtosis sim.', 'y kurtosis sim.']
        else:
            labels = ['N'] + labels
        lineparams = zip(labels, ['k', 'b', 'r', 'b', 'r'],
                         ['', '', '', '^', '^'])

    lines = [mlines.Line2D([], [], color=color, label=label, marker=marker)
             for label, color, marker in lineparams]

    leg = ax.legend(lines, labels, loc='center right', frameon=False,
                    prop=dict(size='small'))

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)

    plt.show()

def plot_smoothness(rhodict1, years, savepath=None):
    """
    Plots the smoothness of the mean x and y profiles of domain growth over time
    :param rhodict1: dict
        Dictionary of population arrays over years
    :return:
    """
    years = sorted(list(rhodict1.keys()))
    xs_1, ys_1 = xy_smoothness(arr_dict=rhodict1, years=years)

    fig, ax = plt.subplots()
    fig.patch.set_facecolor('w')
    ax.set_facecolor('lightgrey')

    ax.plot(xs_1, color='b', label='x profile')
    ax.plot(ys_1, color='r', label='y profile')

    ax.set_xticks(np.arange(len(years)))

    ax.set_xticklabels(years, rotation=90)
    ax.set_ylabel(r'Curve roughness $\int \bar \rho^{\prime\prime}$')
    ax.set_xlabel('Year')

    leg = ax.legend(prop=dict(size='small'))

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)

    plt.show()

def plot_pop_centre_suburbs(rho_dict,savepath=None):
    """
    Plots the total population across study period for the whole city, the city
    centre and a selected suburban area.

    :param rho_dict:
    :param title:
    :param savepath:
    :return:
    """
    years = sorted(list(rho_dict.keys()))

    suburbs = []
    centre = []

    for yr in years:
        cent = np.nansum(rho_dict[yr][30:37, 39:46])
        sub = np.nansum(rho_dict[yr][30:43, 17:33])
        centre.append(cent)
        suburbs.append(sub)

    fig, ax = plt.subplots()
    ax.plot(years, suburbs, color='r', lw=2, label='Suburban')
    ax.plot(years, centre, color='orange', lw=2, label='Central')

    ax.set_xlim(xmin=1831, xmax=2012)

    ax.set_ylabel('Total population')
    ax.set_xlabel('Year')

    ax.legend()

    if savepath is not None:
        plt.savefig(savepath)

    plt.show()

def plot_rho_centre_suburbs(rho_dict, title=None, savepath=None):
    """
    Plots the population density across study period for the city centre and a
    selected suburban area, as well as the total population of the whole city.
    :param rho_dict:
    :param title:
    :param savepath:
    :return:
    """
    years = sorted(list(rho_dict.keys()))

    cent_area = rho_dict[2011][30:37, 39:46].shape[0] * \
                rho_dict[2011][30:37, 39:46].shape[1]
    sub_area = rho_dict[2011][30:43, 17:33].shape[0] * \
               rho_dict[2011][30:43, 17:33].shape[1]

    suburbs = []
    centre = []
    total_pop = []

    for yr in years:
        cent = np.nansum(rho_dict[yr][30:37, 39:46]) / cent_area
        # sub = np.nansum(rho_dict[yr][16:51, 17:68]) - cent
        sub = np.nansum(rho_dict[yr][30:43, 17:33]) / sub_area
        centre.append(cent)
        suburbs.append(sub)
        total_pop.append(np.nansum(rho_dict[yr]))

    fig, ax = plt.subplots()
    twinax = ax.twinx()

    ax.patch.set_visible(False)
    # Set axtwin's patch visible and colorise it in grey
    twinax.patch.set_visible(True)
    # twinax.patch.set_facecolor('lightgrey')
    ax.set_zorder(twinax.get_zorder() + 1)

    twinax.plot(years, total_pop, color='slateblue')
    ax.plot(years, suburbs, color='r', lw=2, label='Suburban')
    ax.plot(years, centre, color='orange', lw=2, label='Central')

    ax.set_xlim(xmin=1831, xmax=2012)

    ax.set_ylabel('Population density (cap km$^{-2}$)')
    twinax.set_ylabel('Total population')

    ax.set_xlabel('Year')

    labels = ['Cental', 'Suburban', 'Total']
    lineparams = zip(labels, ['orange', 'r', 'slateblue'])

    lines = [mlines.Line2D([], [], color=color, label=label)
             for label, color in lineparams]

    leg = ax.legend(lines, labels, loc='center left')

    twinax.yaxis.label.set_color('slateblue')
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)

    plt.show()

def plot_errors(rmse, mae, mbe, years, units, savepath=None):
    """
    Plot the RMSE, MAE and MBE for years.

    :param rmse: numpy.ndarray
        Root mean square error values
    :param mae: numpy.ndarray
        Mean absolute error values
    :param mbe: numpy.ndarray
        Mean bias error values
    :param years: numpy.ndarray
        Years for which error statistics are calculated
    :param units: str
        Units of the variable investigated
    :param savepath: str
        Save path
    :return:
    """
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)

    for ax in (ax1, ax2, ax3):
        ax.axhline(0, ls='--', color='grey', zorder=0)

    # plot rmse
    ax1.scatter(years, rmse, color='red', zorder=1)
    ax1.set_ylabel(f'RMSE {units}')
    # plot mae
    ax2.scatter(years, mae, color='green', zorder=1)
    ax2.set_ylabel(f'MAE {units}')
    # plot mbe
    ax3.scatter(years, mbe, color='blue', zorder=1)
    ax3.set_ylabel(f'MBE {units}')

    if type(years) not in [numpy.ndarray, list]:
        years = list(years)

    ax3.set_xticks(years[::2])

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
    else:
        plt.show()

def plot_error_comparison(errors1, errors2, years, units, name1, name2,
                          savepath=None):
    """
    Plots comparisons of errors between two runs.

    :param errors1: dict
        Errors of a run. Keys are error type (RMSE, MAE, MBE), values are error
        values throughout the years (numpy.ndarray)
    :param errors2: dict
        As errors1
    :param years: numpy.ndarray
        Years for which error statistics are calculated
    :param units: str
        Units of the variable investigated
    :param savepath: str
        Save path
    :return:
    """

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)

    for ax in (ax1, ax2, ax3):
        ax.axhline(0, ls='--', color='grey', zorder=0)

    # plot rmse
    ax1.scatter(years, errors1['rmse'], color='red', label=name1, zorder=1)
    ax1.scatter(years, errors2['rmse'], color='blue', label=name2, zorder=1)

    ax1.set_ylabel(f'RMSE {units}')
    ax1.legend()
    # plot mae
    ax2.scatter(years, errors1['mae'], color='red', zorder=1)
    ax2.scatter(years, errors2['mae'], color='blue', zorder=1)

    ax2.set_ylabel(f'MAE {units}')
    # plot mbe
    ax3.scatter(years, errors1['mbe'], color='red', zorder=1)
    ax3.scatter(years, errors2['mbe'], color='blue', zorder=1)

    ax3.set_ylabel(f'MBE {units}')
    ax3.set_xlabel('Years')
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
    else:
        plt.show()

def plot_accuracy_scatter(obs_data, sim_data, varobs, varsim, year=None,
                          borders=None, savepath=None, ax=None, log=False,
                          cmap='plasma', axlim=None, r2_scores=False,
                          centre=(0,0), nan_val=0.0):
    """
    Scatter plot of observed vs. simulated values including coloured density of
    points and 1:1 line.

    :param obs_data: numpy.ndarray
    :param sim_data: numpy.ndarray
    :param varobs: str
    :param varsim: str
    :param borders: list, tuple
    :param savepath: str
    :param centre: tuple
        Centre point for radial profile if using r2 scores
    :param nan_val: float
        Fill values for NaNs in radial profile if using r2 scores
    :return:
    """
    if borders is not None:
        obs_data = trim_array(obs_data, borders)
        sim_data = trim_array(sim_data, borders)
    else:
        borders = (0, 0)
    if r2_scores:
        obs_rad = radial_profile(obs_data, centre=centre, nan_val=nan_val)[0]
        sim_rad = radial_profile(sim_data, centre=centre, nan_val=nan_val)[0]
        obs_cent = obs_data[15-borders[0]:51-borders[0],
                            25-borders[1]:59-borders[1]]
        sim_cent = sim_data[15-borders[0]:51-borders[0],
                            25-borders[1]:59-borders[1]]

    obs_data = obs_data.flatten()
    sim_data = sim_data.flatten()

    null_filt = ~np.isnan(obs_data) * ~np.isnan(sim_data)
    x = obs_data[null_filt]
    y = sim_data[null_filt]
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    max_val = np.max(np.array([obs_data, sim_data]))
    if r2_scores:
        r2_cent = np.round(r2_score(y_true=obs_cent.flatten(),
                                    y_pred=sim_cent.flatten()), decimals=2)
        r2_rad = np.round(r2_score(y_true=obs_rad, y_pred=sim_rad), decimals=2)

        mask = ~np.isnan(obs_data) & ~np.isnan(sim_data)

        r2_arr = np.round(r2_score(y_true=obs_data[mask],
                                   y_pred=sim_data[mask]), decimals=2)
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        fig.patch.set_facecolor('w')
        tght_lyt = True
        var_ax = False
    else:
        tght_lyt = False
        var_ax = True
    ax.scatter(x, y, c=z, cmap=cmap)
    ax.plot([0, max_val], [0, max_val], ls='--', color='g', lw=2)
    ax.set_xlabel(varobs)
    ax.set_ylabel(varsim)
    if year is not None:
        ax.set_title(year)
    if log:
        ax.set_yscale('log')
        ax.set_xscale('log')

    if axlim:
        ax.set_xlim(axlim)
        ax.set_ylim(axlim)
    else:
        ax.set_ylim(ymin=10 ** 0, ymax=np.max([np.max(x), np.max(y)]))
        ax.set_xlim(xmin=10 ** 0, xmax=np.max([np.max(x), np.max(y)]))
    # ax.tick_params(axis='both', which='both')
    if r2_scores:
        ax.text(0.45, 0.15,
                f"Total R$^{{{2}}}$ = {r2_arr}\n"
                f"Central R$^{{{2}}}$ = {r2_cent}\n"
                f"Radial R$^{{{2}}}$ = {r2_rad}", transform=ax.transAxes,
                fontsize=10)
    if tght_lyt:
        plt.tight_layout()
    ax.set_aspect('equal')
    if not var_ax:
        if savepath is not None:
            plt.savefig(savepath, dpi=300)
        else:
            plt.show()
        plt.close()
    else:
        return(ax)

def write_scientific(x):
    int_pot = np.array([x * 10 ** m for m in np.arange(6)])

    ints = np.where(int_pot.astype(int) == int_pot)[0]

    return(f'{int(int_pot[ints[0]])}e-{ints[0]}')


def plot_sim_obs_network(rho, sim_ntwrk, obs_ntwrk, rho_cmap, ai_cmap,
                         savepath=None):
    """
    Plot the
    :param rho:
    :param sim_ntwrk:
    :param obs_ntwrk:
    :return:
    """
    for ntwrk in (sim_ntwrk, obs_ntwrk):
        rn = list(ntwrk.nodes)[0]
        if 'pos' in ntwrk.nodes[rn].keys():
            pass
        else:
            pos_dict = \
                {n: (ntwrk.nodes[n]['lon'] - 490,
                     ntwrk.nodes[n]['lat'] - 148)
                 for n in ntwrk.nodes}
            nx.set_node_attributes(ntwrk, pos_dict, name='pos')

    a_i_sim = list(nx.get_node_attributes(sim_ntwrk, 'a_i_inv').values())
    a_i_obs = list(nx.get_node_attributes(obs_ntwrk, 'a_i_inv').values())
    a_i_min = np.nanmin(a_i_sim + a_i_obs)
    a_i_max = np.nanmax(a_i_sim + a_i_obs)

    fig, (ax1, ax2) = plt.subplots(ncols=2)
    fig.patch.set_facecolor('w')
    city = ax1.pcolor(rho / 1000., cmap=rho_cmap)

    divider = make_axes_locatable(ax1)
    cax_rho = divider.append_axes('bottom', size='5%', pad=0.05)
    cbarrho = fig.colorbar(city, cax=cax_rho, orientation='horizontal',
                           label='Population (1000 cap km$^{-2}$)')
    cbarrho.ax.set_xlabel(r'Population (1000 cap km$^{-2}$)',
                          fontsize='x-small')
    cbarrho.ax.tick_params(labelsize='x-small')

    nx.draw_networkx(
        sim_ntwrk, with_labels=False, ax=ax1, node_size=3,
        pos=nx.get_node_attributes(sim_ntwrk, 'pos'),
        cmap=ai_cmap, edge_color='grey', width=0.5,
        node_color=a_i_sim, vmin=a_i_min, vmax=a_i_max)

    ax1.set_title(f"$N_{{s}}$ = {len(sim_ntwrk.nodes)}", fontsize=8)
    city2 = ax2.pcolor(rho / 1000., cmap=rho_cmap)
    ax2.set_title(f"$N_{{s}}$ = {len(obs_ntwrk.nodes)}", fontsize=8)

    # nx.draw(G=ntwrk, node_size=5, node_color='r',
    #         pos=ntwrk_pos, ax=ax2)
    nx.draw_networkx(
        obs_ntwrk, with_labels=False, ax=ax2, node_size=3,
        pos=nx.get_node_attributes(obs_ntwrk, 'pos'),
        cmap=ai_cmap, edge_color='grey', width=0.5,
        node_color=a_i_obs, vmin=a_i_min, max=a_i_max)
    sm = plt.cm.ScalarMappable\
        (cmap='Greens', norm=plt.Normalize(vmin=a_i_min, vmax=a_i_max))
    divider = make_axes_locatable(ax2)
    cax_ai = divider.append_axes('bottom', size='5%', pad=0.05)
    cbarai = fig.colorbar(sm, cax=cax_ai, orientation='horizontal',
                          label='$A_{i}$')
    cbarai.ax.set_xlabel(r'$A_{i}$', fontsize='x-small')
    cbarai.ax.tick_params(labelsize='x-small')
    # cbar = ax2.figure.colorbar(sm, orientation='horizontal',
    #                            label='$A_{i}$')
    # plt.plot(q_all)
    for ax in (ax1, ax2):
        ax.set_xlim(0, 84)
        ax.set_ylim(0, 66)
        ax.set_aspect('equal')
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=300)
    plt.show()

def plot_multiple_sim_network(
        rho, ntwrk, rho_cmap, av_cmap, years, borders, rho_vmin=None,
        savepath=None):
    """
    Plot the
    :param rho:
    :param ntwrk:
    :param obs_ntwrk:
    :return:
    """
    av_cmap = cm.get_cmap(av_cmap)

    rho_cmap = cm.get_cmap(rho_cmap)
    if rho_vmin is not None:
        rho_cmap.set_under('ghostwhite')

    a_v_sim, a_v_min, a_v_max = {}, {}, {}
    rho_min, rho_max = {}, {}
    for yr in years:
        ntwrk_yr = ntwrk[yr]
        rn = list(ntwrk_yr.nodes)[0]
        if 'pos' in ntwrk_yr.nodes[rn].keys():
            pass
        else:
            pos_dict = \
                {n: (ntwrk_yr.nodes[n]['lon'] - 490,
                     ntwrk_yr.nodes[n]['lat'] - 148)
                 for n in ntwrk_yr.nodes}
            nx.set_node_attributes(ntwrk_yr, pos_dict, name='pos')

        a_v_sim[yr] = list(
            nx.get_node_attributes(ntwrk[yr], 'a_i_inv').values())
        a_v_min[yr] = np.nanmin(a_v_sim[yr])
        a_v_max[yr] = np.nanmax(a_v_sim[yr])

        if rho_vmin is None:
            rho_min[yr] = np.nanmin(trim_array(rho[yr], borders))
        else:
            rho_min[yr] = rho_vmin
        rho_max[yr] = np.nanmax(trim_array(rho[yr], borders))

    min_av = np.nanmin(list(a_v_min.values()))
    max_av = np.nanmax(list(a_v_max.values()))
    min_rho = np.nanmin(list(rho_min.values())) / 1000
    max_rho = np.nanmax(list(rho_max.values())) / 1000

    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(7.5, 6.75))
    fig.patch.set_facecolor('w')
    axes = axes.flatten()
    for i, yr in enumerate(years):
        city = axes[i].pcolor(rho[yr] / 1000.,
                              cmap=rho_cmap, vmin=min_rho, vmax=max_rho)

        nx.draw_networkx(
            ntwrk[yr], with_labels=False, ax=axes[i], node_size=2,
            pos=nx.get_node_attributes(ntwrk[yr], 'pos'), cmap=av_cmap,
            edge_color='grey', width=0.5, node_color=a_v_sim[yr], vmin=min_av,
            vmax=max_av)

        axes[i].set_title(rf"$\bf{{" + str(yr) + "}}$ " +
                          rf"$N_{{v}}$ = {len(ntwrk[yr].nodes)}", fontsize=10)

        axes[i].tick_params(left=True, bottom=True, labelleft=True,
                            labelbottom=True, labelsize=7)

    sm = plt.cm.ScalarMappable \
        (cmap=rho_cmap, norm=plt.Normalize(vmin=min_rho, vmax=max_rho))
    divider = make_axes_locatable(axes[0])
    cax_rho = divider.append_axes('bottom', size='5%', pad=0.25)
    if rho_vmin is not None:
        cbarrho = fig.colorbar(city, cax=cax_rho, orientation='horizontal',
                               label=r'$\rho$ (1000 cap km$^{-2}$)',
                               extend='min')
        cbarrho.ax.xaxis.set_ticks([rho_vmin/1000] + list(cbarrho.get_ticks()))
    else:
        cbarrho = fig.colorbar(city, cax=cax_rho, orientation='horizontal',
                               label=r'$\rho$ (1000 cap km$^{-2}$)')
    cbarrho.ax.set_xlabel(r'$\rho$ (1000 cap km$^{-2}$)',
                          fontsize='small')
    cbarrho.ax.tick_params(labelsize='small')

    sm = plt.cm.ScalarMappable \
        (cmap=av_cmap, norm=plt.Normalize(vmin=min_av, vmax=max_av))
    divider = make_axes_locatable(axes[1])
    cax_ai = divider.append_axes('bottom', size='5%', pad=0.25)
    cbarai = fig.colorbar(sm, cax=cax_ai, orientation='horizontal',
                          label='$A_{v}$')
    cbarai.ax.set_xlabel(r'$A_{v}$', fontsize='small')
    cbarai.ax.tick_params(labelsize='small')
    # cbar = ax2.figure.colorbar(sm, orientation='horizontal',
    #                            label='$A_{i}$')
    # plt.plot(q_all)
    for ax in axes:
        # ax.set_xlim(0 ,84 )
        # ax.set_ylim(0 ,66)
        ax.set_xlim(0 + 6, 84 - 6)
        ax.set_ylim(0 + 6, 66 - 6)
        ax.set_aspect('equal')
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=300)
    plt.show()

def plot_multiple_obssim_network(
        rho, ntwrk_sim, ntwrk_obs, rho_cmap, av_cmap, years, borders, xv=None,
        yv=None, rho_vmin=None, savepath=None, cbar_savepath=None):
    """
    Plot the
    :param rho:
    :param ntwrk_sim:
    :param obs_ntwrk:
    :return:
    """

    av_cmap = cm.get_cmap(av_cmap)
    rho_cmap = cm.get_cmap(rho_cmap)
    if rho_vmin is not None:
        rho_cmap.set_under('whitesmoke')

    a_v_sim, a_v_min_sim, a_v_max_sim = {}, {}, {}
    a_v_obs, a_v_min_obs, a_v_max_obs = {}, {}, {}
    min_av, max_av = {}, {}
    rho_min, rho_max = {}, {}
    for yr in years:
        sim_ntwrk_yr = ntwrk_sim[yr]
        rn = list(sim_ntwrk_yr.nodes)[0]
        if xv is not None and yv is not None:
            pos_dict = \
                {n: (sim_ntwrk_yr.nodes[n]['lon'] - abs(xv.min()),
                     sim_ntwrk_yr.nodes[n]['lat'] - abs(yv.min()))
                 for n in sim_ntwrk_yr.nodes}
            nx.set_node_attributes(sim_ntwrk_yr, pos_dict, name='pos')
        else:
            pos_dict = \
                {n: (sim_ntwrk_yr.nodes[n]['lon'] - 490,
                     sim_ntwrk_yr.nodes[n]['lat'] - 148)
                 for n in sim_ntwrk_yr.nodes}
            nx.set_node_attributes(sim_ntwrk_yr, pos_dict, name='pos')

        a_v_sim[yr] = list(
            nx.get_node_attributes(ntwrk_sim[yr], 'a_i_inv').values())
        a_v_min_sim[yr] = np.nanmin(a_v_sim[yr])
        a_v_max_sim[yr] = np.nanmax(a_v_sim[yr])

        obs_ntwrk_yr = ntwrk_obs[yr]
        rn = list(obs_ntwrk_yr.nodes)[0]
        if xv is not None and yv is not None:
            pos_dict = \
                {n: (obs_ntwrk_yr.nodes[n]['lon'] - 490 - abs(xv.min()),
                     obs_ntwrk_yr.nodes[n]['lat'] - 148 - abs(yv.min()))
                 for n in obs_ntwrk_yr.nodes}
            nx.set_node_attributes(obs_ntwrk_yr, pos_dict, name='pos')
        else:
            pos_dict = \
                {n: (obs_ntwrk_yr.nodes[n]['lon'] - 490,
                     obs_ntwrk_yr.nodes[n]['lat'] - 148)
                 for n in obs_ntwrk_yr.nodes}
            nx.set_node_attributes(obs_ntwrk_yr, pos_dict, name='pos')

        a_v_obs[yr] = list(
            nx.get_node_attributes(ntwrk_obs[yr], 'a_i_inv').values())
        a_v_min_obs[yr] = np.nanmin(a_v_obs[yr])
        a_v_max_obs[yr] = np.nanmax(a_v_obs[yr])

        min_av[yr] = np.nanmin([a_v_min_sim[yr], a_v_min_obs[yr]])
        max_av[yr] = np.nanmax([a_v_max_sim[yr], a_v_max_obs[yr]])

        if rho_vmin is None:
            rho_min[yr] = np.nanmin(trim_array(rho[yr], borders))
        else:
            rho_min[yr] = rho_vmin
        rho_max[yr] = np.nanmax(trim_array(rho[yr], borders))

    # min_av = np.nanmin(list(a_v_min.values()))
    # max_av = np.nanmax(list(a_v_max.values()))
    min_av = np.nanmin(list(a_v_min_sim.values()) + list(a_v_min_obs.values()))
    max_av = np.nanmax(list(a_v_max_sim.values()) + list(a_v_max_obs.values()))
    min_rho = np.nanmin(list(rho_min.values())) / 1000
    max_rho = np.nanmax(list(rho_max.values())) / 1000

    # WITH COLOURBARS
    fig, axes = plt.subplots(ncols=2, nrows=len(years), figsize=(6, 10.5))
    # # WITHOUT COLOURBARS
    # fig, axes = plt.subplots(ncols=2, nrows=len(years), figsize=(7.5, 12.5))
    fig.patch.set_facecolor('w')
    # axes = axes.flatten()
    for i, yr in enumerate(years):
        city = axes[i, 0].pcolor(rho[yr] / 1000.,
                                 cmap=rho_cmap, vmin=min_rho, vmax=max_rho)
        nx.draw_networkx(
            ntwrk_sim[yr], with_labels=False, ax=axes[i, 0], node_size=2,
            pos=nx.get_node_attributes(ntwrk_sim[yr], 'pos'), cmap=av_cmap,
            edge_color='grey', width=0.5, node_color=a_v_sim[yr], vmin=min_av,
            vmax=max_av)
        axes[i, 0].set_title(rf"$\bf{{" + str(yr) + "}}$: " +
                             rf"$N_{{v}}$ = {len(ntwrk_sim[yr].nodes)}",
                             fontsize=10)
        axes[i, 0].tick_params(left=True, bottom=True, labelleft=True,
                               labelbottom=True, labelsize=7)

        city = axes[i, 1].pcolor(rho[yr] / 1000.,
                                 cmap=rho_cmap, vmin=min_rho, vmax=max_rho)
        nx.draw_networkx(
            ntwrk_obs[yr], with_labels=False, ax=axes[i, 1], node_size=2,
            pos=nx.get_node_attributes(ntwrk_obs[yr], 'pos'), cmap=av_cmap,
            edge_color='grey', width=0.5, node_color=a_v_obs[yr], vmin=min_av,
            vmax=max_av)
        axes[i,1].set_title(rf"$\bf{{" + str(yr) + "}}$: " +
                            rf"$N_{{v}}$ = {len(ntwrk_obs[yr].nodes)}",
                            fontsize=10)
        axes[i,1].tick_params(left=True, bottom=True, labelleft=True,
                            labelbottom=True, labelsize=7)

        axes[i,0].set_ylabel('$y$ (km)', fontsize=8)

    axes[0, 0].set_title(rf"$\it{{Simulated}}$ " + "\n" +
                         rf"$\bf{{" + str(years[0]) + "}}$: " +
                         rf"$N_{{v}}$ = {len(ntwrk_sim[years[0]].nodes)}",
                         fontsize=10)
    axes[0, 1].set_title(rf"$\it{{Observed}}$ " + "\n" +
                         rf"$\bf{{" + str(years[0]) + "}}$: " +
                         rf"$N_{{v}}$ = {len(ntwrk_obs[years[0]].nodes)}",
                         fontsize=10)

    axes[i,0].set_xlabel('$x$ (km)', fontsize=8)
    axes[i,1].set_xlabel('$x$ (km)', fontsize=8)

    for ax in axes.flatten():
        if xv is not None and yv is not None:
            ax.set_xlim(xv.min() + 6, xv.max() - 6)
            ax.set_ylim(yv.min() + 6, yv.max() - 6)
        else:
            ax.set_xlim(0 + 6, 84 - 6)
            ax.set_ylim(0 + 6, 66 - 6)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=8)
    # plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=300)

    cbfig = plt.figure(figsize=(fig.get_size_inches()[0], 1.5))
    cbfig.patch.set_facecolor('w')
    subfigs = cbfig.subfigures(1, 2)
    ax1 = subfigs[0].add_axes([0.05, 0.80, 0.9, 0.1])
    cbfig.patch.set_facecolor('w')
    cb1 = mpl.colorbar.ColorbarBase(ax1, orientation='horizontal',
                                    cmap=rho_cmap,
                                    norm=plt.Normalize(vmin=min_rho,
                                                       vmax=max_rho),
                                    extend='min')

    cb1.ax.tick_params(labelsize='small')
    cb1.ax.set_xlabel(r'$\rho$ (1000 cap km$^{-2}$)', fontsize='small')

    ax2 = subfigs[1].add_axes([0.05, 0.80, 0.9, 0.1])
    cb2 = mpl.colorbar.ColorbarBase(ax2, cmap=av_cmap, orientation='horizontal',
                                    norm=plt.Normalize(vmin=min_av,
                                                       vmax=max_av))
    cb2.ax.tick_params(labelsize='small')
    cb2.ax.set_xlabel(r'$A_{v}$', fontsize='small')

    if cbar_savepath is not None:
        plt.savefig(cbar_savepath, dpi=300)

    plt.show()

def plot_multiple_obssim_network_rho(rho_sim, rho_obs, ntwrk_sim, ntwrk_obs, \
                                     rho_cmap, av_cmap, years, borders,
                                     xv=None, yv=None, rho_vmin=None,
                                     savepath=None, cbar_savepath=None):
    """
    Plot the
    :param rho:
    :param ntwrk_sim:
    :param obs_ntwrk:
    :return:
    """

    av_cmap = cm.get_cmap(av_cmap)
    rho_cmap = cm.get_cmap(rho_cmap)
    if rho_vmin is not None:
        rho_cmap.set_under('whitesmoke')

    a_v_sim, a_v_min_sim, a_v_max_sim = {}, {}, {}
    a_v_obs, a_v_min_obs, a_v_max_obs = {}, {}, {}
    min_av, max_av = {}, {}
    rho_min, rho_max = {}, {}

    if xv is not None and yv is not None:
        pos_dict_sim = \
            {n: (ntwrk_sim[years[-1]].nodes[n]['lon'] - 490 - abs(xv.min()),
                ntwrk_sim[years[-1]].nodes[n]['lat'] - 148 - abs(yv.min()))
            for n in ntwrk_sim[years[-1]].nodes}
        pos_dict_obs = \
            {n: (ntwrk_obs[years[-1]].nodes[n]['lon'] - 490 - abs(xv.min()),
                ntwrk_obs[years[-1]].nodes[n]['lat'] - 148 - abs(yv.min()))
            for n in ntwrk_obs[years[-1]].nodes}
    else:
        pos_dict_sim = \
            {n: (ntwrk_sim[years[-1]].nodes[n]['lon'] - 490,
                ntwrk_sim[years[-1]].nodes[n]['lat'] - 148)
            for n in ntwrk_sim[years[-1]].nodes}
        pos_dict_obs = \
            {n: (ntwrk_obs[years[-1]].nodes[n]['lon'] - 490,
                ntwrk_obs[years[-1]].nodes[n]['lat'] - 148)
            for n in ntwrk_obs[years[-1]].nodes}

    for yr in years:
        sim_ntwrk_yr = ntwrk_sim[yr]
        rn = list(sim_ntwrk_yr.nodes)[0]

        a_v_sim[yr] = list(
            nx.get_node_attributes(ntwrk_sim[yr], 'a_i_inv').values())
        a_v_min_sim[yr] = np.nanmin(a_v_sim[yr])
        a_v_max_sim[yr] = np.nanmax(a_v_sim[yr])

        obs_ntwrk_yr = ntwrk_obs[yr]
        rn = list(obs_ntwrk_yr.nodes)[0]
        # if 'pos' in obs_ntwrk_yr.nodes[rn].keys():
        #     pass

        a_v_obs[yr] = list(
            nx.get_node_attributes(ntwrk_obs[yr], 'a_i_inv').values())
        a_v_min_obs[yr] = np.nanmin(a_v_obs[yr])
        a_v_max_obs[yr] = np.nanmax(a_v_obs[yr])

        min_av[yr] = np.nanmin([a_v_min_sim[yr], a_v_min_obs[yr]])
        max_av[yr] = np.nanmax([a_v_max_sim[yr], a_v_max_obs[yr]])

        if rho_vmin is None:
            rho_min[yr] = np.nanmin([trim_array(rho_sim[yr], borders).flatten(),
                                     trim_array(rho_obs[yr],
                                                borders).flatten()])
        else:
            rho_min[yr] = rho_vmin
        rho_max[yr] = np.nanmax([trim_array(rho_sim[yr], borders).flatten(),
                                 trim_array(rho_obs[yr], borders).flatten()])

    # min_av = np.nanmin(list(a_v_min.values()))
    # max_av = np.nanmax(list(a_v_max.values()))
    min_av = np.nanmin(list(a_v_min_sim.values()) + list(a_v_min_obs.values()))
    max_av = np.nanmax(list(a_v_max_sim.values()) + list(a_v_max_obs.values()))
    min_rho = np.nanmin(list(rho_min.values())) / 1000
    max_rho = np.nanmax(list(rho_max.values())) / 1000

    # WITH COLOURBARS
    fig, axes = plt.subplots(ncols=2, nrows=len(years), figsize=(6, 10))
    # # WITHOUT COLOURBARS
    # fig, axes = plt.subplots(ncols=2, nrows=len(years), figsize=(7.5, 12.5))
    fig.patch.set_facecolor('w')
    # axes = axes.flatten()
    for i, yr in enumerate(years):

        if xv is not None and yv is not None:
            city = axes[i, 0].pcolor(xv, yv, rho_sim[yr] / 1000., cmap=rho_cmap,
                                     vmin=min_rho, vmax=max_rho)
        else:
            city = axes[i, 0].pcolor(rho_sim[yr] / 1000., cmap=rho_cmap,
                                     vmin=min_rho, vmax=max_rho)
        nx.draw_networkx(
            ntwrk_sim[yr], with_labels=False, ax=axes[i, 0], node_size=2,
            pos=pos_dict_sim,  # nx.get_node_attributes(ntwrk_sim[yr], 'pos'),
            # cmap=av_cmap,
            edge_color='grey', width=0.5, node_color=a_v_sim[yr], vmin=min_av,
            vmax=max_av, cmap=av_cmap)
        axes[i, 0].set_title(rf"$\bf{{" + str(yr) + "}}$: " +
                             rf"$N_{{v}}$ = {len(ntwrk_sim[yr].nodes)}",
                             fontsize=8)
        axes[i, 0].tick_params(left=True, bottom=True, labelleft=True,
                               labelbottom=True, labelsize=7)

        axes[i, 0].set_ylabel('$y$ (km)', fontsize=8)

        if xv is not None and yv is not None:
            city = axes[i, 1].pcolor(xv, yv, rho_obs[yr] / 1000.,
                                    cmap=rho_cmap, vmin=min_rho, vmax=max_rho)
        else:
            city = axes[i, 1].pcolor(rho_obs[yr] / 1000.,
                                    cmap=rho_cmap, vmin=min_rho, vmax=max_rho)

        nx.draw_networkx(
            ntwrk_obs[yr], with_labels=False, ax=axes[i, 1], node_size=2,
            pos=pos_dict_obs,  # nx.get_node_attributes(ntwrk_obs[yr], 'pos'),
            # cmap=av_cmap,
            edge_color='grey', width=0.5, node_color=a_v_obs[yr], vmin=min_av,
            vmax=max_av, cmap=av_cmap)
        axes[i, 1].set_title(rf"$\bf{{" + str(yr) + "}}$: " +
                             rf"$N_{{v}}$ = {len(ntwrk_obs[yr].nodes)}",
                             fontsize=8)
        axes[i, 1].tick_params(left=True, bottom=True, labelleft=True,
                               labelbottom=True, labelsize=7)

    for ax in axes.flatten():
        if xv is not None and yv is not None:
            ax.set_xlim(xv.min() + 6, xv.max() - 6)
            ax.set_ylim(yv.min() + 6, yv.max() - 6)
        else:
            ax.set_xlim(0 + 6, 84 - 6)
            ax.set_ylim(0 + 6, 66 - 6)
        ax.set_aspect('equal')

    axes[i,0].set_xlabel('$x$ (km)', fontsize=8)
    axes[i,1].set_xlabel('$x$ (km)', fontsize=8)

    # plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=300)
    plt.show()

    cbfig = plt.figure(figsize=(fig.get_size_inches()[0], 1.5))
    cbfig.patch.set_facecolor('w')
    subfigs = cbfig.subfigures(1, 2)
    ax1 = subfigs[0].add_axes([0.05, 0.80, 0.9, 0.1])
    cbfig.patch.set_facecolor('w')
    cb1 = mpl.colorbar.ColorbarBase(ax1, orientation='horizontal',
                                    cmap=rho_cmap,
                                    norm=plt.Normalize(vmin=min_rho,
                                                       vmax=max_rho),
                                    )
    cb1.ax.tick_params(labelsize='small')
    cb1.ax.set_xlabel(r'$\rho$ (1000 cap km$^{-2}$)', fontsize='small')

    ax2 = subfigs[1].add_axes([0.05, 0.80, 0.9, 0.1])
    cb2 = mpl.colorbar.ColorbarBase(ax2, cmap=av_cmap, orientation='horizontal',
                                    norm=plt.Normalize(vmin=min_av,
                                                       vmax=max_av))
    cb2.ax.tick_params(labelsize='small')
    cb2.ax.set_xlabel(r'$A_{v}$', fontsize='small')

    if cbar_savepath is not None:
        plt.savefig(cbar_savepath, dpi=300)

def plot_multiple_obssim_network_rho_sydney(rho_sim, rho_obs, ntwrk_sim,
    ntwrk_obs, xvyv, rho_cmap, av_cmap, years, borders, rho_vmin=None,
    savepath=None, cbar_savepath=None):
    """
    Plot the
    :param rho:
    :param ntwrk_sim:
    :param obs_ntwrk:
    :return:
    """

    av_cmap = cm.get_cmap(av_cmap)
    rho_cmap = cm.get_cmap(rho_cmap)
    if rho_vmin is not None:
        rho_cmap.set_under('whitesmoke')

    a_v_sim, a_v_min_sim, a_v_max_sim = {}, {}, {}
    a_v_obs, a_v_min_obs, a_v_max_obs = {}, {}, {}
    min_av, max_av = {}, {}
    rho_min, rho_max = {}, {}

    for yr in years:
        sim_ntwrk_yr = ntwrk_sim[yr]
        rn = list(sim_ntwrk_yr.nodes)[0]

        a_v_sim[yr] = list(
            nx.get_node_attributes(ntwrk_sim[yr], 'a_i_inv').values())
        a_v_min_sim[yr] = np.nanmin(a_v_sim[yr])
        a_v_max_sim[yr] = np.nanmax(a_v_sim[yr])

        obs_ntwrk_yr = ntwrk_obs[yr]
        rn = list(obs_ntwrk_yr.nodes)[0]
        # if 'pos' in obs_ntwrk_yr.nodes[rn].keys():
        #     pass

        a_v_obs[yr] = list(
            nx.get_node_attributes(ntwrk_obs[yr], 'a_i_inv').values())
        a_v_min_obs[yr] = np.nanmin(a_v_obs[yr])
        a_v_max_obs[yr] = np.nanmax(a_v_obs[yr])

        min_av[yr] = np.nanmin([a_v_min_sim[yr], a_v_min_obs[yr]])
        max_av[yr] = np.nanmax([a_v_max_sim[yr], a_v_max_obs[yr]])

        if rho_vmin is None:
            rho_min[yr] = np.nanmin([trim_array(rho_sim[yr], borders).flatten(),
                                     trim_array(rho_obs[yr], borders).flatten()])
        else:
            rho_min[yr] = rho_vmin
        rho_max[yr] = np.nanmax([trim_array(rho_sim[yr], borders).flatten(),
                                 trim_array(rho_obs[yr], borders).flatten()])

    # min_av = np.nanmin(list(a_v_min.values()))
    # max_av = np.nanmax(list(a_v_max.values()))
    min_av = np.nanmin(list(a_v_min_sim.values()) + list(a_v_min_obs.values()))
    max_av = np.nanmax(list(a_v_max_sim.values()) + list(a_v_max_obs.values()))
    min_rho = np.nanmin(list(rho_min.values())) / 1000
    max_rho = np.nanmax(list(rho_max.values())) / 1000

    # WITH COLOURBARS
    fig, axes = plt.subplots(ncols=2, nrows=len(years), figsize=(6, 10))
    # # WITHOUT COLOURBARS
    # fig, axes = plt.subplots(ncols=2, nrows=len(years), figsize=(7.5, 12.5))
    fig.patch.set_facecolor('w')
    # axes = axes.flatten()
    for i, yr in enumerate(years):
        city = axes[i, 0].pcolor(xvyv['xv'], xvyv['yv'], rho_sim[yr] / 1000.,
                                 cmap=rho_cmap, vmin=min_rho, vmax=max_rho)

        pos_dict_sim = \
            {n: (ntwrk_sim[yr].nodes[n]['lon']/1000,
                 ntwrk_sim[yr].nodes[n]['lat']/1000)
             for n in ntwrk_sim[yr].nodes}

        nx.draw_networkx(
            ntwrk_sim[yr], with_labels=False, ax=axes[i, 0], node_size=2,
            pos=pos_dict_sim, #nx.get_node_attributes(ntwrk_sim[yr], 'pos'),
            edge_color='grey', width=0.5, node_color=a_v_sim[yr], vmin=min_av,
            vmax=max_av, cmap=av_cmap)
        axes[i, 0].set_title(rf"$\bf{{" + str(yr) + "}}$: " +
                             rf"$N_{{v}}$ = {len(ntwrk_sim[yr].nodes)}",
                             fontsize=8)
        axes[i, 0].tick_params(left=True, bottom=True, labelleft=True,
                               labelbottom=True, labelsize=7)

        city = axes[i, 1].pcolor(xvyv['xv'], xvyv['yv'], rho_obs[yr] / 1000.,
                                 cmap=rho_cmap, vmin=min_rho, vmax=max_rho)
        pos_dict_obs = \
            {n: (ntwrk_obs[yr].nodes[n]['lon'] / 1000,
                 ntwrk_obs[yr].nodes[n]['lat'] / 1000)
             for n in ntwrk_obs[yr].nodes}
        nx.draw_networkx(
            ntwrk_obs[yr], with_labels=False, ax=axes[i, 1], node_size=2,
            pos=pos_dict_obs, #nx.get_node_attributes(ntwrk_obs[yr], 'pos'),
        # cmap=av_cmap,
            edge_color='grey', width=0.5, node_color=a_v_obs[yr], vmin=min_av,
            vmax=max_av, cmap=av_cmap)
        axes[i, 1].set_title(rf"$\bf{{" + str(yr) + "}}$: " +
                             rf"$N_{{v}}$ = {len(ntwrk_obs[yr].nodes)}",
                             fontsize=8)
        axes[i, 1].tick_params(left=True, bottom=True, labelleft=True,
                               labelbottom=True, labelsize=7)

    # sm = plt.cm.ScalarMappable \
    #     (cmap=rho_cmap, norm=plt.Normalize(vmin=min_rho, vmax=max_rho))
    # divider = make_axes_locatable(axes[0, 0])
    # cax_rho = divider.append_axes('bottom', size='5%', pad=0.25)
    # if rho_vmin is not None:
    #     cbarrho = fig.colorbar(city, cax=cax_rho, orientation='horizontal',
    #                            label=r'$\rho$ (1000 cap km$^{-2}$)',
    #                            extend='min')
    #     cbarrho.ax.xaxis.set_ticks([rho_vmin/1000] + list(cbarrho.get_ticks()))
    # else:
    #     cbarrho = fig.colorbar(city, cax=cax_rho, orientation='horizontal',
    #                            label=r'$\rho$ (1000 cap km$^{-2}$)')
    # cbarrho.ax.set_xlabel(r'$\rho$ (1000 cap km$^{-2}$)',
    #                       fontsize='small')
    # cbarrho.ax.tick_params(labelsize='small')
    #
    # sm = plt.cm.ScalarMappable \
    #     (cmap=av_cmap, norm=plt.Normalize(vmin=min_av, vmax=max_av))
    # divider = make_axes_locatable(axes[0, 1])
    # cax_ai = divider.append_axes('bottom', size='5%', pad=0.25)
    # cbarai = fig.colorbar(sm, cax=cax_ai, orientation='horizontal',
    #                       label='$A_{v}$')
    # cbarai.ax.set_xlabel(r'$A_{v}$', fontsize='small')
    # cbarai.ax.tick_params(labelsize='small')
    #
    # axes[0, 0].set_title(rf"$\it{{Simulated}}$ " + "\n" +
    #                      rf"$\bf{{" + str(years[0]) + "}}$: " +
    #                      rf"$N_{{v}}$ = {len(ntwrk_sim[years[0]].nodes)}",
    #                      fontsize=10)
    # axes[0, 1].set_title(rf"$\it{{Observed}}$ " + "\n" +
    #                      rf"$\bf{{" + str(years[0]) + "}}$: " +
    #                      rf"$N_{{v}}$ = {len(ntwrk_obs[years[0]].nodes)}",
    #                      fontsize=10)
    for ax in axes.flatten():
        # ax.set_xlim(9650,9700)
        # ax.set_ylim(4405,4450)
        ax.set_aspect('equal')

    # plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=300)
    plt.show()

    cbfig = plt.figure(figsize=(fig.get_size_inches()[0], 1.5))
    cbfig.patch.set_facecolor('w')
    subfigs = cbfig.subfigures(1, 2)
    ax1 = subfigs[0].add_axes([0.05, 0.80, 0.9, 0.1])
    cbfig.patch.set_facecolor('w')
    cb1 = mpl.colorbar.ColorbarBase(ax1, orientation='horizontal',
                                    cmap=rho_cmap,
                                    norm=plt.Normalize(vmin=min_rho, vmax=max_rho),
                                    )
    cb1.ax.tick_params(labelsize='small')
    cb1.ax.set_xlabel(r'$\rho$ (1000 cap km$^{-2}$)', fontsize='small')

    ax2 = subfigs[1].add_axes([0.05, 0.80, 0.9, 0.1])
    cb2 = mpl.colorbar.ColorbarBase(ax2, cmap=av_cmap, orientation='horizontal',
        norm=plt.Normalize(vmin=min_av, vmax=max_av))
    cb2.ax.tick_params(labelsize='small')
    cb2.ax.set_xlabel(r'$A_{v}$', fontsize='small')

    if cbar_savepath is not None:
        plt.savefig(cbar_savepath, dpi=300)

def plot_obs_sim_contours(obs_yrs, sim_yrs, years, levels, cmap='viridis',
                          xv=None, yv=None, borders=None, savepath=None,
                          cbar_savepath=None):
    if borders is not None:
        sim_yrs = {yr: trim_array(sim_yrs[yr], borders) for yr in years}
        obs_yrs = {yr: trim_array(obs_yrs[yr], borders) for yr in years}
        xv = trim_array(xv, borders)
        yv = trim_array(yv, borders)

    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)
    fig, axes = plt.subplots(nrows=len(years), ncols=2, figsize=(6, 9))

    fig.patch.set_facecolor('w')
    for i, yr in enumerate(years):
        if xv is not None and yv is not None:
            cf_obs=axes[i, 0].contourf(xv, yv,
                obs_yrs[yr], levels=levels, extend='both')
            cf_res = axes[i, 1].contourf(xv, yv,
                sim_yrs[yr], levels=levels, extend='both')
        else:
            cf_obs=axes[i, 0].contourf(
                obs_yrs[yr], levels=levels, extend='both')
            cf_res = axes[i, 1].contourf(
                sim_yrs[yr], levels=levels, extend='both')
        axes[i, 0].set_aspect('equal')
        axes[i, 1].set_aspect('equal')
        axes[i, 0].set_ylabel(r"$\bf{" + str(yr) + "}$")

    axes[0, 0].set_title(r'Observed')
    axes[0, 1].set_title(r'Simulated')
    plt.tight_layout(h_pad=1)
    plt.tight_layout(w_pad=1)
    if savepath is not None:
        plt.savefig(savepath)

    plt.show()
    levels = np.insert(levels, 0, 0)
    levels = np.append(levels, [levels[-1] + 1000])
    norm = mpl.colors.BoundaryNorm(levels, cmap.N)
    cbfig, ax = plt.subplots(figsize=(fig.get_size_inches()[0], 0.2))
    cbfig.patch.set_facecolor('w')
    cb2 = mpl.colorbar.ColorbarBase(ax, orientation='horizontal',
                                    cmap=cmap, ticks=levels, boundaries=levels,
                                    extend='both')
    cb2.ax.tick_params(labelsize='small')
    cb2.ax.set_xlabel(r'$\rho$ (1000 cap km$^{-2}$)', fontsize='small')

    if cbar_savepath is not None:
        plt.savefig(cbar_savepath, dpi=300)
        # print(gfdfg)
    plt.show()

def plot_obs_sim_contours_network(obs_yrs, sim_yrs, ntwrk_sim, ntwrk_obs, years,
                                  levels, rho_cmap='plasma', av_cmap='Greens',
                                  borders=None, savepath=None,
                                  cbar_savepath=None):
    if borders is not None:
        sim_yrs = {yr: trim_array(sim_yrs[yr], borders) for yr in years}
        obs_yrs = {yr: trim_array(obs_yrs[yr], borders) for yr in years}

    av_cmap = cm.get_cmap(av_cmap)
    rho_cmap = cm.get_cmap(rho_cmap)

    a_v_sim, a_v_min_sim, a_v_max_sim = {}, {}, {}
    a_v_obs, a_v_min_obs, a_v_max_obs = {}, {}, {}
    min_av, max_av = {}, {}
    # rho_min, rho_max = {}, {}

    pos_dict_sim = \
        {n: (ntwrk_sim[years[-1]].nodes[n]['lon'] - 490,
             ntwrk_sim[years[-1]].nodes[n]['lat'] - 148)
         for n in ntwrk_sim[years[-1]].nodes}

    pos_dict_obs = \
        {n: (ntwrk_obs[years[-1]].nodes[n]['lon'] - 490,
             ntwrk_obs[years[-1]].nodes[n]['lat'] - 148)
         for n in ntwrk_obs[years[-1]].nodes}

    for yr in years:
        sim_ntwrk_yr = ntwrk_sim[yr]
        rn = list(sim_ntwrk_yr.nodes)[0]

        a_v_sim[yr] = list(
            nx.get_node_attributes(ntwrk_sim[yr], 'a_i_inv').values())
        a_v_min_sim[yr] = np.nanmin(a_v_sim[yr])
        a_v_max_sim[yr] = np.nanmax(a_v_sim[yr])

        obs_ntwrk_yr = ntwrk_obs[yr]
        rn = list(obs_ntwrk_yr.nodes)[0]
        # if 'pos' in obs_ntwrk_yr.nodes[rn].keys():
        #     pass

        a_v_obs[yr] = list(
            nx.get_node_attributes(ntwrk_obs[yr], 'a_i_inv').values())
        a_v_min_obs[yr] = np.nanmin(a_v_obs[yr])
        a_v_max_obs[yr] = np.nanmax(a_v_obs[yr])

        min_av[yr] = np.nanmin([a_v_min_sim[yr], a_v_min_obs[yr]])
        max_av[yr] = np.nanmax([a_v_max_sim[yr], a_v_max_obs[yr]])

    min_av = np.nanmin(list(a_v_min_sim.values()) + list(a_v_min_obs.values()))
    max_av = np.nanmax(list(a_v_max_sim.values()) + list(a_v_max_obs.values()))

    if type(rho_cmap) == str:
        rho_cmap = plt.get_cmap(rho_cmap)
    fig, axes = plt.subplots(nrows=len(years), ncols=2, figsize=(6, 9))

    fig.patch.set_facecolor('w')
    for i, yr in enumerate(years):
        cf_obs = axes[i, 0].contourf(
            obs_yrs[yr], levels=levels, extend='both')
        nx.draw_networkx(
            ntwrk_sim[yr], with_labels=False, ax=axes[i, 0], node_size=2,
            pos=pos_dict_sim, edge_color='grey', width=0.5,
            node_color=a_v_sim[yr], vmin=min_av, vmax=max_av, cmap=av_cmap)
        axes[i, 0].set_title(rf"$N_{{v}}$ = {len(ntwrk_sim[yr].nodes)}",
                             fontsize=10)

        cf_res = axes[i, 1].contourf(
            sim_yrs[yr], levels=levels, extend='both')
        nx.draw_networkx(ntwrk_obs[yr], with_labels=False, ax=axes[i, 1],
                         node_size=2, pos=pos_dict_obs, edge_color='grey',
                         width=0.5, node_color=a_v_obs[yr], vmin=min_av,
                         vmax=max_av, cmap=av_cmap)
        axes[i, 1].set_title(rf"$N_{{v}}$ = {len(ntwrk_obs[yr].nodes)}",
                             fontsize=8)
        axes[i, 0].set_aspect('equal')
        axes[i, 1].set_aspect('equal')
        axes[i, 0].set_ylabel(r"$\bf{" + str(yr) + "}$")

    axes[0, 0].set_title(r'Observed')
    axes[0, 1].set_title(r'Simulated')
    plt.tight_layout(h_pad=1, w_pad=1)
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)

    plt.show()

    levels = np.insert(levels, 0, 0)
    levels = np.append(levels, [levels[-1] + 1])
    norm = mpl.colors.BoundaryNorm(levels, cmap.N)
    cbfig = plt.figure(figsize=(fig.get_size_inches()[0], 1.5))
    cbfig.patch.set_facecolor('w')
    subfigs = cbfig.subfigures(1, 2)
    ax1 = subfigs[0].add_axes([0.05, 0.80, 0.9, 0.1])
    cbfig.patch.set_facecolor('w')
    cb1 = mpl.colorbar.ColorbarBase(ax1, orientation='horizontal',
                                    cmap=cmap, ticks=levels,
                                    boundaries=levels,
                                    norm=cf_obs.norm,
                                    extend='both')
    cb1.ax.tick_params(labelsize='small')
    cb1.ax.set_xlabel(r'$\rho$ (1000 cap km$^{-2}$)', fontsize='small')

    ax2 = subfigs[1].add_axes([0.05, 0.80, 0.9, 0.1])
    cb2 = mpl.colorbar.ColorbarBase(ax2, cmap=av_cmap, orientation='horizontal',
                                    norm=plt.Normalize(vmin=min_av,
                                                       vmax=max_av))
    cb2.ax.tick_params(labelsize='small')
    cb2.ax.set_xlabel(r'$A_{v}$', fontsize='small')

    if cbar_savepath is not None:
        plt.savefig(cbar_savepath, dpi=300)

    if cbar_savepath is not None:
        plt.savefig(cbar_savepath, dpi=300)
        # print(gfdfg)
    plt.show()

def plot_obs_sim_contours_diff(obs_yrs, sim_yrs, years, levels, figsize=(8.5,8.5),
                               xv=None, yv=None, cmap='viridis',
                               cmap_diff='bwr', borders=None, savepath=None,
                               cbar_savepath=None):
    if borders is not None:
        sim_yrs = {yr: trim_array(sim_yrs[yr], borders)/1000 for yr in
                   years}
        obs_yrs = {yr: trim_array(obs_yrs[yr], borders)/1000 for yr in
                   years}
        if xv is not None and yv is not None:
            xv = trim_array(xv, borders)
            yv = trim_array(yv, borders)

    else:
        sim_yrs = {yr: sim_yrs[yr] / 1000 for yr in years}
        obs_yrs = {yr:obs_yrs[yr] / 1000 for yr in years}
    simobs_diff = {yr: (sim_yrs[yr] - obs_yrs[yr]) for yr in years}

    simobs_diffmin = np.min([np.nanmin(simobs_diff[yr]) for yr in years])
    simobs_diffmax = np.max([np.nanmax(simobs_diff[yr]) for yr in years])
    if simobs_diffmin >= 0:
        cm_diff_norm = None
    else:
        cm_diff_norm = \
            colors.TwoSlopeNorm(vmin=simobs_diffmin,
                                vmax=simobs_diffmax, vcenter=0.)

    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)
    if type(cmap_diff) == str:
        cmap_diff = plt.get_cmap(cmap_diff)
    fig, axes = plt.subplots(nrows=len(years), ncols=3, figsize=(9, 9))

    fig.patch.set_facecolor('w')
    for i, yr in enumerate(years):
        if xv is not None and yv is not None:
            cf_res = axes[i, 0].contourf(xv, yv, sim_yrs[yr], levels=levels,
                                         cmap=cmap, extend='both')
            cf_obs = axes[i, 1].contourf(xv, yv, obs_yrs[yr], levels=levels,
                                         cmap=cmap, extend='both')
            diff = axes[i, 2].pcolor(xv, yv, simobs_diff[yr],
                                     cmap=cmap_diff, norm=cm_diff_norm,
                                     shading='auto')
        else:
            cf_res = axes[i, 0].contourf(sim_yrs[yr], levels=levels, cmap=cmap,
                                        extend='both')
            cf_obs = axes[i, 1].contourf(obs_yrs[yr], levels=levels, cmap=cmap,
                                        extend='both')
            diff = axes[i, 2].pcolor(simobs_diff[yr], cmap=cmap_diff,
                                    norm=cm_diff_norm, shading='auto')
        axes[i, 0].set_aspect('equal')
        axes[i, 1].set_aspect('equal')
        axes[i, 2].set_aspect('equal')
        axes[i, 0].set_ylabel(r"$\bf{" + str(yr) + "}$\n$y$ (km)")

    axes[0, 0].set_title(r'Simulated')
    axes[0, 1].set_title(r'Observed')
    axes[0, 2].set_title(r'$\rho_{{sim}} - \rho_{{obs}}$')

    axes[i, 0].set_xlabel("$x$ (km)")
    axes[i, 1].set_xlabel("$x$ (km)")
    axes[i, 2].set_xlabel("$x$ (km)")

    for ax in axes.flatten():
        ax.tick_params(labelsize=8)

    plt.tight_layout(h_pad=1)
    plt.tight_layout(w_pad=1)
    if savepath is not None:
        plt.savefig(savepath, dpi=300)

    plt.show()
    levels = np.insert(levels, 0, 0)
    levels = np.append(levels, [levels[-1] + 1])
    norm = mpl.colors.BoundaryNorm(levels, cmap.N)
    cbfig = plt.figure(figsize=(fig.get_size_inches()[0], 1.5))
    cbfig.patch.set_facecolor('w')
    subfigs = cbfig.subfigures(1, 2)
    ax1 = subfigs[0].add_axes([0.05, 0.80, 0.9, 0.1])
    cbfig.patch.set_facecolor('w')
    cb1 = mpl.colorbar.ColorbarBase(ax1, orientation='horizontal',
                                    cmap=cmap, ticks=levels,
                                    boundaries=levels,
                                    norm=cf_obs.norm,
                                    extend='both')
    cb1.ax.tick_params(labelsize='small')
    cb1.ax.set_xlabel(r'$\rho$ (1000 cap km$^{-2}$)', fontsize='small')

    ax2 = subfigs[1].add_axes([0.05, 0.80, 0.9, 0.1])
    cb2 = mpl.colorbar.ColorbarBase(ax2, orientation='horizontal',
                                    cmap=cmap_diff, norm=cm_diff_norm)
    cb2.ax.tick_params(labelsize='small')
    cb2.ax.set_xlabel(r'$\rho_{{sim}} - \rho_{{obs}}$ (1000 cap km$^{-2}$)',
                      fontsize='small')

    if cbar_savepath is not None:
        plt.savefig(cbar_savepath, dpi=300)
    plt.show()

    def plot_twocity_contours_diff(
            ldn_obs, ldn_sim, syd_obs, syd_sim, levels, xvyv_ldn, xvyv_syd,
            figsize, cmap='viridis', cmap_diff='bwr', borders=None,
            savepath=None, cbar_savepath=None):

        ldn_sim = trim_array(ldn_sim, borders) / 1000
        ldn_obs = trim_array(ldn_obs, borders) / 1000

        syd_sim = trim_array(syd_sim, borders) / 1000
        syd_obs = trim_array(syd_obs, borders) / 1000

        xv_ldn = trim_array(xvyv_ldn['xv_med'], borders)
        yv_ldn = trim_array(xvyv_ldn['yv_med'], borders)

        xv_syd = trim_array(xvyv_syd['xv_med'], borders)
        yv_syd = trim_array(xvyv_syd['yv_med'], borders)

        ldn_diff = ldn_sim - ldn_obs
        syd_diff = syd_sim - syd_obs

        diffmin = np.min((np.nanmin(ldn_diff), np.nanmin(syd_diff)))
        diffmax = np.max((np.nanmax(ldn_diff), np.nanmax(syd_diff)))
        if diffmin >= 0:
            cm_diff_norm = None
        else:
            cm_diff_norm = \
                colors.TwoSlopeNorm(vmin=diffmin, vmax=diffmax, vcenter=0.)

        if type(cmap) == str:
            cmap = plt.get_cmap(cmap)
        if type(cmap_diff) == str:
            cmap_diff = plt.get_cmap(cmap_diff)
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=figsize)

        fig.patch.set_facecolor('w')

        ldn_sax = axes[0, 0].contourf(xv_ldn, yv_ldn, ldn_sim, levels=levels,
                                      # type: ignore
                                      cmap=cmap, extend='both')
        ldn_oax = axes[0, 1].contourf(xv_ldn, yv_ldn, ldn_obs, levels=levels,
                                      # type: ignore
                                      cmap=cmap, extend='both')
        ldn_dax = axes[0, 2].pcolor(xv_ldn, yv_ldn, ldn_diff, cmap=cmap_diff,
                                    # type: ignore
                                    norm=cm_diff_norm, shading='auto')

        syd_sax = axes[1, 0].contourf(xv_syd, yv_syd, syd_sim, levels=levels,
                                      # type: ignore
                                      cmap=cmap, extend='both')
        syd_oax = axes[1, 1].contourf(xv_syd, yv_syd, syd_obs, levels=levels,
                                      # type: ignore
                                      cmap=cmap, extend='both')
        syd_dax = axes[1, 2].pcolor(xv_syd, yv_syd, syd_diff, cmap=cmap_diff,
                                    # type: ignore
                                    norm=cm_diff_norm, shading='auto')

        for ax in axes.flatten():
            ax.set_aspect('equal')
            ax.tick_params(axis='both', labelsize=8)
            ax.set_xlabel("$x$ (km)", fontsize=8)
            ax.set_ylabel("$y$ (km)", fontsize=8)
        axes[0, 0].set_title(r'Simulated', fontsize=10)
        axes[0, 1].set_title(r'Observed', fontsize=10)
        axes[0, 2].set_title(r'$\rho_{{sim}} - \rho_{{obs}}$', fontsize=10)

        plt.tight_layout(h_pad=1)
        plt.tight_layout(w_pad=1)
        if savepath is not None:
            plt.savefig(savepath, dpi=300)
        plt.show()

        levels = np.insert(levels, 0, 0)
        levels = np.append(levels, [levels[-1] + 1])
        norm = mpl.colors.BoundaryNorm(levels, cmap.N)
        cbfig = plt.figure(figsize=(fig.get_size_inches()[0], 1.5))
        cbfig.patch.set_facecolor('w')
        subfigs = cbfig.subfigures(1, 2)
        ax1 = subfigs[0].add_axes([0.05, 0.80, 0.9, 0.1])
        cbfig.patch.set_facecolor('w')
        cb1 = mpl.colorbar.ColorbarBase(ax1, orientation='horizontal',
                                        cmap=cmap, ticks=levels,
                                        boundaries=levels,
                                        norm=ldn_sax.norm,
                                        extend='both')
        cb1.ax.tick_params(labelsize='small')
        cb1.ax.set_xlabel(r'$\rho$ (1000 cap km$^{-2}$)', fontsize='small')

        ax2 = subfigs[1].add_axes([0.05, 0.80, 0.9, 0.1])
        cb2 = mpl.colorbar.ColorbarBase(ax2, orientation='horizontal',
                                        cmap=cmap_diff, norm=cm_diff_norm)
        cb2.ax.tick_params(labelsize='small')
        cb2.ax.set_xlabel(r'$\rho_{{sim}} - \rho_{{obs}}$ (1000 cap km$^{-2}$)',
                          fontsize='small')

        if cbar_savepath is not None:
            plt.savefig(cbar_savepath, dpi=300)
        plt.show()


def plot_cluster(obs_clusters, sim_clusters, obs_dist, sim_dist, years, limit,
                 savepath=None, show=False):
    """
    Plots the contiguous clsuter of the population above a limit

    :param obs_clusters: dict
        Dictionary of observed clusters, keys: year, values: 2D array of data
    :param sim_clusters: dict
        Dictionary of simulated clusters, keys: year, values: 2D array of data
    :param obs_dist: dict
        Dictionary of observed cluster radii, keys: year, values: 1D array of
        radii
    :param sim_dist: dict
        Dictionary of simlated cluster radii, keys: year, values: 1D array of
        radii
    :param years: list
        Years
    :param limit: int
        Population size lower limit
    :param savepath: str
        Save path
    :param show: bool
        Bool to show plot
    :return:
    """
    fig, axes = plt.subplots(nrows=len(years), ncols=2,
                             figsize=(7.5, 3.375 * len(years)))
    fig.patch.set_facecolor('w')
    for i, yr in enumerate(years):
        axes[i, 0].pcolor(obs_clusters[yr])
        axes[i, 1].pcolor(sim_clusters[yr])
        axes[i, 0].set_aspect('equal')
        axes[i, 1].set_aspect('equal')

    for ax in axes[0, :]:
        ax.set_title(rf'$\rho$ > {limit}')

    if savepath:
        plt.savefig(savepath + 'cluster_plots.png')
    if show:
        plt.show()

    fig, ax = plt.subplots(figsize=(4, 2))
    fig.patch.set_facecolor('w')
    ax.plot([obs_dist[y] for y in years], marker='o', label="Obs.")
    ax.plot([sim_dist[y] for y in years], marker='o', label="Sim.")
    ax.set_xticks(np.arange(len(list(obs_dist.keys()))))
    ax.set_xticklabels(list(obs_dist.keys()))
    ax.set_xlabel('Year')
    ax.set_ylabel('Radius (km)')
    ax.set_title(rf'$\rho$ > {limit}')
    ax.legend()
    if savepath:
        plt.savefig(savepath + 'radius_years.png')
    if show:
        plt.show()

def plot_obs_sim_accuracy(obs_yrs, sim_yrs, years, savepath, show=False):
    """
    Plot choropleths of simulated and observed values, along with 1:1 plot

    :param obs_yrs:
    :param sim_yrs:
    :param years:
    :return:
    """
    for yr in years:
        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(13, 4))
        fig.patch.set_facecolor('w')
        obs = obs_yrs[yr]
        sim = sim_yrs[yr]
        vmin = min(np.nanmin(sim), np.nanmin(obs))
        vmax = max(np.nanmax(sim), np.nanmax(obs))
        ax1.pcolor(sim, vmin=vmin, vmax=vmax)
        img1 = ax2.pcolor(obs, vmin=vmin, vmax=vmax)
        ax1.set_aspect('equal')
        ax2.set_aspect('equal')
        #     ax1.set_title("$D$=0.05, $Y_{0}$=4.4, $B$=0.099")
        ax1.set_title("Simulated")
        ax2.set_title("Observed")
        plot_accuracy_scatter(ax=ax3, obs_data=obs, sim_data=sim,
                              varobs=r'Obs. $\rho$ (1000s)',
                              varsim=r'Sim. $\rho$ (1000s)',
                              borders=(6, 6), log=True, cmap='plasma')

        # divider1 = make_axes_locatable(ax2)
        # cax1 = divider1.append_axes('right', size='5%')
        # cbar1 = fig.colorbar(img1, cax=cax1)

        if savepath is not None:
            plt.savefig(f"{savepath}_{yr}.png")
        if show:
            plt.show()

def choropleth_and_accuracy(obs, sim, year, network=None, savepath=None, \
                            show=False):
    """
    Creates 4 plots: choroppleths of simulated and observed values, and 1:1
    plots for central and outskirt areas of the domain. Network may be
    included on the observed choropleth.

    :param obs:
    :param sim:
    :param year:
    :param network:
    :param savepath:
    :param show:
    :return:
    """

    vmin = min(np.nanmin(sim), np.nanmin(obs)) / 1000
    vmax = max(np.nanmax(sim), np.nanmax(obs)) / 1000

    fig = plt.figure(constrained_layout=True, figsize=(7.5, 7.5))
    fig.suptitle(year)
    fig.patch.set_facecolor('w')

    subfigs = fig.subfigures(3, 1, height_ratios=[4, 1, 4])

    spec1 = gridspec.GridSpec(ncols=2, nrows=2, figure=subfigs[0])
    ax1 = subfigs[0].add_subplot(spec1[:, 0:1])
    ax2 = subfigs[0].add_subplot(spec1[:, 1:2])
    spec2 = gridspec.GridSpec(ncols=1, nrows=1, figure=subfigs[1])
    caxrho = subfigs[1].add_subplot(spec2[:, :])
    spec3 = gridspec.GridSpec(ncols=2, nrows=2, figure=subfigs[2])
    ax3 = subfigs[2].add_subplot(spec3[:, 0:1])
    ax4 = subfigs[2].add_subplot(spec3[:, 1:2])

    ax1.pcolor(sim / 1000, vmin=vmin, vmax=vmax)
    img1 = ax2.pcolor(obs / 1000, vmin=vmin, vmax=vmax)
    if network is not None:
        nx.draw_networkx(ntwrk_years[year], with_labels=False, ax=ax2,
            node_size=3, pos=nx.get_node_attributes(ntwrk_years[year], 'pos'),
            edge_color='grey', width=0.5, node_color='r', )
    ax2.set_xlim(0, 84)
    ax2.set_ylim(0, 66)
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    #     ax1.set_title("$D$=0.05, $Y_{0}$=4.4, $B$=0.099")
    ax1.set_title("Simulated")
    ax2.set_title("Observed")

    cb_rho = plt.cm.ScalarMappable(cmap='viridis',
                                   norm=plt.Normalize(vmin=1, vmax=4))
    cb_rho._A = []
    subfigs[0].colorbar(img1, cax=caxrho, shrink=0.3, orientation='horizontal')
    caxrho.set_xlabel('Population density (1000 cap km$^{-2}$)')
    ax3 = plot_accuracy_scatter(ax=ax3, obs_data=obs[15:51, 25:59],
                                sim_data=sim[15:51, 25:59],
                                varobs=r'Obs. $\rho$ (1000s)',
                                varsim=r'Sim. $\rho$ (1000s)', borders=(6, 6),
                                log=True, cmap='plasma')
    ax3.set_title('Centre')
    sim_cut = copy(sim)
    sim_cut[15:51, 25:59] = np.nan
    obs_cut = copy(obs)
    obs_cut[15:51, 25:59] = np.nan
    ax4 = plot_accuracy_scatter(ax=ax4, obs_data=obs_cut,
                                sim_data=sim_cut, varobs=r'Obs. $\rho$ (1000s)',
                                varsim=r'Sim. $\rho$ (1000s)', borders=(6, 6),
                                log=True, cmap='plasma')
    max_val = np.nanmax(np.array([obs_cut, sim_cut]))
    ax4.plot([0, max_val], [0, max_val], ls='--', color='g', lw=2)
    #     ax3.tick_params(axis='both', which='both')
    ax4.set_title('Outskirts')
    if savepath is not None:
        plt.savefig(savepath)
    if show:
        plt.show()

def plot_full_network_rho(stn_graph, rho, xvyv, phi, g, N, year, savepath=None,
                          show=False):
    """

    :param stn_graph:
    :param rho:
    :param phi:
    :param g:
    :param N:
    :param new_nodes:
    :param savepath:
    :return:
    """
    a_v = list(nx.get_node_attributes(stn_graph, 'a_i_inv').values())
    a_v_min = np.nanmin(a_v)
    a_v_max = np.nanmax(a_v)

    node_pos = nx.get_node_attributes(stn_graph, 'pos')

    fig, ax = plt.subplots()
    rho_plot = ax.pcolor(xvyv['xv'], xvyv['yv'], rho / 1000)
    cbar = ax.figure.colorbar(rho_plot)
    cbar.ax.set_ylabel(r'$\rho$ (cap km$^{-2}$')
    ax.set_title(
        f"{year}: $\phi$ = {phi}, g = {g}, N = {N}")
    nx.draw(G=stn_graph, node_size=5, node_color=a_v, pos=node_pos,
            cmap='Greens')

    ax.set_aspect('equal')
    limits = plt.axis('on')  # turns on axis
    ax.tick_params(left=True, bottom=True, labelleft=True,
                   labelbottom=True)
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=300)
    if show:
        plt.show()

def plot_rho_log_log_rad(pop_obs, pop_sim, years, xvyv, city_outline,
                         obs_cluster_dict, sim_cluster_dict, borders=None,
                         savepath=None, inset=False, radius_line=False,
                         unit_convert=None, log_mode='cumsum', centre=[0,0]):
    """
    Plot comparison between observed and simulated values for the scaling of
    the population density. Radius of the city is found via the City
    Clustering algorithm (must be created in R).

    :param pop_obs:
    :param pop_sim:
    :param years:
    :param xvyv:
    :param city_outline:
    :param obs_cluster_dict:
    :param sim_clusyer_path:
    :param savepath:
    :param inset:
    :param radius_line:
    :param unit_convert:
    :param log_mode:
    :return:
    """
    orig_cmap = plt.cm.Set1
    col_space = np.linspace(0, 1, num=len(years))

    obsbeta, simbeta, obsrad, simrad = [], [], [], []
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor('w')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    for i, yr in enumerate(years):

        obsr = find_city_radius(cluster=obs_cluster_dict[yr], centre=centre,
                                xvyv=xvyv, borders=borders,
                                city_outline=city_outline)[1]
        obsx, obsy, obsa, obsb = find_city_scaling(
            data=pop_obs[yr], centre=centre, borders=borders, radius=obsr)

        simr = find_city_radius(cluster=sim_cluster_dict[yr], centre=centre,
                                xvyv=xvyv, borders=borders,
                                city_outline=city_outline)[1]
        simx, simy, sima, simb = find_city_scaling(
            data=pop_sim[yr], borders=borders, radius=simr, centre=centre)

        obsr = int(np.round(obsr))
        simr = int(np.round(simr))

        obsbeta.append(obsa)
        simbeta.append(sima)
        obsrad.append(obsr)
        simrad.append(simr)

        obsx, obsy = remove_nan_inf(obsx, obsy)

        ax.plot(obsx, obsy, label=yr, color=orig_cmap(col_space[i]), lw=1.5,
                ls=':')

        simx, simy = remove_nan_inf(simx, simy)

        ax.plot(simx, simy, label=yr, color=orig_cmap(col_space[i]), lw=1.5)

    ax.set_ylabel("log$_{10}$(Cumulative population) (cap)")
    ax.set_xlabel("log$_{10}rad$ (km)")
    ax.legend(framealpha=0)
    # ax.set_ylim(2.95, 4.45)
    if inset:
        # Create a set of inset Axes: these should fill the bounding box
        # allocated to them.
        ax_ins1 = plt.axes([0, 0, 1, 1])
        # Manually set the position and relative size of the inset axes
        # within ax1
        ip = InsetPosition(ax, [0.1, 0.72, 0.15, 0.25])
        ax_ins1.set_axes_locator(ip)
        print('SIMBETA: ', simbeta)
        print('OBSBETA: ', obsbeta)
        ax_ins1.scatter(np.arange(len(years)), simbeta,
                        color=orig_cmap(col_space))
        ax_ins1.scatter(np.arange(len(years)), obsbeta,
                        color=orig_cmap(col_space), marker='^')

        ax_ins1.tick_params(axis='y', labelsize=8)
        ax_ins1.set_title(r'$b$', fontsize=8)
        ax_ins1.set_xlim(-.5, len(years) - .5)
        ax_ins1.set_ylim(np.min([simbeta, obsbeta]) - .1,
                         np.max([simbeta, obsbeta]) + .1)
        ax_ins1.set_xticks(np.arange(len(years)))
        ax_ins1.set_xticklabels(labels=years, rotation=70, fontsize=8)
        ax_ins1.tick_params(axis='y', labelsize=8)

        ax.plot((.65, .65), (5.3, 6), color='k', lw=1)
        ax.plot((.65, .95), (6, 6), color='k', lw=1)
        ax.annotate(r'$b$', xy=(.65,6.3), xycoords='data', ha='left', va='top',
                    color='k', fontsize=9)

    if radius_line:
        print('SIMRAD: ', simrad)
        print('OBSRAD: ', obsrad)
        for i, yr in enumerate(years):
            ax.axvline(np.log10(simrad[i]), color=orig_cmap(col_space[i]),
                       lw=1, zorder=0)
            ax.axvline(np.log10(obsrad[i]), color=orig_cmap(col_space[i]),
                       ls='--', lw=1, zorder=0)

    labels_a = [yr for yr in years] + ['Obs.', 'Sim.']
    lineparams_a = zip(labels_a, [orig_cmap(c) for c in col_space] + ['k', 'k'],
                       ['', '', '', '', '^', 'o'],
                       ['-', '-', '-', '-', '--', '-'])
    lines_a = [
        mlines.Line2D([], [], color=color, label=label, ls=lsty, marker=m)
        for label, color, m, lsty in list(lineparams_a)[:4]]
    lines_a.append(
        (mlines.Line2D([], [], color='k', label='Obs.', ls='', marker='^'),
         (mlines.Line2D([], [], color='k', label='Obs.', ls=':', marker='',
                        lw=1))))
    lines_a.append(
        (mlines.Line2D([], [], color='k', label='Sim.', ls='', marker='o'),
         (mlines.Line2D([], [], color='k', label='Sim.', ls='-', marker='',
                        lw=1))))

    ax.legend(lines_a, labels_a, framealpha=0, loc='lower right',
              handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=9)
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath + '.png', dpi=300)
    plt.show()

def plot_stns_log_log_rad(pop_obs, stn_obs, obs_cluster_dict, pop_sim, stn_sim,
                          sim_cluster_dict, years, xvyv, city_outline,
                          borders=None, savepath=None, inset=False,
                          radius_top=False, radius_line=False,
                          log_mode='cumsum'):
    orig_cmap = plt.cm.Set1
    col_space = np.linspace(0, 1, num=len(years))

    obsbeta, simbeta, obsrad, simrad = [], [], [], []
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor('w')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    for i, yr in enumerate(years):
        obsr = find_city_radius(cluster=obs_cluster_dict[yr], borders=borders,
                                xvyv=xvyv, city_outline=city_outline)[1]

        simr = find_city_radius(cluster=sim_cluster_dict[yr], borders=borders,
                                xvyv=xvyv, city_outline=city_outline)[1]

        obsr = int(np.round(obsr))
        simr = int(np.round(simr))

        xvar_obs, yvar_obs, a_obs, b_obs = \
            find_city_scaling(data=stn_obs[yr], borders=borders, radius=obsr)
        xvar_sim, yvar_sim, a_sim, b_sim = \
            find_city_scaling(data=stn_sim[yr], borders=borders, radius=simr)

        xvar_obs, yvar_obs = remove_nan_inf(xvar_obs, yvar_obs)

        ax.plot(xvar_obs, yvar_obs, label=yr, color=orig_cmap(col_space[i]),
                lw=1.5, ls=':')

        xvar_sim, yvar_sim = remove_nan_inf(xvar_sim, yvar_sim)

        ax.plot(xvar_sim, yvar_sim, label=yr, color=orig_cmap(col_space[i]),
                lw=1.5)

        obsbeta.append(a_obs)
        simbeta.append(a_sim)
        obsrad.append(obsr)
        simrad.append(simr)

    ax.set_ylabel("log$_{10}$(Cumulative $N_{v}$)")
    ax.set_xlabel("log$_{10}rad$ (km)")

    if inset:
        # Create a set of inset Axes: these should fill the bounding box
        # allocated to them.
        ax_ins1 = plt.axes([0, 0, 1, 1])
        # Manually set the position and relative size of the inset axes
        ip = InsetPosition(ax, [0.1, 0.7, 0.15, 0.25])

        ax_ins1.set_axes_locator(ip)

        print('SIMBETA: ', simbeta)
        print('OBSBETA: ', obsbeta)
        ax_ins1.scatter(np.arange(len(years)), simbeta,
                        color=orig_cmap(col_space), s=15)
        ax_ins1.scatter(np.arange(len(years)), obsbeta,
                        color=orig_cmap(col_space), s=15, marker='^')

        ax_ins1.tick_params(axis='y', labelsize=8)
        ax_ins1.set_title(r'$c $', fontsize=8)
        ax_ins1.set_xlim(-.5, len(years) - .5)
        ax_ins1.set_ylim(np.min([simbeta, obsbeta]) - .1,
                         np.max([simbeta, obsbeta]) + .1)
        ax_ins1.yaxis.set_label_position('right')
        ax_ins1.set_xticks(np.arange(len(years)))
        ax_ins1.set_xticklabels(labels=years, rotation=70, fontsize=8)
        ax_ins1.tick_params(axis='y', labelsize=8)

        ax.plot((0.4, 0.6), (1.3, 1.3), color='k', lw=1)
        ax.plot((0.6, 0.6), (1.3, 1.6), color='k', lw=1)
        ax.annotate(r'$c$', xy=(0.6, 1.3), xycoords='data', ha='left', va='top',
                    color='k', fontsize=9)
        if savepath is not None:
            pickle.dump({'Sim': simbeta, 'Obs': obsbeta, 'years': years},
                        open(savepath + '.p', 'wb'))

    if radius_line:
        for i, yr in enumerate(years):
            ax.axvline(np.log10(simrad[i]), color=orig_cmap(col_space[i]),
                       lw=1, zorder=0)
            ax.axvline(np.log10(obsrad[i]), color=orig_cmap(col_space[i]),
                       ls='--', lw=1, zorder=0)

    labels_a = [yr for yr in years] + ['Obs.', 'Sim.']
    lineparams_a = zip(labels_a, [orig_cmap(c) for c in col_space] + ['k', 'k'],
                       ['', '', '', '', '^', 'o'],
                       ['-', '-', '-', '-', '--', '-'])
    lines_a = [
        mlines.Line2D([], [], color=color, label=label, ls=lsty, marker=m)
        for label, color, m, lsty in list(lineparams_a)[:4]]
    lines_a.append(
        (mlines.Line2D([], [], color='k', label='Obs.', ls='', marker='^'),
         (mlines.Line2D([], [], color='k', label='Obs.', ls=':', marker='',
                        lw=1))))
    lines_a.append(
        (mlines.Line2D([], [], color='k', label='Sim.', ls='', marker='o'),
         (mlines.Line2D([], [], color='k', label='Sim.', ls='-', marker='',
                        lw=1))))

    ax.legend(lines_a, labels_a, framealpha=0, loc='lower right',
              handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=9)
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath + '.png', dpi=300)
    plt.show()

def plot_sensitivity_analysis(results, savepath=None):
    """
    Creates a barplot (including confidence intervals) of senstivity analysis
    results
    :return:
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    sal_barplot(results, ax=ax)
    ax.set_xlabel('Variable')
    ax.set_ylabel('Contribution')
    ax.get_legend().remove()
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()

def plot_total_region_rho(obs_rho, sim_rho, centre, suburbs, years, ymin=None,
                          savepath=None):
    """

    :return:
    """
    sim_pop = np.array([np.nansum(trim_array(sim_rho[yr], (6,6)))
                        for yr in years])
    obs_pop = np.array([np.nansum(trim_array(obs_rho[yr], (6, 6)))
                        for yr in years])

    fig = plt.figure(constrained_layout=True, figsize=(4.5, 6))

    spec = gridspec.GridSpec(ncols=1, nrows=5, figure=fig)
    ax1 = fig.add_subplot(spec[0:3, 0])
    ax2 = fig.add_subplot(spec[3:, 0])
    fig.patch.set_facecolor('w')

    ax1.plot(years, obs_pop, color='forestgreen', lw=2, ls='--',
             label='Obs. $N_{pop}$')
    ax1.plot(years, sim_pop, color='forestgreen', lw=2,
             label='Sim. $N_{pop}$')
    ax1.set_ylabel('$N_{pop}$ (cap)', fontdict={'size': 'small'})
    ax1.tick_params(axis='y', labelsize=8)
    if ymin is not None:
        ax1.set_ylim(ymin=ymin)
    ax1.set_xlim(years[0], years[-1])

    ax1.legend(fontsize=10)
    ax1.tick_params(axis='x', labelbottom=False)

    cent_area = obs_rho[2011][centre[0]:centre[1],
                              centre[2]:centre[3]].shape[0] * \
                obs_rho[2011][centre[0]:centre[1],
                              centre[2]:centre[3]].shape[1]
    sub_area = obs_rho[2011][suburbs[0]:suburbs[1],
                             suburbs[2]:suburbs[3]].shape[0] * \
               obs_rho[2011][suburbs[0]:suburbs[1],
                             suburbs[2]:suburbs[3]].shape[1]

    suburbs_N_obs = []
    centre_N_obs = []
    suburbs_N_sim = []
    centre_N_sim = []
    total_N_obs = []
    total_N_sim = []
    for yr in years:
        print(yr)
        cent_obs = np.nansum(obs_rho[yr][centre[0]:centre[1],
                                          centre[2]:centre[3]]) / cent_area
        sub_obs = np.nansum(obs_rho[yr][suburbs[0]:suburbs[1],
                                         suburbs[2]:suburbs[3]]) / sub_area
        cent_sim = np.nansum(sim_rho[yr][centre[0]:centre[1],
                                          centre[2]:centre[3]]) / cent_area
        sub_sim = np.nansum(sim_rho[yr][suburbs[0]:suburbs[1],
                                         suburbs[2]:suburbs[3]]) / sub_area
        centre_N_obs.append(cent_obs)
        suburbs_N_obs.append(sub_obs)
        total_N_obs.append(np.nansum(obs_rho[yr]))
        total_N_sim.append(np.nansum(sim_rho[yr]))
    suburbs_N_obs = np.array(suburbs_N_obs)/1000
    centre_N_obs = np.array(centre_N_obs)/1000

    obs_sub = np.array([np.nansum(obs_rho[yr][suburbs[0]:suburbs[1],
                                              suburbs[2]:suburbs[3]]) / sub_area
                        for yr in years]) / 1000
    obs_cent = np.array([np.nansum(obs_rho[yr][centre[0]:centre[1],
                                               centre[2]:centre[3]]) / cent_area
                        for yr in years]) / 1000
    sim_sub = np.array([np.nansum(sim_rho[yr][suburbs[0]:suburbs[1],
                                         suburbs[2]:suburbs[3]]) / sub_area
                        for yr in years]) / 1000
    sim_cent = np.array([np.nansum(sim_rho[yr][centre[0]:centre[1],
                                          centre[2]:centre[3]]) / cent_area
                        for yr in years]) / 1000

    ax2.plot(years, obs_sub, color='r', lw=2, label='Obs. suburban',
             ls='--')
    ax2.plot(years, obs_cent, color='orange', lw=2, label='Obs. central',
             ls='--')
    ax2.plot(years, sim_sub, color='r', lw=2, label='Sim. suburban')
    ax2.plot(years, sim_cent, color='orange', lw=2, label='Sim. central')

    ax2.set_ylabel(r'$\rho$ (1000 cap km$^-2$)')
    ax2.set_xlabel('Year')

    ax2.set_xlim(years[0], years[-1])
    ax2.legend(fontsize=8)

    if savepath is not None:
        plt.savefig(savepath, dpi=300)

    plt.show()

def plot_obssim_stncount_profiles(obs_ntwrks, year,  xvyv, sim_ntwrks=None,
                                  title=None, savepath=None, borders=None,
                                  legend=False):
    """
    Plots total station counts along the x and y axes
    """

    xmin = np.min(xvyv['xv'])
    xmax = np.max(xvyv['xv'])
    ymin = np.min(xvyv['yv'])
    ymax = np.max(xvyv['yv'])

    obs_stn_mat = graph_to_array(
        graph=obs_ntwrks[year], stn_mat=np.zeros_like(xvyv['xv']),
        xylim=[0, 0, xmax - xmin, ymax - ymin])

    if sim_ntwrks is not None:
        sim_stn_mat = graph_to_array(
            graph=sim_ntwrks[year], stn_mat=np.zeros_like(xvyv['xv']),
            xylim=[0, 0, xmax - xmin, ymax - ymin])

    if borders is not None:
        obs_stn_mat = trim_array(obs_stn_mat, borders)
        if sim_ntwrks is not None:
            sim_stn_mat = trim_array(sim_stn_mat, borders)
        xv = trim_array(xvyv['xv'], borders)
        yv = trim_array(xvyv['yv'], borders)
    else:
        xv = xvyv['xv']
        yv = xvyv['yv']

    obs_x_profile = obs_stn_mat.sum(axis=0)
    obs_y_profile = obs_stn_mat.sum(axis=1)
    if sim_ntwrks is not None:
        sim_x_profile = sim_stn_mat.sum(axis=0)
        sim_y_profile = sim_stn_mat.sum(axis=1)

    fig, (ax1, ax2) = \
        plt.subplots(nrows=2, figsize=(5, 3))
    fig.patch.set_facecolor('w')

    ax1.plot(xv[0, :], obs_x_profile, color='b', label='Obs.')
    if sim_ntwrks is not None:
        ax1.plot(xv[0, :], sim_x_profile, color='r', label='Sim.')

    ax1.set_xlim(xv[0, 0], xv[0, -1])
    ax1.set_xlabel(r'x-direction (km)', fontsize='small')

    ax2.plot(yv[:, 0], obs_y_profile, color='b')
    if sim_ntwrks is not None:
        ax2.plot(yv[:, 0], sim_y_profile, color='r')

    ax2.set_xlim(yv[0, 0], yv[-1, 0])
    ax2.set_xlabel(r'y-direction (km)', fontsize='small')

    for ax in (ax1, ax2):
        ax.set_ylabel(r'$N_{v}$', fontdict={'size': 8})
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylim(ymax=110)

    len_diff = abs(len(xv[0, :]) - len(yv[:, 0]))

    # only works for current shape as x > y
    ax1.set_xlim(xv[0, :].min(), xv[0, :].max())
    ax2.set_xlim(yv[:, 0].min() - len_diff // 2,
                 yv[:, 0].max() + len_diff // 2)

    ax1.text(0.1, 0.4, 'W', color='k', transform=ax1.transAxes)
    ax1.text(0.9, 0.4, 'E', color='k', transform=ax1.transAxes)
    ax2.text(0.1, 0.4, 'S', color='k', transform=ax2.transAxes)
    ax2.text(0.9, 0.4, 'N', color='k', transform=ax2.transAxes)

    if title is not None:
        fig.suptitle(title, fontsize=10, x=0.05, y=0.55, rotation=90,
                     fontweight='bold')

    if legend:
        ax1.legend(ncol=2, fontsize=8, frameon=False)
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=300)
    plt.show()
    plt.close()


if __name__ == "__main__":
    rho_res = pickle.load(
        open("./Runs/domain_AICT_accrho_1881ct_1991regrowth_IVKc/rho_results.p",
             'rb'))
    obs_rho = pickle.load(open("./growth_results/rho_yrs.p", 'rb'))

    xvyv = pickle.load(open("./growth_results/xvyv.p", "rb"))
    './core_data/network/networks_5years_nodup_factcnct.p'
    pop_grid = gpd.read_file("./core_data/population/"
                             "reg_grid_pop_1881_2011_nullfixed_allyears.shp")
    years = [1881, 1951, 1991, 2011]

    ntwrk_years = \
        pickle.load(
            open(f'./core_data/network/networks_5years_nodup_factcnct.p',
                 'rb'))
    sim_ntwrk_yrs = \
        pickle.load(
            open("./NetworkRuns/initialnetwork_rerun/simulated_networks.p",
                 'rb'))

    for yr in sim_ntwrk_yrs.keys():
        sim_ntwrk_yrs[yr] = set_accessibility(G=sim_ntwrk_yrs[yr])[0]

    plot_multiple_obssim_network(
        rho=obs_rho, ntwrk_sim=sim_ntwrk_yrs, ntwrk_obs=ntwrk_years,
        rho_cmap='plasma', av_cmap='Greens',
        years=years, borders=(6, 6), rho_vmin=300, savepath=None)
