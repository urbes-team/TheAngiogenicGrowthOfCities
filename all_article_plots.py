"""
Functions for creating all plots for the article.

@author: Isabella Capel-Timms
"""
# %load_ext autoreload
# %autoreload 2
import numpy as np
import pickle
import networkx as nx
import geopandas as gpd
from copy import copy
import os
import inspect
import math
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.lines as mlines
import matplotlib.patches as patches
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)

from plot_growth import shiftedColorMap, plot_xy_mean_years, plot_var
from graph_functions import set_accessibility
from analysis_functions import mean_smooth, trim_array, find_city_radius, \
    find_city_scaling
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple


def remove_nan_inf(xvar, yvar, zero=False):
    """
    Removes the union of NaN and inf values from two 1D arrays
    """
    xvar = np.where(np.isinf(xvar), np.nan, xvar)
    yvar = np.where(np.isinf(yvar), np.nan, yvar)

    nanxvar = np.isnan(xvar)
    nanyvar = np.isnan(yvar)

    nanboth = np.logical_or(nanxvar, nanyvar)

    xvar = xvar[~nanboth]
    yvar = yvar[~nanboth]

    return xvar, yvar


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


def radial_profile(data, centre=None, unit_convert=None):
    """
    Calculates a radial profile of data focussing on an origin
    https://stackoverflow.com/questions/21242011/most-efficient-way-to-
    calculate-radial-profile

    :param data: np.ndarray
        Array of data
    :param centre: tuple, list, None
        Origin point of data from which radial profile will originate
    :param unit_convert:
        Factor by which to multiply the distance in order to convert
        units. By using smaller units this will increase the accuracy
        of r
    :return:
    """

    if centre is None:
        centre = [0, 0]
        centre[0] = math.floor(data.shape[1] / 2)
        centre[1] = math.floor(data.shape[0] / 2)

    # Clean NaNs and infs from data
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    y, x = np.indices(data.shape)
    r = np.sqrt((x - centre[0]) ** 2 + (y - centre[1]) ** 2)

    if unit_convert is not None:
        r *= unit_convert
    # Radius rounded for use as bin edges
    r = np.round(r).astype(int)

    keep = ~np.isnan(data.ravel())
    # Sum of data within each radius category
    tbin = np.bincount(r.ravel()[keep], data.ravel()[keep])
    # Number of grid cells in each radius category
    nr = np.bincount(r.ravel()[keep])
    # Mean of data in each radius category
    radialprofile = tbin / nr

    return radialprofile, r, tbin, nr


def find_city_boundary(rho_arr, log_trim, city_bndry):
    # Find radius (xvar) and cumulative population (yvar) of data
    xvar, yvar, dist = \
        log_log_radial(rho_arr, boundary=log_trim, centre=None,
                       unit_convert=1000)
    # Log and arg of defined city boundary
    city_bndry = np.log10(city_bndry * 1000)
    city_bndry = np.argwhere(xvar > city_bndry).min()
    # Find MAE of best fit to radius (from defined city_bndry to better
    # determine city radius)
    mae = []
    for i, x in enumerate(xvar[::-1]):
        if i > 0:
            a, b = np.polyfit(xvar[:-i], yvar[:-i], 1)
            mae.append(mean_absolute_error(y_true=yvar[:-i],
                                           y_pred=a * xvar[:-i] + b))

    min_mae = np.argwhere(mae == np.min(mae[::-1][city_bndry:]))[0][0]

    a, b = np.polyfit(xvar[:min_mae], yvar[:min_mae], 1)

    r = 10 ** xvar[min_mae]

    return r, a, b


def log_log_radial(data, boundary=None, centre=None, unit_convert=None):
    """
    Finds the log10 values of the cumulative sum of radial data.

    :param data: np.ndarray
        Array of original data
    :param boundary: int
        Distance from the centre after which the radial profile will be
        truncated
    :param centre: tuple, list, None
        origin point of data from which radial profile will originate
    """

    rad_prof = radial_profile(data, centre=centre, unit_convert=unit_convert)

    #     if unit_convert is not None:
    #         boundary = np.argwhere(np.unique(rad_prof[1]) == boundary*unit_convert)[0][0]
    #         print(boundary)

    if unit_convert is not None:
        zero_change = np.argwhere(rad_prof[2][:boundary * unit_convert] > 0)
        rad_prof_sum = np.cumsum(rad_prof[2])[zero_change].flatten()
        dist = np.arange(boundary * unit_convert)[zero_change].flatten()
    else:
        rad_prof_sum = np.cumsum(rad_prof[2])[:boundary]
        dist = np.unique(rad_prof[1])[:boundary]

    #     print(rad_prof_sum)
    xvar = np.log10(dist)
    yvar = np.log10(rad_prof_sum)

    return xvar, yvar, dist


def plot_obs_stations_pop(pop_dict, years, ntwrk_yrs, N_pop, N_stns):
    # POPULATION AND NO. OF STATIONS
    plas = cm.plasma(0.25)
    gre = cm.Greens(0.75)

    # years = [1831, 1841, 1851, 1861, 1871, 1881, 1891, 1901, 1911, 1921, 1931,
    #          1941, 1951,
    #          1961, 1971, 1981, 1991, 2001, 2011]

    fig = plt.figure(constrained_layout=True, figsize=(4, 5))

    spec = gridspec.GridSpec(ncols=1, nrows=3, figure=fig)
    ax1 = fig.add_subplot(spec[0:2, 0])
    ax2 = fig.add_subplot(spec[2, 0])
    fig.patch.set_facecolor('w')
    ax1_2 = ax1.twinx()
    print(fig.get_figwidth(), fig.get_figheight())

    # y_anno = 0.02e7
    # yr_range = [[1831, 1881], [1881, 1951], [1951, 1991], [1991, 2011]]
    # yr_cols = ['darkgrey', 'lightskyblue', 'darkgrey', 'lightskyblue']
    # label_cols = ['dimgrey', 'deepskyblue', 'dimgrey', 'deepskyblue']
    # yr_names = ['R1', 'R2', 'R3', 'R4']

    # for i in np.arange(len(yr_cols)):
    #     #     ax1.fill_between(yr_range[i], 0, 1, color=yr_cols[i], edgecolor=None,
    #     #                     alpha=0.2, transform=ax.get_xaxis_transform())
    #
    #     ax1.annotate('', xy=(yr_range[i][0], y_anno), xycoords='data',
    #                  xytext=(yr_range[i][1], y_anno),
    #                  arrowprops=dict(arrowstyle="<->",
    #                                  color=label_cols[i], shrinkA=0, shrinkB=0))
    #     ax1.annotate('', xy=(yr_range[i][0], y_anno), xycoords='data',
    #                  xytext=(yr_range[i][1], y_anno),
    #                  arrowprops=dict(arrowstyle="|-|, widthA=0.3, widthB=0.3",
    #                                  color=label_cols[i], shrinkA=0, shrinkB=0))
    #     ax1.annotate(yr_names[i], xy=(np.mean(yr_range[i]), y_anno * 2),
    #                  xycoords='data', ha='center', color=label_cols[i],
    #                  fontsize=9)

    ax1.plot(sorted(list(pop_dict.keys())), N_pop, color=plas, lw=2)
    ax1_2.plot(sorted(list(ntwrk_yrs.keys())), N_stns, color=gre, lw=2)
    ax1.set_ylabel('Population (cap)', color=plas, fontdict={'size': 'small'})
    ax1.tick_params(axis='y', labelsize=8)
    ax1.set_ylim(ymin=0)
    ax1.set_xlim(1831, 2011)
    ax1_2.tick_params(axis='y', labelsize=8)
    ax1_2.set_ylabel('No. of stations', color=gre, fontdict={'size': 'small'})
    ax1_2.set_ylim(ymin=0)
    labels = ['Population', 'No. of stations']
    lineparams = zip(labels, [plas, gre])
    lines = [mlines.Line2D([], [], color=color, label=label)
             for label, color in lineparams]
    leg = ax1.legend(lines, labels, loc='upper left', fontsize='x-small')

    ax1.tick_params(axis='x', labelbottom=False)

    cent_area = pop_dict[2011][30:37, 39:46].shape[0] * \
                pop_dict[2011][30:37, 39:46].shape[1]
    sub_area = pop_dict[2011][30:43, 17:33].shape[0] * \
               pop_dict[2011][30:43, 17:33].shape[1]

    suburbs = []
    centre = []
    total_pop = []

    for yr in years:
        cent = np.nansum(pop_dict[yr][30:37, 39:46]) / cent_area
        # sub = np.nansum(rho_dict[yr][16:51, 17:68]) - cent
        sub = np.nansum(pop_dict[yr][30:43, 17:33]) / sub_area
        centre.append(cent)
        suburbs.append(sub)
        total_pop.append(np.nansum(pop_dict[yr]))
    suburbs = np.array(suburbs) / 1000
    centre = np.array(centre) / 1000

    #     ax2.patch.set_visible(False)
    #     # Set axtwin's patch visible and colorise it in grey
    #     twinax.patch.set_visible(True)
    #     # twinax.patch.set_facecolor('lightgrey')
    #     ax.set_zorder(twinax.get_zorder() + 1)
    ax2.plot(years, suburbs, color='r', lw=2, label='Suburban')
    ax2.plot(years, centre, color='orange', lw=2, label='Central')

    ax2.set_xlim(xmin=1831, xmax=2012)

    ax2.set_ylabel('Population density\n(1000 cap km$^{-2}$)', fontsize='small')

    ax2.set_xlabel('Year', fontsize='small')
    ax2.minorticks_on()
    ax1.set_xticks(years[::2])
    ax2.set_xticks(years[::2])
    ax2.set_xticklabels(years[::2])
    ax2.tick_params(axis='both', labelsize=8)
    ax2.sharex(ax1)

    labels = ['Central', 'Suburban']
    lineparams = zip(labels, ['orange', 'r'])
    lines = [mlines.Line2D([], [], color=color, label=label)
             for label, color in lineparams]
    leg = ax2.legend(lines, labels, loc='upper right', fontsize='x-small')
    # plt.subplots_adjust(hspace=0.15)
    # if savepath is not None:
    #     plt.savefig(savepath)
    plt.tight_layout()
    plt.savefig(
        './plots/paperplots/paper_concept_plots/totalpopulationstations.png',
        dpi=300)
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

        # ax.plot((0.25, 0.5), (4.9, 4.9), color='k', lw=1)
        # ax.plot((.5, .5), (4.9, 5.3), color='k', lw=1)
        # ax.annotate(r'$b$', xy=(.5, 4.9), xycoords='data', ha='left', va='top',
        #             color='k', fontsize=9)
        # if savepath is not None:
        #     pickle.dump({'Sim': simbeta, 'Obs': obsbeta, 'years': years},
        #                 open(savepath + '.p', 'wb'))

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


def plot_sim_obs_stncount(pop_obs, obs_ntwrk_yrs, sim_ntwrk_yrs, savepath=None):
    obs_years = np.array(list(nx.get_node_attributes(obs_ntwrk_yrs[2011],
                                                     'year').values()))
    sim_years = np.array(list(nx.get_node_attributes(sim_ntwrk_yrs[2011],
                                                     'year').values()))

    sim_stn_count = []
    obs_stn_count = []
    total_pop = []
    for yr in np.sort(list(pop_obs.keys())):
        sim_stn_count.append(np.count_nonzero(sim_years <= yr))
        obs_stn_count.append(np.count_nonzero(obs_years <= yr))
        total_pop.append(np.nansum(pop_obs[yr]))

    years = np.sort(list(pop_obs.keys()))
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('w')
    ax2 = ax.twinx()
    ax.plot(years, total_pop, color='purple', ls='--', label='$N_{pop}$')
    ax2.plot(years[1:], obs_stn_count[1:], color='blue', label='Obs. $N_{s}$')
    ax2.plot(years[1:], sim_stn_count[1:], color='r', label='Sim. $N_{s}$')
    ax.set_ylabel('Population within extent', color='purple')
    ax2.set_ylabel('No. of stations in extent', color='k')
    ax.set_xlabel('Year')
    labels = ["$N_{pop}$", 'Obs. $N_{s}$', 'Sim. $N_{s}$']
    lineparams = zip(labels, ['purple', 'b', 'r'], ['--', '-', '-'])
    lines = [mlines.Line2D([], [], color=color, label=label, ls=style)
             for label, color, style in lineparams]
    leg = ax2.legend(lines, labels, loc='upper left')
    if savepath is not None:
        plt.savefig(savepath, dpi=300)
    plt.show()


def plot_networtk_spatial_stats(obs_count_data, obs_pdf_data, obs_hierarchy,
                                sim_count_data, sim_pdf_data, sim_hierarchy,
                                yrs, savepath=None):
    fig, axes = plt.subplots(ncols=3, nrows=len(yrs), figsize=(12, 10))
    fig.patch.set_facecolor('w')
    for i, yr in enumerate(yrs):
        obs_hier = np.array([[k, obs_hierarchy[yr][k]]
                             for k in obs_hierarchy[yr]
                             if type(k) is int or type(k) is float or
                             type(k) is np.int32])
        sim_hier = np.array([[k, sim_hierarchy[yr][k]]
                             for k in sim_hierarchy[yr]
                             if type(k) is int or type(k) is float or
                             type(k) is np.int32])
        obs_root = obs_hierarchy[yr]['max_deg']
        sim_root = sim_hierarchy[yr]['max_deg']

        axes[i, 0].scatter(obs_hier[:, 0], obs_hier[:, 1], color='b',
                           label='Obs.')
        axes[i, 0].scatter(sim_hier[:, 0], sim_hier[:, 1], color='r',
                           label='Sim')

        marker_y = axes[i, 0].get_ylim()[1] * 0.9
        if obs_root == sim_root:
            obs_ms = 60
        else:
            obs_ms = 30

        axes[i, 0].scatter(obs_root, marker_y, color='b', marker='*',
                           s=obs_ms)  # , label='Obs. root)
        axes[i, 0].scatter(sim_root, marker_y, color='r', marker='*',
                           s=30)  # , label='Sim. root)
        axes[i, 0].set_ylabel(
            r"$\bf{" + str(yr) + "}$" + "\n\n" + r"$\bar{d}$ (km)")
        axes[i, 0].tick_params(labelbottom=False)

        axes[i, 1].scatter(obs_count_data[yr][0], obs_pdf_data[yr][0],
                           color='b', label='Obs.', zorder=2)
        axes[i, 1].scatter(sim_count_data[yr][0], sim_pdf_data[yr][0],
                           color='r', label='Sim', zorder=2)

        a, b = np.polyfit(xsim[1:], ysim[1:], 1)

        (coeffobs, expobs), pcov_log, ydatafit_lo = \
            curve_fit_log(obs_count_data[yr][0][1:],
                          obs_pdf_data[yr][0][1:])
        coeffobs = 10 ** coeffobs
        axes[i, 1].plot(obs_count_data[yr][0][1:],
                        (coeffobs) * obs_count_data[yr][0][1:] ** expobs,
                        color='royalblue', zorder=0)

        (coeffsim, expsim), pcov_log, ydatafit_lo = \
            curve_fit_log(sim_count_data[yr][0][1:],
                          sim_pdf_data[yr][0][1:])
        coeffsim = 10 ** coeffsim
        axes[i, 1].plot(sim_count_data[yr][0][1:],
                        (coeffsim) * sim_count_data[yr][0][1:] ** expsim,
                        color='firebrick', zorder=1)
        #         print(obs_count_data[yr][0][1:], sim_count_data[yr][0][1:])

        axes[i, 1].set_ylabel('PDF(deg)', fontsize='small')
        axes[i, 1].tick_params(labelbottom=False)
        axes[i, 1].set_yscale('log')
        axes[i, 1].set_xscale('log')
        axes[i, 1].text(0.1, 0.2,
                        f'$y={np.around(coeffsim, 1)}x^{{{np.around(expsim, 1)}}}$',
                        color='firebrick', transform=axes[i, 1].transAxes)
        axes[i, 1].text(0.1, 0.1,
                        f'$y={np.around(coeffobs, 1)}x^{{{np.around(expobs, 1)}}}$',
                        color='royalblue', transform=axes[i, 1].transAxes)
        #         axes[i, 1].annotate(yr, xy=(0.85, 0.9), xycoords='axes fraction',
        #                             fontsize=9)

        axes[i, 2].scatter(np.log10(obs_count_data[yr][0]),
                           np.log10(obs_pdf_data[yr][1]),
                           color='b', label='Obs.')
        axes[i, 2].scatter(np.log10(sim_count_data[yr][0]),
                           np.log10(sim_pdf_data[yr][1]),
                           color='r', label='Sim.')
        axes[i, 2].set_ylabel('log$_{10}$(CDF(deg))', fontsize='small')
        axes[i, 2].tick_params(labelbottom=False)
    #         axes[i, 2].annotate(yr, xy=(0.85, 0.9), xycoords='axes fraction',
    #                             fontsize=9)

    axes[len(yrs) - 1, 0].set_xlabel('Degree', fontsize='small')
    axes[len(yrs) - 1, 0].tick_params(labelbottom=True)
    axes[len(yrs) - 1, 1].set_xlabel('Degree', fontsize='small')
    axes[len(yrs) - 1, 1].tick_params(labelbottom=True)
    axes[len(yrs) - 1, 2].set_xlabel('log$_{10}$(Degree)',
                                     fontsize='small')
    axes[len(yrs) - 1, 2].tick_params(labelbottom=True)

    col0 = get_xlim_ylim(axes[:, 0])
    col1 = get_xlim_ylim(axes[:, 1])
    col2 = get_xlim_ylim(axes[:, 2])

    for ax in axes[:, 0]:
        ax.set_xlim(col0[0])
        ax.set_ylim(col0[1])

    for ax in axes[:, 1]:
        ax.set_xlim(col1[0])
        ax.set_ylim(col1[1])

    for ax in axes[:, 2]:
        ax.set_xlim(col2[0])
        ax.set_ylim(col2[1])

    #     for ax in axes[0, 1:]:
    #         ax.legend()
    labels_a = ['Obs.', 'Sim.', 'Root deg.']
    lineparams_a = zip(labels_a, ['b', 'r', 'grey'], ['.', '.', '*'])
    lines_a = [
        mlines.Line2D([], [], color=color, label=label, ls='', marker=m)
        for label, color, m in lineparams_a]

    leg_a = axes[0, 0].legend(lines_a, labels_a, loc='upper right')

    labels_b = ['Obs.', 'Sim.', '$ax^{b}$ fit']
    lineparams_b = zip(labels_b, ['b', 'r', 'grey'], ['.', '.', ''],
                       ['', '', '-'])
    lines_b = [
        mlines.Line2D([], [], color=color, label=label, ls=sty, marker=m)
        for label, color, m, sty in lineparams_b]

    leg_b = axes[0, 1].legend(lines_b, labels_b, loc='upper right')

    axes[0, 2].legend()

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=300)

    plt.show()


def plot_two_xy_mean_years_paper(rho_dict1, rho_dict2, years, xv, yv,
                                 title=None, cmaps=None, savepath=None,
                                 borders=None, bt_titles=None):
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
    :param bt_titles: list, numpy.ndarray, tuple
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
    clrs_x = [cm.get_cmap(cmaps[0])(c) for c in np.linspace(0, 1, len(years))]
    clrs_y = [cm.get_cmap(cmaps[-1])(c) for c in np.linspace(0, 1, len(years))]

    fig, (ax1, ax2) = \
        plt.subplots(nrows=2, figsize=(5, 4))
    fig.patch.set_facecolor('w')
    for i, yr in enumerate(years):
        arr1 = rho_dict1[yr]
        arr2 = rho_dict2[yr]
        if borders is not None:
            arr1 = trim_array(arr1, borders)
            arr2 = trim_array(arr2, borders)
        ax1.plot(xv[0, :], np.nanmean(arr1 / 1000, axis=0), color=clrs_x[i],
                 lw=1)
        ax1.plot(xv[0, :], np.nanmean(arr2 / 1000, axis=0), color=clrs_x[i],
                 ls='--', lw=1)

    ax1.set_xlim(xv[0, 0], xv[0, -1])
    ax1.set_xlabel(r'x-direction (km)', fontsize='small')
    # ax1.set_ylim(ymax= np.max([ax11.get_ylim()[1], ax21.get_ylim()[1]]))
    # ax.legend(fontsize='small')

    for i, yr in enumerate(years):
        arr1 = rho_dict1[yr]
        arr2 = rho_dict2[yr]
        if borders is not None:
            arr1 = trim_array(arr1, borders)
            arr2 = trim_array(arr2, borders)
        ax2.plot(yv[:, 0], np.nanmean(arr1 / 1000, axis=1), color=clrs_y[i],
                 lw=1)
        ax2.plot(yv[:, 0], np.nanmean(arr2 / 1000, axis=1), color=clrs_y[i],
                 ls='--', lw=1)

    ax2.set_xlim(yv[0, 0], yv[-1, 0])
    ax2.set_xlabel(r'y-direction (km)', fontsize='small')

    for ax in (ax1, ax2):
        ax.set_ylabel(r'Mean $\rho$ (10$^{3}$ cap km$^{-2}$)',
                      fontdict={'size': 8})

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    lines = [mlines.Line2D([], [], color=colour, label=label)
             for label, colour in zip(years, clrs_x)]
    leg = ax1.legend(lines, years, loc='upper left', frameon=False, ncol=2,
                     fontsize=8)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(1.5)

    len_diff = abs(len(xv[0, :]) - len(yv[:, 0]))

    # only works for current shape as x > y
    ax1.set_xlim(xv[0, :].min(), xv[0, :].max())
    ax2.set_xlim(yv[:, 0].min() - len_diff // 2, yv[:, 0].max() + len_diff // 2)

    ax1.text(0.1, 0.55, 'W', color='k', transform=ax1.transAxes)
    ax1.text(0.9, 0.55, 'E', color='k', transform=ax1.transAxes)
    ax2.text(0.1, 0.55, 'S', color='k', transform=ax2.transAxes)
    ax2.text(0.9, 0.55, 'N', color='k', transform=ax2.transAxes)

    if title is not None:
        fig.suptitle(title)

    if bt_titles is not None:
        if len(bt_titles) == 2:
            ax1.set_title(bt_titles[0], fontdict={'fontsize': 'small'})
            ax2.set_title(bt_titles[1], fontdict={'fontsize': 'small'})

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=300)
    plt.show()
    plt.close()


def observed_npop_nv(rho_dict, ntwrk, years, ybound, xbound, savepath=None):
    N_pop = []
    N_stns = []
    stns = []
    stn_count = 0
    for yr in sorted(list(rho_dict.keys())):
        N_pop.append(np.nansum(trim_array(rho_dict[yr], borders=(6, 6))))

    for yr in sorted(list(ntwrk.keys())):
        for n in list(ntwrk[yr].nodes):
            if n in stns:
                pass
            else:
                node = ntwrk[yr].nodes[n]
                if node['lat'] >= ybound[0] and node['lat'] <= ybound[1] and \
                        node['lon'] >= xbound[0] and node['lon'] <= xbound[1]:
                    stn_count += 1
                    stns.append(n)
        N_stns.append(stn_count)

    # POPULATION AND NO. OF STATIONS
    plas = cm.plasma(0.25)
    gre = cm.Greens(0.75)

    fig = plt.figure(constrained_layout=True, figsize=(4, 5))

    spec = gridspec.GridSpec(ncols=1, nrows=3, figure=fig)
    ax1 = fig.add_subplot(spec[0:2, 0])
    ax2 = fig.add_subplot(spec[2, 0])
    fig.patch.set_facecolor('w')
    ax1_2 = ax1.twinx()
    print(fig.get_figwidth(), fig.get_figheight())

    y_anno = 0.02e7
    yr_range = [[1831, 1891], [1891, 1951], [1951, 1991], [1991, 2011]]
    yr_cols = ['darkgrey', 'lightskyblue', 'lightskyblue', 'lightskyblue']
    label_cols = ['dimgrey', 'deepskyblue', 'deepskyblue', 'deepskyblue']
    yr_names = ['I', 'II.1', 'II.2', 'II.3']

    for i in np.arange(len(yr_cols)):
        ax1.annotate('', xy=(yr_range[i][0], y_anno), xycoords='data',
                     xytext=(yr_range[i][1], y_anno),
                     arrowprops=dict(arrowstyle="<->",
                                     color=label_cols[i], shrinkA=0, shrinkB=0))
        ax1.annotate('', xy=(yr_range[i][0], y_anno), xycoords='data',
                     xytext=(yr_range[i][1], y_anno),
                     arrowprops=dict(arrowstyle="|-|, widthA=0.3, widthB=0.3",
                                     color=label_cols[i], shrinkA=0, shrinkB=0))
        ax1.annotate(yr_names[i], xy=(np.mean(yr_range[i]), y_anno * 2),
                     xycoords='data', ha='center', color=label_cols[i],
                     fontsize=9)

    ax1.plot(years, N_pop, color=plas, lw=2)
    ax1_2.plot(years, N_stns, color=gre, lw=2)
    ax1.set_ylabel('$N_{pop}$ (cap)', color=plas, fontdict={'size': 'small'})
    ax1.tick_params(axis='y', labelsize=8)
    ax1.set_ylim(ymin=0)
    ax1.set_xlim(1831, 2011)
    ax1_2.tick_params(axis='y', labelsize=8)
    ax1_2.set_ylabel('$N_{v}$', color=gre, fontdict={'size': 'small'})
    ax1_2.set_ylim(ymin=0)
    labels = ['$N_{pop}$', '$N_{v}$']
    lineparams = zip(labels, [plas, gre])
    lines = [mlines.Line2D([], [], color=color, label=label)
             for label, color in lineparams]
    leg = ax1.legend(lines, labels, loc='upper left', fontsize=8)

    ax1.tick_params(axis='x', labelbottom=False)

    cent_area = rho_dict[2011][30:37, 39:46].shape[0] * \
                rho_dict[2011][30:37, 39:46].shape[1]
    sub_area = rho_dict[2011][30:43, 17:33].shape[0] * \
               rho_dict[2011][30:43, 17:33].shape[1]

    suburbs = []
    centre = []
    total_pop = []

    for yr in years:
        cent = np.nansum(rho_dict[yr][30:37, 39:46]) / cent_area
        sub = np.nansum(rho_dict[yr][30:43, 17:33]) / sub_area
        centre.append(cent)
        suburbs.append(sub)
        total_pop.append(np.nansum(rho_dict[yr]))
    suburbs = np.array(suburbs) / 1000
    centre = np.array(centre) / 1000

    ax2.plot(years, suburbs, color='r', lw=2, label='Suburban')
    ax2.plot(years, centre, color='orange', lw=2, label='Central')

    ax2.set_xlim(xmin=1831, xmax=2012)

    ax2.set_ylabel(r'$\rho$ (1000 cap km$^{-2}$)', fontsize='small')

    ax2.set_xlabel('Year', fontsize='small')
    ax2.minorticks_on()
    ax1.set_xticks(years[::2])
    ax2.set_xticks(years[::2])
    ax2.set_xticklabels(years[::2])
    ax2.tick_params(axis='both', labelsize=8)
    ax2.sharex(ax1)

    labels = ['Central', 'Suburban']
    lineparams = zip(labels, ['orange', 'r'])
    lines = [mlines.Line2D([], [], color=color, label=label)
            for label, color in lineparams]
    leg = ax2.legend(lines, labels, loc='upper right', fontsize=8)
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=300)
    plt.show()

def observed_npop_nv_small(rho_dict, ntwrk, years, ybound, xbound,
                           centre, suburbs, savepath=None):

    N_pop = []
    N_stns = []
    stns = []
    stn_count = 0
    for yr in years:
        N_pop.append(np.nansum(trim_array(rho_dict[yr], borders=(6,6))))


    for yr in years:
        if yr not in ntwrk.keys():
            N_stns.append(None)
        else:
            for n in list(ntwrk[yr].nodes):
                if n in stns:
                    pass
                else:
                    node = ntwrk[yr].nodes[n]
                    if node['lat'] >= ybound[0] and node['lat'] <= ybound[1] and \
                        node['lon'] >= xbound[0] and node['lon'] <= xbound[1]:
                        stn_count += 1
                        stns.append(n)
            N_stns.append(stn_count)

    # POPULATION AND NO. OF STATIONS
    plas = cm.plasma(0.25)
    gre = cm.Greens(0.75)

    fig = plt.figure(constrained_layout=True, figsize=(4,2.7))

    spec = gridspec.GridSpec(ncols=1, nrows=2, figure=fig)
    ax1 = fig.add_subplot(spec[0:1, 0])
    ax2 = fig.add_subplot(spec[1, 0])
    fig.patch.set_facecolor('w')
    ax1_2 = ax1.twinx()
    print(fig.get_figwidth(), fig.get_figheight())

    y_anno = 0.04e7
    yr_range = [[1831, 1891], [1891, 1951], [1951, 1991], [1991, 2011]]
    yr_cols = ['darkgrey', 'lightskyblue', 'lightskyblue', 'lightskyblue']
    label_cols = ['dimgrey', 'deepskyblue', 'deepskyblue', 'deepskyblue']
    yr_names = ['I', 'II.1', 'II.2', 'II.3']

    for i in np.arange(len(yr_cols)):
        ax1.annotate('', xy=(yr_range[i][0], y_anno), xycoords='data',
                    xytext=(yr_range[i][1], y_anno),
                    arrowprops=dict(arrowstyle="<->",
                                    color=label_cols[i], shrinkA=0, shrinkB=0))
        ax1.annotate('', xy=(yr_range[i][0], y_anno), xycoords='data',
                    xytext=(yr_range[i][1], y_anno),
                    arrowprops=dict(arrowstyle="|-|, widthA=0.3, widthB=0.3",
                                    color=label_cols[i], shrinkA=0, shrinkB=0))
        ax1.annotate(yr_names[i], xy=(np.mean(yr_range[i]), y_anno*2),
                     xycoords='data', ha='center', color=label_cols[i],
                     fontsize=9)

    ax1.plot(years, N_pop, color=plas, lw=2)
    ax1_2.plot(years, N_stns, color=gre, lw=2)
    ax1.set_ylabel('$N_{pop}$ (cap)', color=plas, fontdict={'size': 'small'})
    ax1.tick_params(axis='y', labelsize=8)
    ax1.set_ylim(ymin=0)
    ax1.set_xlim(1831, 2011)
    ax1_2.tick_params(axis='y', labelsize=8)
    ax1_2.set_ylabel('$N_{v}$', color=gre, fontdict={'size': 'small'})
    ax1_2.set_ylim(ymin=0)
    labels = ['$N_{pop}$', '$N_{v}$']
    lineparams = zip(labels, [plas, gre])
    lines = [mlines.Line2D([], [], color=color, label=label)
            for label, color in lineparams]
    leg = ax1.legend(lines, labels, loc='upper left', fontsize=8)

    ax1.tick_params(axis='x', labelbottom=False)

    cent_area = rho_dict[2011][centre[0]:centre[1],
                               centre[2]:centre[3]].shape[0] *\
                rho_dict[2011][centre[0]:centre[1],
                               centre[2]:centre[3]].shape[1]
    sub_area = rho_dict[2011][suburbs[0]:suburbs[1],
                              suburbs[2]:suburbs[3]].shape[0] * \
                rho_dict[2011][suburbs[0]:suburbs[1],
                               suburbs[2]:suburbs[3]].shape[1]

    suburbs_pop = []
    centre_pop = []
    total_pop = []
    print(type(rho_dict[2011]))
    for yr in years:
        print(yr)
        cent = np.nansum(rho_dict[yr][centre[0]:centre[1],
                                      centre[2]:centre[3]]) / cent_area
        sub = np.nansum(rho_dict[yr][suburbs[0]:suburbs[1],
                                     suburbs[2]:suburbs[3]]) / sub_area
        centre_pop.append(cent)
        suburbs_pop.append(sub)
        total_pop.append(np.nansum(rho_dict[yr]))
    suburbs_pop = np.array(suburbs_pop)/1000
    centre_pop = np.array(centre_pop)/1000

    ax2.plot(years, suburbs_pop, color='r', lw=2, label='Suburban')
    ax2.plot(years, centre_pop, color='orange', lw=2, label='Central')

    ax2.set_xlim(xmin=1831, xmax=2012)

    ax2.set_ylabel(r'$\rho$ (1000 cap km$^{-2}$)', fontsize='small')

    ax2.set_xlabel('Year', fontsize='small')
    ax2.minorticks_on()
    ax1.set_xticks(years[::2])
    ax2.set_xticks(years[::2])
    ax2.set_xticklabels(years[::2])
    ax2.tick_params(axis='both', labelsize=8)
    ax2.sharex(ax1)

    labels = ['Central', 'Suburban']
    lineparams = zip(labels, ['orange', 'r'])
    lines = [mlines.Line2D([], [], color=color, label=label)
             for label, color in lineparams]
    leg = ax2.legend(lines, labels, loc='upper right', fontsize=8)
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=300)
    plt.show()

def observed_npop_nv_sydney(rho_dict, ntwrk, years, ybound, xbound,
                            centre, suburbs, savepath=None):

    N_pop = []
    N_stns = []
    open_stns = []
    close_stns = []
    stn_count = 0
    for yr in years:
        N_pop.append(np.nansum(trim_array(rho_dict[yr], borders=(6,6))))

    for i, yr in enumerate(years):
        if yr in ntwrk.keys():
            stn_count = 0
            for n in list(ntwrk[yr].nodes):
                node = ntwrk[yr].nodes[n]
                if node['lat']/1000 >= ybound[0] and \
                node['lat']/1000 <= ybound[1] and \
                    node['lon']/1000 >= xbound[0] and \
                node['lon']/1000 <= xbound[1] and node['close'] >= yr:
                    stn_count += 1
                    # open_stns.append(n)
                pass
            N_stns.append(stn_count)
        else:
            N_stns.append(None)

    # POPULATION AND NO. OF STATIONS
    plas = cm.plasma(0.25)
    gre = cm.Greens(0.75)

    fig = plt.figure(constrained_layout=True, figsize=(4,2.7))

    spec = gridspec.GridSpec(ncols=1, nrows=2, figure=fig)
    ax1 = fig.add_subplot(spec[0:1, 0])
    ax2 = fig.add_subplot(spec[1, 0])
    fig.patch.set_facecolor('w')
    ax1_2 = ax1.twinx()
    print(fig.get_figwidth(), fig.get_figheight())

    y_anno = 0.02e7
    yr_range = [[1851, 1881], [1881, 2011]]
    yr_cols = ['darkgrey', 'lightskyblue']
    label_cols = ['dimgrey', 'deepskyblue']
    yr_names = ['I', 'II.1']

    for i in np.arange(len(yr_cols)):
        ax1.annotate('', xy=(yr_range[i][0], y_anno), xycoords='data',
                    xytext=(yr_range[i][1], y_anno),
                    arrowprops=dict(arrowstyle="<->",
                                    color=label_cols[i], shrinkA=0, shrinkB=0))
        ax1.annotate('', xy=(yr_range[i][0], y_anno), xycoords='data',
                    xytext=(yr_range[i][1], y_anno),
                    arrowprops=dict(arrowstyle="|-|, widthA=0.3, widthB=0.3",
                                    color=label_cols[i], shrinkA=0, shrinkB=0))
        ax1.annotate(yr_names[i], xy=(np.mean(yr_range[i]), y_anno*2),
                     xycoords='data', ha='center', color=label_cols[i],
                     fontsize=9)

    ax1.plot(years, N_pop, color=plas, lw=2)
    ax1_2.plot(years, N_stns, color=gre, lw=2)
    ax1.set_ylabel('$N_{pop}$ (cap)', color=plas, fontdict={'size': 'small'})
    ax1.tick_params(axis='y', labelsize=8)
    ax1.set_ylim(ymin=0)
    ax1.set_xlim(1851, 2011)
    ax1_2.tick_params(axis='y', labelsize=8)
    ax1_2.set_ylabel('$N_{v}$', color=gre, fontdict={'size': 'small'})
    ax1_2.set_ylim(ymin=0)
    labels = ['$N_{pop}$', '$N_{v}$']
    lineparams = zip(labels, [plas, gre])
    lines = [mlines.Line2D([], [], color=color, label=label)
            for label, color in lineparams]
    leg = ax1.legend(lines, labels, loc='upper left', fontsize=8)

    ax1.tick_params(axis='x', labelbottom=False)

    cent_area = rho_dict[2011][centre[0]:centre[1],
                               centre[2]:centre[3]].shape[0] *\
                rho_dict[2011][centre[0]:centre[1],
                               centre[2]:centre[3]].shape[1]
    sub_area = rho_dict[2011][suburbs[0]:suburbs[1],
                              suburbs[2]:suburbs[3]].shape[0] * \
                rho_dict[2011][suburbs[0]:suburbs[1],
                               suburbs[2]:suburbs[3]].shape[1]

    suburbs_pop = []
    centre_pop = []
    total_pop = []
    print(type(rho_dict[2011]))
    for yr in years:
        print(yr)
        cent = np.nansum(rho_dict[yr][centre[0]:centre[1],
                                      centre[2]:centre[3]]) / cent_area
        sub = np.nansum(rho_dict[yr][suburbs[0]:suburbs[1],
                                     suburbs[2]:suburbs[3]]) / sub_area
        centre_pop.append(cent)
        suburbs_pop.append(sub)
        total_pop.append(np.nansum(rho_dict[yr]))
    suburbs_pop = np.array(suburbs_pop)/1000
    centre_pop = np.array(centre_pop)/1000

    ax2.plot(years, suburbs_pop, color='r', lw=2, label='Suburban')
    ax2.plot(years, centre_pop, color='orange', lw=2, label='Central')

    ax2.set_xlim(xmin=1851, xmax=2012)

    ax2.set_ylabel(r'$\rho$ (1000 cap km$^{-2}$)', fontsize='small')

    ax2.set_xlabel('Year', fontsize='small')
    ax2.minorticks_on()
    ax1.set_xticks(years[::2])
    ax2.set_xticks(years[::2])
    ax2.set_xticklabels(years[::2])
    ax2.tick_params(axis='both', labelsize=8)
    ax2.sharex(ax1)

    labels = ['Central', 'Suburban']
    lineparams = zip(labels, ['orange', 'r'])
    lines = [mlines.Line2D([], [], color=color, label=label)
            for label, color in lineparams]
    leg = ax2.legend(lines, labels, loc='lower right', fontsize=8)
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=300)
    plt.show()

def obs_rho_network_twoyears(pop_grid, years, ntwrk, vmin=1000, tickspace=10,
                             realign=False, xv=None, yv=None, xlim=None,
                             ylim=None, savepath=None):
    # DOMAIN WITH POPULATION AND NETWORK
    yr1, yr2 = years

    vmax = max(pop_grid[f'{yr1}_pop'].max(), pop_grid[f'{yr2}_pop'].max())

    xmin, ymin, xmax, ymax = pop_grid.total_bounds

    arrow_font = {'fontsize': 'xx-large', 'ha': 'center', 'va': 'center'}

    cmap_rho = cm.get_cmap('plasma')
    cmap_rho.set_under('ghostwhite')
    props = dict(boxstyle='round', facecolor='w', alpha=0.9)

    fig, (ax1, ax2) = plt.subplots(nrows=2, constrained_layout=True,
                                   figsize=(4,8))
    fig.patch.set_facecolor('w')

    ax1.plot([xmin, xmin + 10], [ymin, ymin + 10], color='w')

    pg1 = pop_grid.plot(column=f'{yr1}_pop', ax=ax1, cmap=cmap_rho, vmin=vmin,
                        vmax=vmax)

    orig_cmap = cm.Greens
    shifted_cmap = shiftedColorMap(cmap=orig_cmap, start=0.15,
                                   midpoint=0.15 + (1 - 0.15) / 2, stop=1)
    a_i1 = list(nx.get_node_attributes(ntwrk[yr1], 'a_i_inv').values())

    nx.draw_networkx(ntwrk[yr1], with_labels=False, ax=ax1,
                    pos=nx.get_node_attributes(ntwrk[yr1], 'pos'),
                    node_size=5, cmap=shifted_cmap, edge_color='dimgrey',
                    width=0.5, node_color=a_i1)# , zorder=rect.get_zorder() + 1)
    ax1.text(0.05, 0.95, yr1, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)


    ax2.plot([xmin, xmin + 10], [ymin, ymin + 10], color='w')
    pg2 = pop_grid.plot(column=f'{yr2}_pop', ax=ax2, cmap='plasma', vmin=vmin,
                        vmax=vmax)
    orig_cmap = cm.Greens
    shifted_cmap = shiftedColorMap(cmap=orig_cmap, start=0.15,
                                   midpoint=0.15 + (1 - 0.15) / 2, stop=1)
    a_i2 = list(nx.get_node_attributes(ntwrk[yr2], 'a_i_inv').values())

    nx.draw_networkx(ntwrk[yr2], with_labels=False, ax=ax2,
                    pos=nx.get_node_attributes(ntwrk[yr2], 'pos'),
                    node_size=5, cmap=shifted_cmap, edge_color='dimgrey',
                    width=0.5, node_color=a_i2)#, zorder=rect.get_zorder() + 1)
    ax2.text(0.05, 0.95, yr2, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

    brd = 6000
    for ax in (ax1, ax2):
        print(ax.get_xlim())
        print(ax.get_ylim())
        lines = [mlines.Line2D([0], [0], markerfacecolor=shifted_cmap(0.75),
                            label='Station', ls='', marker='o', markersize=10,
                            markeredgecolor=shifted_cmap(0.75)),
                mlines.Line2D([0], [0], label='Edge', ls='-', color='grey')]

        leg = ax1.legend(handles=lines, loc='upper right',
                         prop=dict(size='medium'), labels=['$V$', '$E$'])

        ax.set_frame_on(False)

        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        ax.set_aspect('equal')

        if xlim is not None and ylim is not None:
            ax.set_xlim(xlim[0], xlim[1])
            ax.set_ylim(ylim[0], ylim[1])
        else:
            ax.set_xlim(xmin+brd, xmax-brd)
            ax.set_ylim(ymin+brd, ymax-brd)

    if xv is not None and yv is not None:
        xv = xv[0,:]
        x_neg_start = xv[int(brd/1000)]
        x_neg_stop, x_pos_start = 0, 0
        x_pos_stop = xv[-int(brd/1000)]

        xax = np.concatenate(
            (-1* np.arange(
                x_neg_stop, abs(x_neg_start), tickspace)[::-1].astype(int),
                np.arange(x_pos_start, x_pos_stop, tickspace).astype(int)[1:]))


        yv = yv[:,0]
        y_neg_start = yv[int(brd/1000)]
        y_neg_stop, y_pos_start = 0, 0
        y_pos_stop = yv[-int(brd/1000)]
        yax = np.concatenate(
            (-1* np.arange(
                y_neg_stop, abs(y_neg_start), tickspace)[::-1].astype(int),
                np.arange(y_pos_start, y_pos_stop, tickspace).astype(int)[1:]))

        if xlim is not None and ylim is not None:
            xtick = np.arange(xlim[0], xlim[1], 1000)[
                np.where(np.in1d(xv[int(brd/1000):-int(brd/1000)+1], xax))]
            ytick = np.arange(ylim[0], ylim[1], 1000)[
                np.where(np.in1d(yv[int(brd/1000):-int(brd/1000)+1], yax))]

        else:
            xtick = np.arange(xmin+brd, xmax-brd +1, 1000)[
                np.where(np.in1d(xv[int(brd/1000):-int(brd/1000)+1], xax))]
            ytick = np.arange(ymin+brd, ymax-brd +1, 1000)[
                np.where(np.in1d(yv[int(brd/1000):-int(brd/1000)+1], yax))]

        for ax in (ax1, ax2):
            ax.set_xticks(xtick)
            ax.set_xticklabels(xax, fontsize=8)
            ax.set_yticks(ytick)
            ax.set_yticklabels(yax, fontsize=8)
            ax.set_xlabel("$x$ (km)", fontsize=8)
            ax.set_ylabel("$y$ (km)", fontsize=8)

    else:
        for ax in (ax1, ax2):
            ax.set_xticks(np.arange(xmin+brd, xmax-brd +1, 10000))
            ax.set_yticks(np.arange(ymin+brd, ymax-brd +1, 10000))

            ax.set_xticklabels(np.arange(0, ((xmax-brd)/1000-(xmin+brd)/1000)+1, 10).astype(int))
            ax.set_yticklabels(np.arange(0, ((ymax-brd)/1000-(ymin+brd)/1000)+1, 10).astype(int))

        # else:
        # ax.set_xticklabels(
        #     np.arange(0, (xmax-brd)/1000-(xmin+brd)/1000+1,
        #             tickspace).astype(int))
        # ax.set_yticklabels(
        #     np.arange(0, (ymax-brd)/1000-(ymin+brd)/1000+1,
        #             tickspace).astype(int))

    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes(size="3%", pad=0.3, position='top')
    # norm = Normalize(vmin=vmin/1000, vmax=vmax/1000)
    # n_cmap = cm.ScalarMappable(norm=norm, cmap=cmap_rho)
    # n_cmap.set_array([])
    # cbar=fig.colorbar(n_cmap, cax=cax, label=r'$\rho$ (1000 cap km$^{-2}$)',
    #                 orientation='horizontal', extend='min')
    # cbar.ax.xaxis.set_ticks([vmin/1000] + list(cbar.get_ticks()))
    # # cbar.set_under('whitesmoke')
    # cbar.ax.xaxis.set_ticks_position('top')
    # cbar.ax.xaxis.set_label_position('top')
    # cbar.ax.tick_params(labelsize=8)
    # cbar.ax.set_xlabel(r'$\rho$ (1000 cap km$^{-2}$)', fontsize=8)

    # divider2 = make_axes_locatable(ax)
    # cax2 = divider2.append_axes(position='bottom', size="3%", pad=0.3)
    # norm2 = Normalize(vmin=np.min([np.min(a_i1), np.min(a_i2)]),
    #                   vmax=np.max([np.max(a_i1), np.max(a_i2)]))
    # n_cmap2 = cm.ScalarMappable(norm=norm2, cmap=shifted_cmap)
    # n_cmap2.set_array([])
    # cbar2=fig.colorbar(n_cmap2, cax=cax2, label=r'$A_{v}$',
    #                    orientation='horizontal')
    # cbar2.ax.tick_params(labelsize=8)
    # cbar2.ax.set_xlabel("$A_{v}$", fontsize=8)

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=300, bbox_inches = 'tight')
    plt.show()

def obs_rho_network_twoyears_array(rho_arr, years, ntwrk, xv, yv, vmin=1000,
                                   savepath=None, cbar_side='top'):
    # DOMAIN WITH POPULATION AND NETWORK
    yr1, yr2 = years

    vmax = max(np.nanmax(rho_arr[yr1]), np.nanmax(rho_arr[yr2]))

    arrow_font = {'fontsize': 'xx-large', 'ha':'center', 'va':'center'}

    cmap_rho = cm.get_cmap('plasma')
    cmap_rho.set_under('ghostwhite')
    props = dict(boxstyle='round', facecolor='w', alpha=0.9)

    fig, (ax1, ax2) = plt.subplots(nrows=2, constrained_layout=True,
                                   figsize=(4,8))
    fig.patch.set_facecolor('w')

    ax1.pcolor(xv, yv, rho_arr[yr1], cmap=cmap_rho, vmin=vmin, vmax=vmax)

    orig_cmap = cm.Greens
    shifted_cmap = shiftedColorMap(cmap=orig_cmap, start=0.15,
                                midpoint=0.15 + (1 - 0.15)/2, stop=1)
    a_i1 = list(nx.get_node_attributes(ntwrk[yr1], 'a_i_inv').values())

    nx.draw_networkx(ntwrk[yr1], with_labels=False, ax=ax1,
                    pos=nx.get_node_attributes(ntwrk[yr1], 'pos'),
                    node_size=5, cmap=shifted_cmap, edge_color='dimgrey',
                    width=0.5, node_color=a_i1)# , zorder=rect.get_zorder() + 1)
    ax1.text(0.05, 0.95, yr1, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)


    ax2.pcolor(xv, yv, rho_arr[yr2], cmap=cmap_rho, vmin=vmin, vmax=vmax)

    orig_cmap = cm.Greens
    shifted_cmap = shiftedColorMap(cmap=orig_cmap, start=0.15,
                                midpoint=0.15 + (1 - 0.15)/2, stop=1)
    a_i2 = list(nx.get_node_attributes(ntwrk[yr2], 'a_i_inv').values())

    nx.draw_networkx(ntwrk[yr2], with_labels=False, ax=ax2,
                    pos=nx.get_node_attributes(ntwrk[yr2], 'pos'),
                    node_size=5, cmap=shifted_cmap, edge_color='dimgrey',
                    width=0.5, node_color=a_i2)#, zorder=rect.get_zorder() + 1)
    ax2.text(0.05, 0.95, yr2, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

    brd = 6000
    for ax in (ax1, ax2):
        lines = [mlines.Line2D([0], [0], markerfacecolor=shifted_cmap(0.75),
                            label='Station', ls='', marker='o', markersize=10,
                            markeredgecolor=shifted_cmap(0.75)),
                mlines.Line2D([0], [0], label='Edge', ls='-', color='grey')]

        leg = ax1.legend(handles=lines, loc='upper right',
                         prop=dict(size='large'), labels=['$V$', '$E$'])

        ax.set_frame_on(False)

        ax.tick_params(axis='both', labelsize=8)

        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        ax.set_aspect('equal')

        ax.set_xlabel('$x$ (km)', fontsize=8)
        ax.set_ylabel('$y$ (km)', fontsize=8)

    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes(size="3%", pad=0.3, position=cbar_side)
    # norm = Normalize(vmin=vmin/1000, vmax=vmax/1000)
    # n_cmap = cm.ScalarMappable(norm=norm, cmap=cmap_rho)
    # n_cmap.set_array([])
    # if cbar_side in ('top', 'bottom'):
    #     orient = 'horizontal'
    # else:
    #     orient = 'vertical'
    # cbar=fig.colorbar(n_cmap, cax=cax, label=r'$\rho$ (1000 cap km$^{-2}$)',
    #                 orientation=orient, extend='min')
    # if cbar_side in ('top', 'bottom'):
    #     cbar.ax.xaxis.set_ticks([vmin/1000] + list(cbar.get_ticks()))
    # else:
    #     cbar.ax.yaxis.set_ticks([vmin/1000] + list(cbar.get_ticks()))
    # cbar.ax.tick_params(labelsize=8)

    # divider2 = make_axes_locatable(ax)
    # cax2 = divider2.append_axes(position=cbar_side, size="3%", pad=0.3)
    # norm2 = Normalize(vmin=np.min([np.min(a_i1), np.min(a_i2)]),
    #                   vmax=np.max([np.max(a_i1), np.max(a_i2)]))
    # n_cmap2 = cm.ScalarMappable(norm=norm2, cmap=shifted_cmap)
    # n_cmap2.set_array([])
    # if cbar_side in ('top', 'bottom'):
    #     orient = 'horizontal'
    # else:
    #     orient = 'vertical'
    # cbar2=fig.colorbar(n_cmap2, cax=cax2, label=r'$A_{v}$',
    #                    orientation=orient)
    # cbar2.ax.tick_params(labelsize=8)

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=300, bbox_inches = 'tight')
    plt.show()

def model_concept_rho_ntwrk(pop_grid, ntwrk, cbar, savepath=None):
    # %matplotlib qt
    # CONCEPT OF MODEL
    yr = 1971
    _xmin = 526000
    _xmax = 530000
    _ymin = 170000
    _ymax = 174000

    pop_patch = pop_grid[((pop_grid.geometry.bounds.minx >= _xmin) &
                          (pop_grid.geometry.bounds.maxx <= _xmax) &
                          (pop_grid.geometry.bounds.miny >= _ymin) &
                          (pop_grid.geometry.bounds.maxy <= _ymax))]
    vmax_pop = pop_patch[f'{yr}_pop'].max()
    vmin_pop = pop_patch[f'{yr}_pop'].min()

    arrow_font = {'fontsize': 'xx-large', 'ha': 'center', 'va': 'center'}
    yr_graph = nx.Graph(ntwrk[yr])
    yr_graph.remove_edge(9, 182)
    patch_stns = []
    ai_min = 1.2
    ai_max = 1.2
    # a_i={}
    for n in list(yr_graph.nodes):
        node = yr_graph.nodes[n]
        #     a_i[n] = node['a_i_inv']
        if node['lat'] >= _ymin / 1e3 and node['lat'] <= _ymax / 1e3 and \
                node['lon'] >= _xmin / 1e3 and node['lon'] <= _xmax / 1e3:
            patch_stns.append(n)
            print(node['a_i_inv'])
            if node['a_i_inv'] < ai_min:
                ai_min = node['a_i_inv']
            if node['a_i_inv'] > ai_max:
                ai_max = node['a_i_inv']
    print(ai_min, ai_max)
    a_i = list(nx.get_node_attributes(yr_graph, 'a_i_inv').values())
    orig_cmap = cm.Greens
    cmap_green = shiftedColorMap(cmap=orig_cmap, start=0.15,
                                 midpoint=0.15 + (1 - 0.15) / 2, stop=1,
                                 name='sh_green')

    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor('w')
    pop_grid.plot(column=f'{yr}_pop', ax=ax, cmap='plasma', vmax=vmax_pop,
                  vmin=vmin_pop)
    nx.draw_networkx(yr_graph, with_labels=False, ax=ax,
                     pos=nx.get_node_attributes(yr_graph, 'pos'),
                     cmap=cmap_green, node_color=a_i, vmin=ai_min, vmax=ai_max,
                     edge_color='grey', width=2)
    ax.text(528500, 171000, s=r'$\longleftarrow$', rotation=90, **arrow_font)
    ax.text(528500, 172000, s=r'$\longleftarrow$', rotation=270, **arrow_font)
    ax.text(528000, 171500, s=r'$\longleftarrow$', **arrow_font)
    ax.text(529000, 171500, s=r'$\longrightarrow$', **arrow_font)
    ax.text(528500, 171500, s='$\\rho$(X)', ha='center', va='center',
            fontsize=12)

    ax.annotate('', xy=(0, -0.03), xycoords='axes fraction',
        xytext=(0.25, -0.03), arrowprops=dict(arrowstyle="<->", color='k',
        shrinkA=0, shrinkB=0))
    ax.annotate('', xy=(0, -0.03), xycoords='axes fraction',
        xytext=(0.25, -0.03), color='k',
        arrowprops=dict(arrowstyle="|-|, widthA=0.5, widthB=0.5",
                        shrinkA=0, shrinkB=0))
    ax.annotate('$dx$', xy=(0.125, -0.07), xycoords='axes fraction',
                ha='center')
    ax.annotate('', xy=(-0.03, 0), xycoords='axes fraction',
                xytext=(-0.03, 0.25), arrowprops=dict(arrowstyle="<->",
                color='k', shrinkA=0, shrinkB=0))
    ax.annotate('', xy=(-0.03, 0), xycoords='axes fraction',
                xytext=(-0.03, 0.25),
                arrowprops=dict(arrowstyle="|-|, widthA=0.5, widthB=0.5",
                                color='k', shrinkA=0, shrinkB=0))
    ax.annotate('$dy$', xy=(-0.07, 0.125), xycoords='axes fraction',
                ha='center')

    if cbar == 'rho':
        sm2 = cm.ScalarMappable(cmap='plasma',
                                norm=plt.Normalize(vmin=vmin_pop,
                                                   vmax=vmax_pop))
        sm2._A = []
        cbar = plt.colorbar(sm2, label=r'$\rho$ (cap km$^{-1}$)')

        savepath = "./plots/paperplots/paper_concept_plots/modellayers_rho.png"
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.2)
        norm = Normalize(vmin=vmin_pop, vmax=vmax_pop)
        n_cmap = cm.ScalarMappable(norm=norm, cmap='plasma')
        n_cmap.set_array([])
        cbar = fig.colorbar(n_cmap, cax=cax, label=r'$\rho$ (cap km$^{-2}$)')
        cax.set_frame_on(False)
        cbar.set_ticks([])
    elif cbar == 'av':
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.2)
        norm = Normalize(vmin=ai_min, vmax=ai_max)
        n_cmap = cm.ScalarMappable(norm=norm, cmap=cmap_green)
        n_cmap.set_array([])
        cbar = fig.colorbar(n_cmap, cax=cax, label=r'$A_{v}$')
        cax.set_frame_on(False)
        cbar.set_ticks([])

    lines = [
        mlines.Line2D([0], [0], markerfacecolor='#15B01A', label='$V$', ls='',
                      marker='o', ms=8, markeredgecolor='#15B01A'),
        mlines.Line2D([0], [0], label='$E$', ls='-', color='grey')]

    leg = ax.legend(handles=lines, loc='upper right', prop=dict(size='large'),
                    labels=['$V$', '$E$'])

    ax.set_xlim(_xmin, _xmax)
    ax.set_ylim(_ymin, _ymax)
    ax.set_frame_on(False)

    if savepath is not None:
        plt.savefig(savepath,dpi=300)
    plt.show()

def npop_nv_simobs_comparison(rho_obs, rho_sim, ntwrk_obs, ntwrk_sim,
                              stn_count=None, savepath=None):
    obs_years = np.array(list(nx.get_node_attributes(ntwrk_obs[2011],
                                                    'year').values()))
    if ntwrk_sim is not None:
        sim_years = np.array(list(nx.get_node_attributes(ntwrk_sim[2011],
                                                        'year').values()))
    sim_stn_count = []
    obs_stn_count = []
    total_obs_pop = []
    total_sim_pop = []

    for yr in np.sort(list(rho_obs.keys())):
        if ntwrk_sim is not None:
            sim_stn_count.append(np.count_nonzero(sim_years <= yr))
        obs_stn_count.append(np.count_nonzero(obs_years <= yr))
        total_obs_pop.append(np.nansum(trim_array(rho_obs[yr], (6,6))))
        total_sim_pop.append(np.nansum(trim_array(rho_sim[yr], (6,6))))

    if stn_count is not None:
        obs_stn_count = []
        for i, yr in enumerate(np.sort(list(rho_obs.keys()))):
            if yr in ntwrk_obs.keys():
                stn_count = 0
                for n in list(ntwrk_obs[yr].nodes):
                    node = ntwrk_obs[yr].nodes[n]
                    if node['close'] >= yr:
                        stn_count += 1
                        # open_stns.append(n)
                    pass
                obs_stn_count.append(stn_count)
            else:
                obs_stn_count.append(None)

    years = np.sort(list(rho_obs.keys()))
    cmap = plt.get_cmap('copper')#, np.max(years[1:]) - np.min(years[1:]) + 1)
    norm = colors.BoundaryNorm(np.arange(np.min(years[1:]) - 5,
                            np.max(years[1:]) + 11, 10), cmap.N)
    fig, ax = plt.subplots(figsize=(5,4))
    fig.patch.set_facecolor('w')

    ax.plot(total_obs_pop[1:], obs_stn_count[1:], ls=':', color='red', lw=2)
    ax.plot(total_sim_pop[1:], obs_stn_count[1:], ls='-', color='grey')
    if ntwrk_sim is not None:
        ax.plot(total_obs_pop[1:], sim_stn_count[1:], ls='--', color='grey')

    popstn = ax.scatter(x=total_sim_pop[1:], y=obs_stn_count[1:], c=years[1:],
                        marker='^', cmap='copper', zorder=3, norm=norm)

    if ntwrk_sim is not None:
        ax.scatter(x=total_obs_pop[1:], y=sim_stn_count[1:],
                            c=years[1:], cmap='copper', zorder=3, norm=norm)

    ax.set_xlabel('$N_{pop}$ (cap)')
    ax.set_ylabel('$N_{v}$')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    cbar = fig.colorbar(popstn, ticks=np.arange(np.min(years[1:]),
                        np.max(years[1:]) + 21, 10), label='Year',)
    cbar.ax.tick_params(labelsize=9)

    if ntwrk_sim is not None:
        labels = ["Obs. $N_{pop}$, Sim. $N_{v}$", "Sim. $N_{pop}$, Obs. $N_{v}$",
                "Obs. $N_{pop}$, Obs. $N_{v}$"]
        lineparams = zip(labels, ['o', '^', None], ['--', '-', ':'],
                        ['grey', 'grey', 'red'])
    else:
        labels = ["Sim. $N_{pop}$, Obs. $N_{v}$",
                  "Obs. $N_{pop}$, Obs. $N_{v}$"]
        lineparams = zip(labels, ['^', None], ['-', ':'],
                        ['grey', 'red'])

    lines = [mlines.Line2D([], [], marker=m, label=label, ls=style,
                        color=c, mfc='k', mec='k')
            for label, m, style, c in lineparams]
    leg = ax.legend(lines, labels)

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=300)
    plt.show()
