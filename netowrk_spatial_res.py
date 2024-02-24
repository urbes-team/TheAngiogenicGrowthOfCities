# %%
import numpy as np
import pickle
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import networkx as nx
import geopandas as gpd
import pandas as pd
from analysis_functions import mean_smooth, trim_array
from plot_growth import basic_pcolor
from mpl_toolkits.axes_grid1 import make_axes_locatable
from plot_growth import plot_xy_mean_years, plot_var
from scipy.optimize import curve_fit
import matplotlib.colors as colors
import matplotlib.cm as cmx
from scipy.ndimage import measurements
from copy import copy
from graph_functions import set_accessibility
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
from matplotlib.patches import ConnectionPatch


# %%
def curve_fit_log(xdata, ydata, zero=False):
    """Fit data to a power law with weights according to a log scale"""
    # Weights according to a log scale
    # Apply fscalex
    xdata_log = np.log10(xdata)
    # Apply fscaley
    ydata_log = np.log10(ydata)

    xdata_log, ydata_log = remove_nan_inf(xdata_log, ydata_log)

    # Fit linear
    popt_log, pcov_log = curve_fit(linlaw, xdata_log, ydata_log)
    # print(popt_log, pcov_log)
    # Apply fscaley^-1 to fitted data
    ydatafit_log = np.power(10, linlaw(xdata_log, *popt_log))
    # There is no need to apply fscalex^-1 as original data is already available
    # popt_log[0]: exponent; popt_log[1]: coefficient
    return (popt_log, pcov_log, ydatafit_log)


def powlaw(x, a, b):
    return a * np.power(x, b)


def linlaw(x, a, b):
    return a + x * b


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


def log_log_radial(data, boundary=None, centre=None, unit_convert=None,
                   mode='sum'):
    """
    Finds the log10 values of the cumulative sum of radial data.

    :param data: np.ndarray
        Array of original data
    :param boundary: int
        Distance from the centre after which the radial profile will be
        truncated
    :param centre: tuple, list, None
        origin point of data from which radial profile will originate
    :param unit_convert: int
        Factor by which the radial distance is multiplied (needs to be tested!)
    :param mode: str
        Mode of aggregation.
        'sum': sum of the data feature at each radial bin
        'cumsum': cumulative sum of data feature along radius
        'mean': mean of data feature at

    """
    if mode not in ['sum', 'cumsum', 'mean']:
        raise ValueError("mode must be sum, cumsum or mean")
    rad_prof = radial_profile(data, centre=centre, unit_convert=unit_convert)

    if unit_convert is not None:
        zero_change = np.argwhere(rad_prof[2][:boundary * unit_convert] > 0)
        rad_prof_cumsum = np.cumsum(rad_prof[2])[zero_change].flatten()
        rad_prof_sum = rad_prof[2][zero_change].flatten()
        rad_prof_mean = rad_prof[0][zero_change].flatten()
        dist = np.arange(boundary * unit_convert)[zero_change].flatten()
    else:
        rad_prof_cumsum = np.cumsum(rad_prof[2])[:boundary]
        rad_prof_sum = rad_prof[2][:boundary]
        rad_prof_mean = rad_prof[0][:boundary]
        dist = np.unique(rad_prof[1])[:boundary]

    #     print(rad_prof_sum)
    xvar = np.log10(dist)
    if mode == "cumsum":
        yvar = np.log10(rad_prof_cumsum)
    elif mode == "mean":
        yvar = np.log10(rad_prof_mean)
    elif mode == "sum":
        yvar = np.log10(rad_prof_sum)
    xvar, yvar = remove_nan_inf(xvar, yvar, zero=False)

    return xvar, yvar, dist

# %%
pop_grid = gpd.read_file('./Data/London/reg_grid_pop_1881_2011.shp'). \
    set_index('num_codes')
xmin, ymin, xmax, ymax = pop_grid.total_bounds / 1000
centre = [xmin + (xmax - xmin) / 2, ymin + (ymax - ymin) / 2]  # [532.0, 181.0]

ntwrk_years = \
    pickle.load(open(f'./Data/London/networks_5years_nodup_factcnct.p',
                     'rb'))

sim_ntwrk_yrs = \
    pickle.load(
        open("./Results/NetworkGrowth/simulated_networks.p", 'rb'))

rho_yrs = pickle.load(open("./Data/London/rho_yrs.p", 'rb'))
xvyv = pickle.load(open("./Data/London/xvyv.p", "rb"))

rho_res = pickle.load(
    open("./Results/LdnPopulationGrowth/rho_results.p", 'rb'))

for yr in ntwrk_years.keys():
    lon = nx.get_node_attributes(ntwrk_years[yr], 'lon')
    lat = nx.get_node_attributes(ntwrk_years[yr], 'lat')
    pos = {}
    for n in ntwrk_years[yr]:
        pos[n] = [lon[n] * 1000, lat[n] * 1000]

    nx.set_node_attributes(ntwrk_years[yr], pos, name='pos')

for yr in sim_ntwrk_yrs.keys():
    sim_ntwrk_yrs[yr] = set_accessibility(sim_ntwrk_yrs[yr])[0]

years = [1891, 1951, 1991, 2011]

# %%
# SIMULATED DATA
# AVERAGE DISTANCE OF NODES OF DEG(N) FROM THE ROOT NODE
node_stats = {}
# NODES OF DEG(1) ON THE EDGE OF THE NETWORK NEED TO CHANGE TO DEG(2)
# AS THEY WILL CONNECT TO SOMEWHERE
for yr in years:
    # for yr in [1851, 1861, 1871]:
    g = nx.Graph(sim_ntwrk_yrs[yr])
    if len(g.nodes) > 0:
        node_stats[yr] = {}
        degs = dict(g.degree)
        max_deg = max(degs.values())
        if max_deg > 3:
            max_keys = \
                np.array(list(degs.keys()))[
                    np.where(np.array(list(degs.values())) == max_deg)]
            #         max_keys = [key for key, value in g.items()
            #                     if value == max(g.values())]
            if len(max_keys) > 1:
                # in the case of multiple nodes with max degree, the most central is
                # chosen as the root
                pos = []
                for n in max_keys:
                    pos.append([g.nodes[n]['pos'][0] + xmin,
                                g.nodes[n]['pos'][1] + ymin])
                pos = np.array(pos)
                rel = pos - centre
                dist = np.sqrt(rel[:, 0] ** 2 + rel[:, 1] ** 2)
                min_dist = np.argmin(dist)
                root = max_keys[min_dist]
            else:
                root = max_keys[0]
        root_pos = np.array([g.nodes[root]['pos'][0] + xmin,
                             g.nodes[root]['pos'][0] + ymin])
        node_stats[yr]['root_node'] = root
        node_stats[yr]['max_deg'] = max_deg
        # find the distance from other nodes of lesser/same degree
        # find nodes of lesser degree
        for deg in np.arange(1, max_deg + 1):
            pos = []
            lsr_nodes = np.array(list(degs.keys()))[
                np.where(np.array(list(degs.values())) == deg)]
            if deg == max_deg:
                # remove root node from list
                lsr_nodes = np.delete(lsr_nodes,
                                      np.argwhere(lsr_nodes == root))

            if len(lsr_nodes) > 0:
                for n in lsr_nodes:
                    pos.append([g.nodes[n]['pos'][0] + xmin,
                                g.nodes[n]['pos'][1] + ymin])
                pos = np.array(pos)
                rel = pos - root_pos
                # Euclidean distance of the lesser nodes to the chosen root node'
                dist = np.sqrt(rel[:, 0] ** 2 + rel[:, 1] ** 2)
                # Average of distance recorded
                node_stats[yr][deg] = np.mean(dist)

pickle.dump(node_stats,
            open("./Results/NetworkGrowth/node_deg_root_sim.p", 'wb'))

# %%
# OBSERVED DATA
# AVERAGE DISTANCE OF NODES OF DEG(N) FROM THE ROOT NODE
node_stats = {}
# NODES OF DEG(1) ON THE EDGE OF THE NETWORK NEED TO CHANGE TO DEG(2)
# AS THEY WILL CONNECT TO SOMEWHERE
for yr in ntwrk_years.keys():
    print(yr)
    node_stats[yr] = {}
    g = nx.Graph(ntwrk_years[yr])
    degs = dict(g.degree)
    max_deg = max(degs.values())
    if max_deg > 3:
        max_keys = \
            np.array(list(degs.keys()))[
                np.where(np.array(list(degs.values())) == max_deg)]
        if len(max_keys) > 1:
            # in the case of multiple nodes with max degree, the most central is
            # chosen as the root
            pos = []
            for n in max_keys:
                pos.append([g.nodes[n]['lon'], g.nodes[n]['lat']])
            pos = np.array(pos)
            rel = pos - centre
            dist = np.sqrt(rel[:, 0] ** 2 + rel[:, 1] ** 2)
            min_dist = np.argmin(dist)
            root = max_keys[min_dist]
        else:
            root = max_keys[0]
        root_pos = np.array([g.nodes[root]['lon'],
                             g.nodes[root]['lat']])
        node_stats[yr]['root_node'] = root
        node_stats[yr]['max_deg'] = max_deg
    # find the distance from other nodes of lesser/same degree
    # find nodes of lesser degree
    for deg in np.arange(1, max_deg + 1):
        pos = []
        lsr_nodes = np.array(list(degs.keys()))[
            np.where(np.array(list(degs.values())) == deg)]
        if deg == max_deg:
            # remove root node from list
            lsr_nodes = np.delete(lsr_nodes,
                                  np.argwhere(lsr_nodes == root))

        if len(lsr_nodes) > 0:
            for n in lsr_nodes:
                pos.append([g.nodes[n]['lon'], g.nodes[n]['lat']])
            pos = np.array(pos)
            rel = pos - root_pos
            # Euclidean distance of the lesser nodes to the chosen root node'
            dist = np.sqrt(rel[:, 0] ** 2 + rel[:, 1] ** 2)
            # Average of distance recorded
            node_stats[yr][deg] = np.mean(dist)

pickle.dump(node_stats, open("./network_scale_data/node_deg_root_obs.p", 'wb'))
# TODO add this in from urbangrowth, change path

# %%
# OBSERVED PDF OF NUMBER OF NODES FOR
obs_ntwrk_deg_count = {}
for yr in ntwrk_years.keys():
    deg_count = \
        np.unique(list(dict(ntwrk_years[yr].degree).values()),
                  return_counts=True)
    obs_ntwrk_deg_count[yr] = deg_count

obs_ntwrk_deg_pdf = {}
# obs_ntwrk_deg_cdf = {}

for yr in ntwrk_years.keys():
    deg_cdf = np.cumsum(obs_ntwrk_deg_count[yr][1][::-1])[::-1]
    deg_cdf = deg_cdf / np.nanmax(deg_cdf)

    deg_pdf = obs_ntwrk_deg_count[yr][1] / np.nansum(obs_ntwrk_deg_count[yr][1])
    obs_ntwrk_deg_pdf[yr] = [deg_pdf, deg_cdf]

pickle.dump(obs_ntwrk_deg_count,
            open("./network_scale_data/obs_deg_count.p", 'wb'))
pickle.dump(obs_ntwrk_deg_pdf, open("./network_scale_data/obs_deg_pdf.p", 'wb'))
# TODO add this in from urbangrowth, change path
# %%

# SIMULATED PDF OF NUMBER OF NODES FOR
sim_ntwrk_deg_count = {}
for yr in sim_ntwrk_yrs.keys():
    deg_count = \
        np.unique(list(dict(sim_ntwrk_yrs[yr].degree).values()),
                  return_counts=True)
    sim_ntwrk_deg_count[yr] = deg_count

sim_ntwrk_deg_pdf = {}

for yr in sim_ntwrk_yrs.keys():
    if len(sim_ntwrk_yrs[yr].nodes):
        deg_cdf = np.cumsum(sim_ntwrk_deg_count[yr][1][::-1])[::-1]
        deg_cdf = deg_cdf / np.nanmax(deg_cdf)

        deg_pdf = sim_ntwrk_deg_count[yr][1] / np.nansum(
            sim_ntwrk_deg_count[yr][1])
        sim_ntwrk_deg_pdf[yr] = [deg_pdf, deg_cdf]

pickle.dump(sim_ntwrk_deg_count,
            open("./Results/NetworkGrowth/sim_deg_count.p", 'wb'))
pickle.dump(sim_ntwrk_deg_pdf,
            open("./Results/NetworkGrowth/sim_deg_pdf.p", 'wb'))

# %%
sim_stn_count_dict = {}

sim_ai = {}

sim_ai_inv = {}

sim_deg = {}

for yr in [1891, 1951, 1991, 2011]:
    print(yr)

    sim_ntwrk_coords = pd.concat(
        [pd.DataFrame.from_dict({n: sim_ntwrk_yrs[yr].nodes[n]['pos'][0]
                                 for n in sim_ntwrk_yrs[yr].nodes},
                                columns=['lon'], orient='index'),
         pd.DataFrame.from_dict({n: sim_ntwrk_yrs[yr].nodes[n]['pos'][1]
                                 for n in sim_ntwrk_yrs[yr].nodes},
                                columns=['lat'], orient='index'),
         pd.DataFrame.from_dict({n: sim_ntwrk_yrs[yr].nodes[n]['a_i_inv']
                                 for n in sim_ntwrk_yrs[yr].nodes},
                                columns=['a_i_inv'], orient='index'),
         pd.DataFrame.from_dict({n: sim_ntwrk_yrs[yr].nodes[n]['a_i']
                                 for n in sim_ntwrk_yrs[yr].nodes},
                                columns=['a_i'], orient='index'),
         pd.DataFrame.from_dict(dict(sim_ntwrk_yrs[yr].degree()),
                                columns=['degree'], orient='index')],
        axis=1)

    sim_stn_grid = \
        sim_ntwrk_coords[
            (0 <= sim_ntwrk_coords['lon']) & (sim_ntwrk_coords['lon'] < 84) &
            (0 <= sim_ntwrk_coords['lat']) & (sim_ntwrk_coords['lat'] < 66)]
    sim_stn_count = np.histogram2d(x=sim_stn_grid['lat'], y=sim_stn_grid['lon'],
                                   bins=[np.arange(0, 66 + 1, 1),
                                         np.arange(0, 84 + 1, 1)])[0]

    sim_stn_count_dict[yr] = sim_stn_count

    simai_nearest = np.zeros_like(sim_stn_count)
    simai_inv_nearest = np.zeros_like(sim_stn_count)
    simdeg_nearest = np.zeros_like(sim_stn_count)
    mindist = np.zeros_like(sim_stn_count)
    for i in np.arange(0, mindist.shape[0]):
        for j in np.arange(0, mindist.shape[1]):
            d = np.hypot((xvyv['xv'][i, j] - xvyv['xv'].min() + 0.5 * 1) -
                         np.around(sim_stn_grid['lon'].values * 2) / 2,
                         (xvyv['yv'][i, j] - xvyv['yv'].min() + 0.5 * 1) -
                         np.around(sim_stn_grid['lat'].values * 2) / 2)
            mindist[i, j] = min(d)
            simai_nearest[i, j] = \
                np.mean(sim_stn_grid.iloc[np.where(d == min(d))]['a_i'])
            simai_inv_nearest[i, j] = \
                np.mean(sim_stn_grid.iloc[np.where(d == min(d))]['a_i_inv'])

    sim_ai[yr] = simai_nearest
    sim_ai_inv[yr] = simai_inv_nearest
    sim_deg[yr] = simdeg_nearest

pickle.dump(sim_ai, open("./Results/NetworkGrowth/sim_ai.p", 'wb'))
pickle.dump(sim_ai_inv, open("./Results/NetworkGrowth/sim_ai_inv.p", 'wb'))
pickle.dump(sim_deg, open("./Results/NetworkGrowth/sim_deg.p", 'wb'))
pickle.dump(sim_stn_count_dict,
            open("./Results/NetworkGrowth/sim_stncount.p", 'wb'))
