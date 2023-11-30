"""
All functions needed for analysis either during or after runs
"""
import networkx as nx
import numpy as np
import math
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cmx
from scipy.stats import gaussian_kde
from scipy.stats import kurtosis, kstest
import pygeos
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import scipy.ndimage as ndimage
import powerlaw

def round_down(x, a):
    return math.floor(x / a) * a


def round_nearest(x, a):
    return round(x / a) * a


def round_up(x, a):
    return math.ceil(x / a) * a

def percentile_array(arr):
    """
    Calculates the percentile score of all values in the array
    :param arr: numpy.ndarray
        Array of values
    :return:
    """
    arr_min = np.nanmin(arr)
    arr_max = np.nanmax(arr)
    percentiles = ((arr - arr_min) / (arr_max - arr_min)) * 100

    return percentiles

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

def radial_profile(data, centre=None, unit_convert=None, nan_val=0.0):
    """
    Calculates a radial profile of data focussing on an origin
    https://stackoverflow.com/questions/21242011/most-efficient-way-to-
    calculate-radial-profile

    :param data: np.ndarray
        Array of data
    :param centre: tuple, list, None
        origin point of data from which radial profile will originate
    :param unit_convert:
    :return:
    """

    if centre is None:
        centre = [0, 0]
        centre[0] = math.floor(data.shape[1] / 2)
        centre[1] = math.floor(data.shape[0] / 2)

    # Clean NaNs and infs from data
    data = np.nan_to_num(data, nan=nan_val, posinf=0.0, neginf=0.0)
    y, x = np.indices(data.shape)
    r = np.sqrt((x - centre[0]) ** 2 + (y - centre[1]) ** 2)
    # Radius rounded for use as bin edges
    r = np.round(r).astype(int)

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

def log_log_radial(data, boundary=None, centre=None, mode='sum'):
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
    rad_prof = radial_profile(data, centre=centre)

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


def network_to_gdf(network):
    df = network.get_xy_wbc()

    df['lon'] = df['lon'] * 1000.  # km to m for CRS
    df['lat'] = df['lat'] * 1000.

    ntwrk_geo = gpd.GeoDataFrame(df,
                                 geometry=gpd.points_from_xy(df.lon, df.lat),
                                 crs={'init': 'epsg:27700'})
    return ntwrk_geo


def join_networkpnt_poppoly(network_gdf, pop_gdf, year):

    join_gdf = gpd.sjoin(network_gdf, pop_gdf[[f'{year}_pop', 'geometry']],
                         how='left')
    return join_gdf


def network_buffer_poly(network_gdf, distance):
    ntwk_buff = network_gdf.buffer(distance)

    ntwk_buff = gpd.GeoDataFrame(pd.concat([ntwk_buff,
                                            network_gdf[['lon', 'lat', 'WBC',
                                                         'degrees']]],
                                           axis=1))
    return ntwk_buff

def join_networkbuff_poppoly(network_buff, pop_gdf, year):

    join_gdf = gpd.sjoin(network_buff, pop_gdf[[f'{year}_pop', 'geometry']],
                         how='left')
    return join_gdf


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
                                    vmin=min(pop_gdf[f'{year}_pop'] / 1000),
                                    vmax=max(pop_gdf[f'{year}_pop'] / 1000)))
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

    null_filt = ~ntwrkyr_pop[f'{year}_pop'].isnull() * \
                ~ntwrkyr_pop[f'WBC'].isnull()
    x = ntwrkyr_pop['WBC'].values[null_filt]
    y = ntwrkyr_pop[f'{year}_pop'].values[null_filt]
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

    plt.figtext(x=0.9, y=0.9, s=f'{year}', )
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)
    plt.show()

def find_growth_rate(K, P0, P1, t, A):
    """
    Finds the growth rate of the logistic function from data

    :param K: int, float
        Carrying capacity
    :param P0: int, float
        Initial population/population density P
    :param P1: int, float
        Population/population density after t
    :param t: int
        Number of time steps between P0, P1
    :param A: int, float
        Area of the domain
    :return:
        r: Growth rate
    """

    # r = (np.log((((K/P1) - 1)*(P0 / (K-P0)))**-1) / t) * A # both correct
    r = A / t * (np.log(P1 * (K - P0)) - np.log(P0 * (K - P1))) # neater
    return r

def count_stations(xy_wbc, xmin, xmax, ymin, ymax):
    """
    Counts the number of stations present in each grid square

    :param xy_wbc:
    :param xmin:
    :param xmax:
    :param ymin:
    :param ymax:
    :return:
    """
    stn_in_grid = xy_wbc[(xmin <= xy_wbc['lon']) & (xy_wbc['lon'] < xmax) &
                         (ymin <= xy_wbc['lat']) & (xy_wbc['lat'] < ymax)]

    stn_count = np.histogram2d(x=stn_in_grid['lat'], y=stn_in_grid['lon'],
                               bins=[np.arange(ymin, ymax + 1, 1),
                                     np.arange(xmin, xmax + 1, 1)])[0]

    return stn_count

def dist_from_point(xv, yv, centrepoint, dx, dy):
    """
    Finds the distance of location points (e.g. nodes) from a single reference
    point (e.g. centre).
    :param xv:
    :param yv:
    :param centrepoint:
    :param dx:
    :param dy:
    :return:
    """

    d = np.hypot((centrepoint[0] + 0.5 * dx) -
                 np.around(xv.flatten() * 2) / 2,
                 (centrepoint[1] + 0.5 * dy) -
                 np.around(yv.flatten() * 2) / 2)

    return d

def compare_var_runs(run_varobs, run_varsim, var_name, save_path=None):
    """
    For comparing the difference with variables or results between two runs
    :return:
    """

    vars2d = run_varobs - run_varsim

    fig, ax = plt.subplots()

    cmap_ = cmx.seismic
    midpoint = 1 - np.max(vars2d) / (np.max(vars2d) + abs(np.min(vars2d)))
    cmap_shift = shiftedColorMap(cmap=cmap_, start=0, midpoint=midpoint, stop=1)

    ax.pcolor(vars2d, cmap=cmap_shift)
    sm2 = plt.cm.ScalarMappable(cmap=cmap_shift,
                                norm=plt.Normalize(vmin=np.min(vars2d),
                                                   vmax=np.max(vars2d)))
    sm2._A = []
    cb = plt.colorbar(sm2, label=var_name, ax=ax, location='bottom')

    ax.set_title(f"{var_name} (obs) - {var_name} (sim)")

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()

def compare_var_runs_stats(varobs, varsim, var_name,
                           save_path=None):

    varobs_stats = \
        [f"Mean: {np.mean(varobs.flatten())}", f"Std. dev.: {np.std(varobs)}",
         f"Variance: {np.var(varobs)}"]
    varsim_stats = \
        [f"Mean: {np.mean(varsim.flatten())}", f"Std. dev.: {np.std(varsim)}",
         f"Variance: {np.var(varsim)}"]

    if not save_path:
        print(f"Observed {var_name}")
        print("\n".join(varobs_stats))

        print(f"Simulated {var_name}")
        print("\n".join(varsim_stats))

    else:
        with open(save_path, 'w') as f:
            f.writelines(f"Observed {var_name}")
            f.writelines("\n".join(varobs_stats))

            f.writelines(f"Simulated {var_name}")
            f.writelines("\n".join(varsim_stats))
        f.close()

def prerun_check(model_params, rho_init, ntwk, ntwk_char, xv, yv):
    """
    Check that Y is not negative anywhere
    :param model_params:
    :param rho_init:
    :param ntwk_init:
    :param xv:
    :param yv:
    :return:
    """
    tau = model_params['tau']
    kappa = model_params['kappa']  # applied to Cr
    mu = model_params['mu']  # applied to Ct
    nu = model_params['nu']  # applied to Ct
    gm = model_params['gm']
    # r = model_params['r']  # accounts for out-of-system growth
    Ygross = model_params['ygross']

    cr_max = kappa * rho_init.max() ** tau

    xmin, xmax = xv.min(), xv.max()
    ymin, ymax = yv.min(), yv.max()
    ntwk_init = min([k for k in ntwk.keys()])
    stns = np.where(count_stations(xy_wbc=ntwk[ntwk_init], xmin=xmin, xmax=xmax,
                                   ymin=ymin, ymax=ymax).flatten() > 0)

    mindist_max = \
        min([dist_from_point(xv=xv.flatten()[stns], yv=yv.flatten()[stns],
                             centrepoint=c, dx=1, dy=1).min()
             for c in
             [(xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)]])

    # only after 1950 as the network is a little more established by then
    av_max_gi = np.array([ntwk[yr][ntwk_char].max()
                          for yr in ntwk.keys()
                          if yr > 1950]).mean()

    # max node importance factor will be 1 with dynamic method

    ct_max = mu * mindist_max + nu * (gm - av_max_gi)

    Ynet = Ygross - cr_max - ct_max

    print(f"dmin max: {mu * mindist_max}")
    print(f"gi max: {nu}")
    print(f"CT max: {ct_max}")
    print(f"CR max: {cr_max}")
    print(f"Ynet min: {Ynet}")

    if Ynet < 0:
        print("Negative Ynet present: CT + CR > Ygross")


# noinspection PyUnreachableCode
def calc_ct(xy, mu, nu):
    """
    Finds the betweenness centrality of and minimum distance from the
    nearest network node for each space on the population grid

    :param xy: pandas.core.frame.DataFrame
        Dataframe holding the longitude ('lon'), latitude ('lat') and
        weighted betweenness centrality ('WBC') of each node on the network

    :return:
    """
    # finding minimum distance from station for each grid (assuming centre
    # of gridspace, different from .m script) and NBC of nearest station
    # (mean in case multiple stations at same distance)
    raise ImportError('Function not ready')
    for i in np.arange(0, rho.shape[0]):
        for j in np.arange(0, rho.shape[1]):
            # nodes in the nearest gridspace(s) with nodes rather than just
            # single nearest node. Can
            d = np.hypot((xv[i, j] + 0.5 * dx) -
                         np.around(xy['lon'].values * 2) / 2,
                         (yv[i, j] + 0.5 * dy) -
                         np.around(xy['lat'].values * 2) / 2)
            mindist[i, j] = min(d)
            bc_nearest[i, j] = \
                np.mean(xy.iloc[np.where(d == min(d))]['WBC'])
            # * 100 + 1e-3)

    ct = mu * mindist + nu * bc_nearest

def dnorm(x, mu, sd):
    """
    Probability density for the normal distribution
    http://www.adeveloperdiary.com/data-science/computer-vision/applying-
    gaussian-smoothing-to-an-image-using-python-from-scratch/
    :param x: float, int
        Value
    :param mu: float, int
        Distribution mean
    :param sd: float, int
        Distribution standard deviation
    :return:
    """
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (
            -np.power((x - mu) / sd, 2) / 2)

def gaussian_kernel(size, sigma=1, verbose=False):
    """
    Generates gaussian kernel
    http://www.adeveloperdiary.com/data-science/computer-vision/applying-
    gaussian-smoothing-to-an-image-using-python-from-scratch/
    :param size: int
        Required size of gaussian kernel. Greater size means more smoothing
    :param sigma: float, int
        Standard deviation
    :return: numpy.ndarray
        Gaussian kernel

    """
    # 1D Gaussian of size
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    # Normal probability density of 1D
    for i in range(size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    # 1D to 2D
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
    # Normalising 2D kernel so centre is 1
    kernel_2D *= 1.0 / kernel_2D.max()

    return kernel_2D

def convolution(image, kernel, average=False):
    """
    Convolutes an image or numpy array to a Gaussian kernel to smoothing the
    image/array.
    http://www.adeveloperdiary.com/data-science/computer-vision/applying-
    gaussian-smoothing-to-an-image-using-python-from-scratch/
    :param image: numpy.ndarray
        The original image or numpy array to be smoothed
    :param kernel: numpy.ndarray
        Kernel array
    :param average: bool
        boolean for averaging the smoothing
    :return:
    """
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

    output = np.zeros(image.shape)

    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)

    padded_image = np.zeros(
        (image_row + (2 * pad_height), image_col + (2 * pad_width)))

    padded_image[pad_height:padded_image.shape[0] - pad_height,
    pad_width:padded_image.shape[1] - pad_width] = image

    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.nansum(
                kernel * padded_image[row:row + kernel_row,
                         col:col + kernel_col])

            if average:
                output[row, col] /= kernel.shape[0] * kernel.shape[1]

    return output

def gaussian_smooth(image, kernel_size, average):
    """
    Blurs and image or data array via a Gaussian kernel
    http://www.adeveloperdiary.com/data-science/computer-vision/applying-
    gaussian-smoothing-to-an-image-using-python-from-scratch/
    :param image: numpy.ndarray
        Image or data array
    :param kernel_size: int
        Required size of the kernel
    :return:
    """
    kernel = gaussian_kernel(kernel_size, sigma=math.sqrt(kernel_size))

    return convolution(image, kernel, average=average)

def mean_kernel(size):
    """
    Generates mean filtering kernel

    :param size: int
        Required size of gaussian kernel. Greater size means more smoothing
    :return: numpy.ndarray
        Mean kernel

    """
    return np.full(shape=(size, size), fill_value=1 / size ** 2)

def mean_smooth(image, kernel_size, average):
    """
    Blurs and image or data array via a Gaussian kernel
    http://www.adeveloperdiary.com/data-science/computer-vision/applying-
    gaussian-smoothing-to-an-image-using-python-from-scratch/
    :param image: numpy.ndarray
        Image or data array
    :param kernel_size: int
        Required size of the kernel
    :return:
    """
    kernel = mean_kernel(kernel_size)

    return convolution(image, kernel, average=average)

def xy_kurtosis(arr_dict, years):
    """
    Returns the kurtosis of the x and y means of arrays across the years

    :param arr_dict: dict
        Dictionary with array of
    :param years: list, numpy.ndarray
        List of years from which to extract kurtosis statistics. Must be keys
        of arr_dict.
    :return:
    """
    x_kurt = []
    y_kurt = []

    for yr in years:
        x_kurt.append(kurtosis(np.nanmean(arr_dict[yr] / 1000, axis=0)))
        y_kurt.append(kurtosis(np.nanmean(arr_dict[yr] / 1000, axis=1)))

    return np.array(x_kurt), np.array(y_kurt)

def spatial_binning(array, dir, bin_width):
    """
    Finds the spatial sum of boolean values within spatial bins

    :param array: numpy.ndarray
        Counts of objects across array
    :param dir: str
        'x' or 'y'
    :param bin_width: int
        Width of spatial bins
    :return:
    """
    if dir == 'x':
        axis = 0
    elif dir == 'y':
        axis = 1
    else:
        raise ValueError('dir must be "x" or "y"')

    sum_dir = np.nansum(array, axis=axis)

    ind_bins = np.append(np.arange(0, len(sum_dir), bin_width), len(sum_dir))

    cnt_sum = [sum_dir[ind_bins[i]:ind_bins[i + 1]].sum()
               for i, x in enumerate(ind_bins) if i != len(ind_bins) - 1]

    return cnt_sum, ind_bins


def second_deriv_finite(x, y):
    """
    Calculates the second derivate of a curve
    https://stackoverflow.com/a/40227340/8764287
    """
    dy = np.diff(y, 1)
    dx = np.diff(x, 1)

    yfirst = dy / dx
    xfirst = 0.5 * (x[:-1] + x[1:])

    dyfirst = np.diff(yfirst, 1)
    dxfirst = np.diff(xfirst, 1)

    ysecond = dyfirst / dxfirst
    xsecond = 0.5 * (xfirst[:-1] + xfirst[1:])

    return xsecond, ysecond


def xy_smoothness(arr_dict, years):
    """
    Calculates the 'smoothness' of a curve as the integral of the squared second
    derivative (via finite differences)

    Not the most accurate method as it is dependent on the parameterisation of
    the data (https://math.stackexchange.com/a/1594601)

    """
    # i and j are used to refer to the x and y directions of the DOMAIN to avoid
    # confusion with the x and y axis of the mean curves
    i_smoothness = []
    j_smoothness = []

    for yr in years:
        i = np.nanmean(arr_dict[yr] / 1000, axis=0)
        j = np.nanmean(arr_dict[yr] / 1000, axis=1)

        i_smoothness.append(
            second_deriv_finite(x=np.arange(len(i)), y=i)[1].sum())
        j_smoothness.append(
            second_deriv_finite(x=np.arange(len(j)), y=j)[1].sum())

    return i_smoothness, j_smoothness

def mbe(y_true, y_pred):
    """
    Calculates the mean bias error across the difference between predicted
    (simulated) and true (observed) values.

    If either array contains a NaN value, zero difference between the two arrays
    is recorded for corresponding elements.

    :param y_true:
        Array of observed values
    :param y_pred:
        Array of prediction values
    :return: mbe (float)
        Mean bias error score
    """
    # y_true = y_true.reshape(len(y_true),1)
    # y_pred = y_pred.reshape(len(y_pred),1)
    ypred, y_true = filter_nans(y_pred, y_true)
    diff = (y_true - y_pred)
    mbe = diff.mean()
    return mbe

def rmse(y_pred, y_true):
    """
    Calculates the root mean squared error across the difference between
    predicted (simulated) and true (observed) values.

    If either array contains a NaN value, zero difference between the two arrays
    is recorded for corresponding elements.

    :param y_true:
        Array of observed values
    :param y_pred:
         Array of prediction values
    :return: rmse (float)
        Root mean squared error score
    """
    ypred, y_true = filter_nans(y_pred, y_true)
    rmse = np.sqrt(((y_pred - y_true) ** 2).mean())
    return rmse

def mae(y_pred, y_true):
    """
    Calculates the mean absoluute error between predicted (simulated) and true
    (observed) values.

    If either array contains a NaN value, zero difference between the two arrays
    is recorded for corresponding elements.

    :param y_true:
        Array of observed values
    :param y_pred:
         Array of prediction values
    :return: mae (float)
        Mean absolute error score
    """
    y_pred, y_true = filter_nans(y_pred, y_true)
    mae = mean_absolute_error(y_pred=y_pred, y_true=y_true)
    return mae


def r_squared(y_pred, y_true):
    """
    Calculates the r2 value between predicted (simulated) and true
    (observed) values.

    If either array contains a NaN value, zero difference between the two arrays
    is recorded for corresponding elements.

    :param y_true:
        Array of observed values
    :param y_pred:
         Array of prediction values
    :return: r2 (float)
        Mean absolute error score
    """
    y_pred, y_true = filter_nans(y_pred, y_true)
    r2 = r2_score(y_pred=y_pred, y_true=y_true)
    return r2

def errors_from_results(obs, sim, savepath):
    """
    Create statistical errors file from existing results

    :param obs: dict
        Dict of observed values, years as keys and results as values
    :param sim: dict
        Dict of simulated values, years as keys and results as values
    :param savepath: str
        Save path
    :return:
    """

    errors = {'rmse': [], 'mae': [], 'mbe': []}
    for yr in sorted(list(obs.keys())):
        errors['rmse'] = rmse(y_pred=sim[yr], y_true=obs[yr])
        errors['mae'] = mae(y_pred=sim[yr], y_true=obs[yr])
        errors['mbe'] = mbe(y_pred=sim[yr], y_true=obs[yr])

    pickle.dump(errors, open(savepath, "wb"))

def filter_nans(arr1, arr2):
    """
    Where NaNs are present, set elements in both arrays to 0

    :param arr1: numpy.ndarray
    :param arr2: numpy.ndarray
    :return:
    """
    nans = np.where(np.logical_or(np.isnan(arr1), np.isnan(arr2)), 1, 0)

    arr1 = np.where(nans == 1, 0, arr1)
    arr2 = np.where(nans == 1, 0, arr2)

    return arr1, arr2


def trim_array(array, borders):
    """
    Trims border off array (to remove boundary errors)

    :param array:
    :param borders:
    :return:
    """

    return array[borders[0]:-(borders[0]), borders[1]:-(borders[1])]

def create_array_gdf(xvyv, array_dict):
    """
    Convert numpy array to GeoDataFrame

    :param xvyv:
    :param array_dict:
    :return:
    """
    yv = np.column_stack((xvyv['yv'], xvyv['yv'][:, 0]))
    yv = np.row_stack(
        (yv, np.full_like(yv[-1, :], fill_value=yv[-1, 0] + 1))) * 1000
    xv = np.column_stack((xvyv['xv'], np.full_like(xvyv['xv'][:, -1],
                                                   fill_value=xvyv['xv'][
                                                                  0, -1] + 1)))
    xv = np.row_stack((xv, xv[0, :])) * 1000

    points = pygeos.creation.points(xv.ravel(), yv.ravel())
    gdf = gpd.GeoDataFrame(geometry=points)
    # gdf.plot()
    coords = pygeos.get_coordinates(pygeos.from_shapely(gdf.geometry))

    nrow = xv.shape[0] - 1
    ncol = yv.shape[1] - 1
    n = nrow * ncol
    nvertex = (nrow + 1) * (ncol + 1)
    assert len(coords) == nvertex

    # Make sure the coordinates are ordered into grid form:
    x = coords[:, 0]
    y = coords[:, 1]
    order = np.lexsort((x, y))
    x = x[order].reshape((nrow + 1, ncol + 1))
    y = y[order].reshape((nrow + 1, ncol + 1))

    # Setup the indexers
    left = lower = slice(None, -1)
    upper = right = slice(1, None)
    corners = [
        [lower, left],
        [lower, right],
        [upper, right],
        [upper, left],
    ]

    # Allocate output array
    xy = np.empty((n, 4, 2))

    # Set the vertices
    for i, (rows, cols) in enumerate(corners):
        xy[:, i, 0] = x[rows, cols].ravel()
        xy[:, i, 1] = y[rows, cols].ravel()

    # Create geodataframe and plot result
    mesh_geometry = pygeos.creation.polygons(xy)
    mesh_gdf = gpd.GeoDataFrame(geometry=mesh_geometry)

    for k in array_dict.keys():
        mesh_gdf[k] = array_dict[k].ravel()

    return mesh_gdf

def array_to_irreg_gdf(mesh_gdf, res_array_dict, obs_array_dict, irreg_gdf,
                       area_reg=1):
    """
    Apply array values to existing GeoDataFrame

    :param mesh_gdf:
    :param res_array_dict:
    :return:
    """
    # intersection of res and obs keys (years) for for-loops

    for k in res_array_dict.keys():
        mesh_gdf[f'{k}_res'] = res_array_dict[k].ravel()
        mesh_gdf[f'{k}_obs'] = obs_array_dict[k].ravel()

    merge_gdf = gpd.overlay(irreg_gdf, mesh_gdf, how="intersection")

    for k in res_array_dict.keys():
        merge_gdf[f'{k}_res'] = \
            merge_gdf[f'{k}_res'] * (merge_gdf.area / area_reg)
        merge_gdf[f'{k}_obs'] = \
            merge_gdf[f'{k}_obs'] * (merge_gdf.area / area_reg)

    merge_gdf_bor = merge_gdf.dissolve(by='NAME', aggfunc='sum')

    for k in res_array_dict.keys():
        merge_gdf_bor[f'{k}_res'] = \
            merge_gdf_bor[f'{k}_res'] / merge_gdf_bor.area
        merge_gdf_bor[f'{k}_obs'] = \
            merge_gdf_bor[f'{k}_obs'] / merge_gdf_bor.area

    return merge_gdf_bor

def irreg_to_reg(irreg_data, irreg_gdf, reg_gdf, irreg_vars, reg_aggby,
                 aggfunc, quantity):
    """
    Convert irregular data to regular GeoDataFrame

    :param irreg_data:
        Irregular data
    :param irreg_gdf:
        Irregular GeoDataFrame corresponding to irreg_data
    :param reg_gdf:
        Regular GeoDataFrame
    :param irreg_vars:
        Variables to be converted
    :param reg_aggby:
        Variable on which to dissolve data
    :param aggfunc:
        Aggregation function for dissolving data: 'sum' or 'mean'

    :return:
    """
    irreg_gdf = irreg_gdf.merge(irreg_data, left_index=True, right_index=True)
    irreg_gdf['irreg_area'] = irreg_gdf.area

    reg_gdf['reg_area'] = reg_gdf.area
    grid_gdf = gpd.overlay(irreg_gdf, reg_gdf, how='union')
    grid_gdf['intersectarea'] = grid_gdf.area

    if quantity == 'count':
        for var in irreg_vars:
            grid_gdf[var] = grid_gdf[var].multiply(
                grid_gdf['intersectarea'] / grid_gdf['irreg_area'],
                axis='index')
    elif quantity == 'feature':
        for var in irreg_vars:
            grid_gdf[var] = grid_gdf[var].multiply(
                grid_gdf['intersectarea'] / grid_gdf['reg_area'], axis='index')

    grid_gdf = grid_gdf.dissolve(by=reg_aggby, aggfunc=aggfunc)

    return grid_gdf


def find_city_radius(cluster, xvyv, city_outline, borders=None, centre=[0,0]):
    """
    Find the radius of the city according to population density limits.
    Assumes that the city is clipped to real city limits.

    Cluster data must be generated in R using the City Clustering Algorithm
    (Rozenfeld et al., 2008, 2011) and saved as a TXT file with columns
    "long", "lat" (which match units of xvyv)and "add.data".
    :param cluster:
    :param xvyv:
    :param outline:
    :return:
    """
    xv, yv = xvyv['xv'], xvyv['yv']

    if borders is not None:
        xv = trim_array(xv, borders)
        yv = trim_array(yv, borders)
        cluster = cluster[(cluster['long'] >= xv.min()) &
                          (cluster['long'] <= xv.max()) &
                          (cluster['lat'] >= yv.min()) &
                          (cluster['lat'] <= yv.max())]
        city_outline = trim_array(city_outline, borders)

    # Access clusters (from CCA in R -  Rozenfeld et al., 2008, 2011) as
    # XY data and population
    x_data = cluster['long'].values - np.min(xv).astype(int)
    y_data = cluster['lat'].values - np.min(yv).astype(int)
    z_data = cluster['add.data'].values.astype(int)

    # Create empty array of domain and fill with rho values
    arr = np.full_like(xv, np.nan)
    arr[y_data, x_data] = z_data

    # Set bool of rho data and NaN, find contiguous values
    arr_bool = np.where(np.isnan(arr), 1, 0)
    labels, numL = ndimage.label(arr_bool)
    # Assume contiguous at corner is all NaN and create from this,
    # filling any holes in the main cluster
    outskirts = labels[0, 0]
    # Clip cluster by extent of Greater London
    # (as we're are considering GL only)
    fullclstr = np.where(labels == outskirts, 0, 1) * city_outline

    if centre == [0,0]:
        centre[0] = math.floor(fullclstr.shape[1] / 2)
        centre[1] = math.floor(fullclstr.shape[0] / 2)

    # Find edges of cluster
    struct = ndimage.generate_binary_structure(2, 2)
    erode = ndimage.binary_erosion(fullclstr, struct)
    edges = fullclstr ^ erode
    edge_args = np.argwhere(edges == 1)
    # Find distances between centre and each edge pixel
    dists = np.sqrt(abs(centre[1] - edge_args[:, 0]) ** 2 +
                    abs(centre[0] - edge_args[:, 1]) ** 2)
    # Median distance assumed to be radius of city
    distance = np.median(dists)

    return fullclstr, distance

def find_city_scaling(data, radius, centre=None, borders=None,
                      logmode='cumsum'):

    if borders is not None:
        data = trim_array(array=data, borders=borders)

    xvar, yvar, dist = log_log_radial(data=data, centre=centre, mode=logmode)

    radius_arg = np.argwhere(xvar > np.log10(radius)).min()

    a, b = np.polyfit(xvar[:radius_arg], yvar[:radius_arg], 1)

    return xvar, yvar, a, b

def KStest_power_law(pop_obs, pop_sim, years, xvyv, city_outline,
                      obs_cluster_dict, sim_cluster_dict, borders=None,
                      savepath=None, unit_convert=None, log_mode='cumsum',
                      centre=[0,0]):

    obsalpha, simalpha, obsbeta, simbeta, obsrad, simrad, obsks, simks, \
        obspv, simpv = \
            [], [], [], [], [], [], [], [], [], []
    # fig, ax = plt.subplots(figsize=(5, 4))
    # fig.patch.set_facecolor('w')
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    for i, yr in enumerate(years):

        obsr = find_city_radius(cluster=obs_cluster_dict[yr], centre=centre,
                                xvyv=xvyv, borders=borders,
                                city_outline=city_outline)[1]
        obsx, obsy, obsa, obsb = find_city_scaling(
            data=pop_obs[yr], centre=centre, borders=borders, radius=obsr)

        obs_trunc_rad_arg = np.argmin(abs(obsx-np.log10(obsr))) + 1

        obspl = obsb + obsa * obsx[:obs_trunc_rad_arg]
        # obspl = obsb*obsx[:obs_trunc_rad_arg]**(-obsa)
        ks_obs = kstest(obsy[:obs_trunc_rad_arg], obspl)

        simr = find_city_radius(cluster=sim_cluster_dict[yr], centre=centre,
                                xvyv=xvyv, borders=borders,
                                city_outline=city_outline)[1]
        simx, simy, sima, simb = find_city_scaling(
            data=pop_sim[yr], borders=borders, radius=simr, centre=centre)

        sim_trunc_rad_arg = np.argmin(abs(simx-np.log10(simr))) + 1

        simpl = simb + sima * simx[:sim_trunc_rad_arg]

        ks_sim = kstest(simy[:sim_trunc_rad_arg], simpl)

        obsbeta.append(obsa)
        simbeta.append(sima)
        obsrad.append(obsr)
        simrad.append(simr)

        obsks.append(ks_obs.statistic)
        simks.append(ks_sim.statistic)
        obspv.append(ks_obs.pvalue)
        simpv.append(ks_sim.pvalue)

def KStest_power_law_clauset(pop_obs, pop_sim, years, xvyv, city_outline,
                      obs_cluster_dict, sim_cluster_dict, borders=None,
                      savepath=None, unit_convert=None, log_mode='cumsum',
                      centre=[0,0]):

    obsalpha, simalpha, obsbeta, simbeta, obsrad, simrad, obsks, simks, \
        obspv, simpv = \
            [], [], [], [], [], [], [], [], [], []
    # fig, ax = plt.subplots(figsize=(5, 4))
    # fig.patch.set_facecolor('w')
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    for i, yr in enumerate(years):

        obsr = find_city_radius(cluster=obs_cluster_dict[yr], centre=centre,
                                xvyv=xvyv, borders=borders,
                                city_outline=city_outline)[1]
        obsx, obsy, obsa, obsb = find_city_scaling(
            data=pop_obs[yr], centre=centre, borders=borders, radius=obsr)

        rad_prof = radial_profile(trim_array(array=pop_obs[yr],
                                             borders=borders), centre=centre)

        rad_prof_cumsum = np.cumsum(rad_prof[2])
        rad_prof_sum = rad_prof[2]
        rad_prof_mean = rad_prof[0]
        dist = np.unique(rad_prof[1])

        obs_trunc_rad_arg = np.argmin(abs(obsx-np.log10(obsr))) + 1

        obspl = obsb + obsa * obsx[:obs_trunc_rad_arg]
        # obspl = obsb*obsx[:obs_trunc_rad_arg]**(-obsa)
        results = powerlaw.Fit(rad_prof_sum)

        simr = find_city_radius(cluster=sim_cluster_dict[yr], centre=centre,
                                xvyv=xvyv, borders=borders,
                                city_outline=city_outline)[1]
        simx, simy, sima, simb = find_city_scaling(
            data=pop_sim[yr], borders=borders, radius=simr, centre=centre)

        sim_trunc_rad_arg = np.argmin(abs(simx-np.log10(simr))) + 1

        simpl = simb + sima * simx[:sim_trunc_rad_arg]

        ks_sim = kstest(simy[:sim_trunc_rad_arg], simpl)

        obsbeta.append(obsa)
        simbeta.append(sima)
        obsrad.append(obsr)
        simrad.append(simr)

        obsks.append(ks_obs.statistic)
        simks.append(ks_sim.statistic)
        obspv.append(ks_obs.pvalue)
        simpv.append(ks_sim.pvalue)

    return obsalpha, simalpha, obsbeta, simbeta, obsrad, simrad, obsks, simks, \
            obspv, simpv

def rho_to_txt(xv, yv, rho, savepath):
    """
    Turns population density array data into TXT files with longitude,
    latitude, population density columns
    :param xv: np.ndarray
    :param yv:
    :param rho:
        Population density array
    :param savepath: str
        Save filepath

    :return:
    """
    rho_yr_flat = rho.flatten()
    xv_flat = xv.flatten()
    yv_flat = yv.flatten()

    data = np.vstack((xv_flat, yv_flat, rho_yr_flat)).T
    data = np.nan_to_num(data, nan=0)

    np.savetxt(savepath, data, fmt='%d',
               delimiter=',', header="xv,yv,rho", comments='')
