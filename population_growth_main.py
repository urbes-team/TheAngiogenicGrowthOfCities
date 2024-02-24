"""
This script simulates population growth for the London case, initiating with
observed population data and allowing the model to respond to the growth of the
observed network.
Using optimised parameters.
Phase I (1831-1891): Ynet(i) = Ygross(i), D = DI
Phase II.1 (1892-1950): Ynet(i) = Ygross(i) - CL(i) - CT(i), R
Phase II.2 (1951-1991): S, R
Phase II.3 (1992-2011): Ynet(i) = Ygross(i) - CL(i) - CT(i), S, R = 0

@author: Isabella Capel-Timms
"""
# %%
from graph_functions import NetworkGraph, SubGraph, RhoGrid
from plot_growth import *
from analysis_functions import *
import networkx as nx
import numpy as np
import pickle
import geopandas as gpd
import pandas as pd
from copy import copy, deepcopy
import os
import datetime
import f90nml

# %%
os.chdir('')

# plt_dir = 'roughSA/beta2_Dx10_local'

run_name = 'domain_AICT_highygrossonlyDx2_1871'
# run_name = 'test_for_coupling'

dir_path = f'./Runs/{run_name}'
if not os.path.isdir(dir_path):
    os.mkdir(dir_path)

make_plots = True
record_details = True
simple_test = True

nml_filepath = None
notes = 'optimised values with the all sink in tIV = 0' \
        '1st regime (1831-1891): Ynet(i) = Ygross(i), D = DI; ' \
        '2nd regime (1892-1950): Ynet(i) = Ygross(i) - CL(i) - CT(i), R' \
        '3rd regime (1951-1991): S, R' \
        '4th regime (1992-2011): Ynet(i) = Ygross(i) - CL(i) - CT(i), S, R = 0 '

meta_data = {'Timestamp': str(datetime.datetime.now()), 'Notes': notes}

# for basic inclusion of phases. Only logistic growth and
# income densification before this year
ct_year_lim = 1891

# int_sink_year = 1921
sink_year = 1951
if sink_year != 0:
    int_sink_rate = 0.00934
    ext_sink_rate = 0.0026  # for linear sink
else:
    int_sink_rate = 0.009
    ext_sink_rate = 0.0026  # for linear sink

regrowth_year = 1992

if nml_filepath is None:

    ntwrk_factor = 'a_i_inv'
    start_yr = 1831
    end_yr = 2012

    # None, "record" or filepath
    file_paths = {'xy_wbc_fp': "./Data/London/xy_wbc_cc.p",
                  'pop_grid_fp':
                      './Data/London/reg_grid_pop_1881_2011_'
                      'nullfixed_allyears.shp',
                  'rho_yrs_fp': "./Data/London/rho_yrs.p",
                  'network_years_fp': './Data/London/networks_5years_nodup_factcnct.p',
                  'smooth_yrs_fp': "./Data/London/rho_yrs_smooth.p"}

    # # initiate with 1831 population data
    grid_params = {'dx': 1, 'dy': 1, 'dt': 1, 'tot_ts': 180}

    model_params = {'ntwrk_factor': ntwrk_factor, 'cr_fctr': 'rhohist',
                    'start_yr': start_yr, 'end_yr': end_yr, 'beta': 2.07,
                    'tau': 0.995, 'kappa': 0.00023, 'mu': 0.359, 'nu': 2.43,
                    'gm': 1.8, 'd': 0.03, 'dI': 0.03, 'r': 158.5,
                    'ygross': 4.89, 'ygross0': 4.14, 'ygross_e': 0.099,
                    'int_sink_rate': int_sink_rate,
                    'ext_sink_rate': ext_sink_rate, 'KctIV': 4256.6}
    # For Phase I beta is coeff of Y0 so combined here (i.e. beta not
    # explicitly used in Phase I, and ygross0 = Y0 * beta here)
    # ygross_e refers to B in paper.

else:
    #  READ NML FROM PATH
    nml_all = f90nml.read(nml_filepath)

    grid_params = dict(nml_all['grid_params'])
    model_params = dict(nml_all['model_params'])
    file_paths = dict(nml_all['file_paths'])

xy_wbc_ready = pickle.load(open(file_paths['xy_wbc_fp'], "rb"))
gen_network = False

if not gen_network:
    if 'network_years_fp' not in file_paths.keys():
        file_paths['network_years_fp'] = \
            './core_data/network/networks_5years_nodup_factcnct.p'
    ntwrk_years = pickle.load(open(file_paths['network_years_fp'], 'rb'))

rho_obs = pickle.load(open(file_paths['rho_yrs_fp'], "rb"))
smooth_obs = pickle.load(open(file_paths['smooth_yrs_fp'], 'rb'))

pop_grid = gpd.read_file(file_paths['pop_grid_fp']).set_index('num_codes')
xmin, ymin, xmax, ymax = pop_grid.total_bounds / 1000

grid_params.update(
    {'xlim': [xmin, xmax], 'ylim': [ymin, ymax],
     'N0': rho_obs[model_params['start_yr']]})

model_params.update(
    {'Kc': rho_obs[2011].sum() / ((xmax - xmin) * (ymax - ymin)),
     'rho_limit': np.percentile(rho_obs[2011], 98), 'ct_year': ct_year_lim,
     'sink_year': sink_year, 'int_sink_rate': int_sink_rate})

rho_growth = RhoGrid(grid_params=grid_params, model_params=model_params)

# %%

# Load the rail network
# xy of stations
print('Loading network')
xy = pd.read_csv('./Data/London/ldn_stn_xy_w27700.csv', header=0,
                 index_col=0)[['lon_m', 'lat_m', 'Year']] \
    .rename(columns={'lon_m': 'lon', 'lat_m': 'lat'})

xy['lon'] = xy['lon'] / 1000.  # m to km
xy['lat'] = xy['lat'] / 1000.

if ct_year_lim is None:
    ct_year_lim = xy['Year'].min()
    model_params['ct_year'] = ct_year_lim

# find source and target nodes for all edges
ldn_conx = pd.read_csv('./Data/London/stations_with_yr_norep.csv', header=0)
G = nx.from_pandas_edgelist(ldn_conx, source='StationA', target='StationB')

ldn_coords = {}
for n in list(G.nodes):
    # converting m to km
    ldn_coords[n] = (xy.loc[n, 'lon'], xy.loc[n, 'lat'])

ldn_full = NetworkGraph(graph=G, coords=ldn_coords, nodes=G.nodes)
# %%
network_years = 1
ts_per_year = 1  # TODO: eventually sync this with RhoGrid.dt

# %%
plot_years = [1831, 1851, 1891, 1931, 1961, 1991, 2011]
gi_plot_years = np.arange(1841, 2012, 10)

# if nml_filepath is None:
f90nml.write({'metadata': meta_data,
              'grid_params': {key: val for key, val in grid_params.items()
                              if key not in ['N0', 'topography']},
              'model_params': model_params, 'file_paths': file_paths},
             open(f'{dir_path}/{run_name}_settings.nml', 'w'))

simulated_N = []
max_nbc = []
gmgi = {}
ct = {}
cr = {}
xy_wbc_dict = {}
rho_results = {}
eta_dict = {}
gmgi_arr = {}
dmin_arr = {}
eta_redist = {}
ynet = {}
qm = {}
qneg = {}
errors = {'rmse': [], 'mae': [], 'mbe': [], 'rmse_smooth': [], 'mae_smooth': [],
          'mbe_smooth': []}

for year in np.arange(model_params['start_yr'], model_params['end_yr'],
                      network_years):
    print(year)
    # Load network for year and fix connections
    if gen_network:
        Gyr_nodes = xy[xy['Year'] <= year].index.values
        if year < model_params['ct_year']:  # network avoided until t=ct_year
            xy_wbc = None
        elif type(xy_wbc_ready) == dict and len(Gyr_nodes) > 0:
            xy_wbc = xy_wbc_ready[year]
        elif len(Gyr_nodes) > 0:
            ntwrk_yr = \
                SubGraph(graph=ldn_full.graph,
                         subgraph=nx.Graph(ldn_full.graph.subgraph(Gyr_nodes)),
                         coords={n: (xy.loc[n, 'lon'], xy.loc[n, 'lat'])
                                 for n in Gyr_nodes},
                         nodes=Gyr_nodes)
            ntwrk_yr.add_new_edges(distance=4, hub_limit=7)
            ntwrk_yr.set_wbc_cc()
            xy_wbc = ntwrk_yr.get_xy_wbc_cc()
            # Run within RhoGrid (adjusts factors for network)
            if xy_wbc_ready == "record":
                xy_wbc_dict[year] = xy_wbc
        else:
            xy_wbc = None
    elif year >= min(list(ntwrk_years.keys())):
        # pre-generated networks every 5 years from 1836
        if year < model_params['ct_year']:  # network avoided until t=ct_year
            xy_wbc = None

        else:
            ntwrk_yr = ntwrk_years[year - (year % 5) + 1]

            xy_wbc = pd.DataFrame.from_dict(dict(ntwrk_yr.degree()),
                                            orient='index', columns=['degree'])
            for k in ['lat', 'lon', 'year', 'WBC', 'CC', 'a_i', 'a_i_inv']:
                xy_wbc[k] = pd.Series(nx.get_node_attributes(ntwrk_yr, k))

    else:
        xy_wbc = None

    # PHASE I
    if xy_wbc is None:
        print('No costs')
        rho_growth.set_Ygross_i(average=False)
        rho_growth.run_growth_ts_domain_ygross_only(ts_to_run=1)
    # PHASE II.1
    elif year < sink_year or sink_year == 0:
        # INTERNAL SINK - Y gross not variable in this phase
        rho_growth.Ygross = model_params['ygross']
        # REDISTRIBUTION OF POPULATION FROM AREAS WITH -VE ETA
        rho_growth.run_growth_ts_domain_sinknegeta(
            ts_to_run=1, remove=False, r_ynet=False,
            sink_rate=model_params['int_sink_rate'])
        # DISTRIBUTED LOGISTIC GROWTH
        rho_growth.run_growth_ts_domain_r(ts_to_run=1, ntwrkxy=xy_wbc,
            ntwrk_fctr=model_params['ntwrk_factor'],
            cr_fctr=model_params['cr_fctr'])
    # PHASE II.2
    elif sink_year <= year < regrowth_year:
        # EXTERNAL SINK - Y gross not variable in this phase
        rho_growth.Ygross = model_params['ygross']
        rho_growth.run_growth_ts_domain_sinknegeta(
            ts_to_run=1, remove=False, r_ynet=False,
            sink_rate=model_params['int_sink_rate'], calc_ynet=True,
            ntwrkxy=xy_wbc, cl_fctr=model_params['cr_fctr'],
            ntwrk_fctr=model_params['ntwrk_factor'])
        rho_growth.run_growth_ts_domain_ext_sink_simple(
            ts_to_run=1, sink_rate=model_params['ext_sink_rate'])
    # PHASE II.3
    elif year >= regrowth_year:
        if year == regrowth_year:
            rho_growth.set_Kc(model_params['KctIV'])

        rho_growth.run_growth_ts_domain_r(ts_to_run=1, ntwrkxy=xy_wbc,
            ntwrk_fctr=model_params['ntwrk_factor'],
            cr_fctr=model_params['cr_fctr'])
    # TODO: remove year from run_growth, for testing only

    simulated_N.append(rho_growth.rho.sum())

    if record_details:
        eta_dict[year] = rho_growth.eta
        ct[year] = rho_growth.ct
        cr[year] = rho_growth.cr
        rho_results[year] = rho_growth.rho
        eta_redist[year] = rho_growth.eta_redist
        qm[year] = rho_growth.qm
        ynet[year] = rho_growth.Ynet
        qneg[year] = rho_growth.qneg
    if year in rho_obs.keys():
        errors['rmse'].append(rmse(y_pred=rho_growth.rho,
                                   y_true=rho_obs[year]))
        errors['mae'].append(mae(y_pred=rho_growth.rho,
                                 y_true=rho_obs[year]))
        errors['mbe'].append(mbe(y_pred=rho_growth.rho,
                                 y_true=rho_obs[year]))
        errors['rmse_smooth'].append(rmse(y_pred=rho_growth.rho,
                                          y_true=smooth_obs[year]))
        errors['mae_smooth'].append(mae(y_pred=rho_growth.rho,
                                        y_true=smooth_obs[year]))
        errors['mbe_smooth'].append(mbe(y_pred=rho_growth.rho,
                                        y_true=smooth_obs[year]))

    if year in plot_years:
        if year == model_params['start_yr']:
            pass
            last_yr = year
        else:
            if make_plots and year in rho_obs.keys():
                plot_evolution(
                    rho=rho_growth.rho - past_decade, xv=rho_growth.xv,
                    yv=rho_growth.yv, year=year, network=None, show=False,
                    observed_rho=rho_obs[year] - rho_obs[last_yr],
                    save_path=f'{dir_path}/{year}_growthdiff.png',
                    borders=(6, 6))
                last_yr = year
        # # RHO SIM - OBS
        if make_plots:
            if year in rho_obs.keys():
                plot_obs_sim(rho_sim=rho_growth.rho, rho_obs=rho_obs[year],
                             xv=rho_growth.xv, yv=rho_growth.yv, year=year,
                             cmap="viridis", cmap_diff="bwr", borders=(6, 6),
                             threshold_rho=np.nanpercentile(rho_obs[year], 10),
                             under_col_rho='grey', network=None, show=False,
                             save_path=f'{dir_path}/{year}_obssim_growth.png')
                # # RHO SIM - OBS BLURRED
                plot_obs_sim(rho_sim=rho_growth.rho, rho_obs=smooth_obs[year],
                             xv=rho_growth.xv, yv=rho_growth.yv, year=year,
                             cmap="viridis", cmap_diff="bwr", borders=(6, 6),
                             threshold_rho=np.nanpercentile(rho_obs[year], 10),
                             under_col_rho='grey', network=None, show=False,
                             save_path=f'{dir_path}/'
                                       f'{year}_obssim_growth_smoothed.png')

                plot_obs_sim(rho_sim=rho_growth.rho, rho_obs=rho_obs[year],
                             xv=rho_growth.xv, yv=rho_growth.yv, year=year,
                             cmap="viridis", cmap_diff="bwr", perc_diff=True,
                             threshold_rho=np.nanpercentile(rho_obs[year], 10),
                             under_col_rho='grey', network=None, show=False,
                             borders=(6, 6), threshold_perc=100,
                             over_col_perc='grey',
                             save_path=f'{dir_path}/{year}'
                                       f'_obssim_growth_perc.png')

                plot_obs_sim(rho_sim=rho_growth.rho, rho_obs=smooth_obs[year],
                             xv=rho_growth.xv, yv=rho_growth.yv, year=year,
                             cmap="viridis", cmap_diff="bwr", perc_diff=True,
                             threshold_rho=np.nanpercentile(rho_obs[year], 10),
                             under_col_rho='grey', network=None, show=False,
                             borders=(6, 6), threshold_perc=100,
                             over_col_perc='grey',
                             save_path=f'{dir_path}/{year}'
                                       f'_obssim_growth_smoothed_perc.png')
                # # RHO
                plot_evolution(rho=rho_growth.rho, xv=rho_growth.xv,
                               yv=rho_growth.yv, year=year, network=None,
                               show=False, observed_rho=rho_obs[year],
                               save_path=f'{dir_path}/{year}_growth.png',
                               borders=(6, 6))

                plot_accuracy_scatter(
                    obs_data=rho_obs[year], sim_data=rho_growth.rho,
                    varobs="Observed $\\rho$ (cap km$^{-2}$",
                    varsim="Simulated $\\rho$ (cap km$^{-2}$", borders=(6, 6),
                    savepath=f'{dir_path}/{year}_scatter.png', log=True,
                    cmap='plasma')

                # # RADIAL RHO OBS & SIM
                plot_radial_obs_sim(
                    rho_sim=rho_growth.rho, rho_obs=rho_obs[year],
                    borders=(6, 6),
                    savepath=f'{dir_path}/{year}_radial_rho.png')
                plot_radial_obs_sim(rho_sim=rho_growth.rho,
                                    rho_obs=smooth_obs[year],
                                    savepath=f'{dir_path}/{year}_smooth_'
                                             f'radial_rho.png', borders=(6, 6))
            # # CR
            if year in rho_obs.keys():
                obs_rho = rho_obs[year]
            else:
                obs_rho = None
            plot_var(var=rho_growth.cr, xv=rho_growth.xv,
                     yv=rho_growth.yv, year=year, network=None, show=False,
                     observed_rho=obs_rho, var_name='$C_{R}$',
                     save_path=f'{dir_path}/', borders=(6, 6),
                     save_var_name='cr')
            # # CT
            plot_var(var=rho_growth.ct, xv=rho_growth.xv,
                     yv=rho_growth.yv, year=year, network=None, show=False,
                     observed_rho=obs_rho, var_name='$C_{T}$',
                     save_path=f'{dir_path}/', borders=(6, 6),
                     save_var_name='ct')
            # CR + CT
            plot_var(var=rho_growth.ct + rho_growth.cr, xv=rho_growth.xv,
                     yv=rho_growth.yv, year=year, network=None, show=False,
                     observed_rho=obs_rho, var_name='$C_{T} + C_{R}$',
                     save_path=f'{dir_path}/', borders=(6, 6),
                     save_var_name='ctcr')
            # # ETA
            plot_var(var=rho_growth.eta, xv=rho_growth.xv,
                     yv=rho_growth.yv, year=year, network=None, show=False,
                     observed_rho=obs_rho, var_name='$\eta$',
                     save_path=f'{dir_path}/', borders=(6, 6),
                     save_var_name='eta')
            # ETA REDIST
            plot_var(var=rho_growth.eta_redist, xv=rho_growth.xv,
                     yv=rho_growth.yv, year=year, network=None, show=False,
                     observed_rho=obs_rho, var_name='$\\eta$ redist',
                     save_path=f'{dir_path}/', borders=(6, 6),
                     save_var_name='etaredist')

            plot_var(var=rho_growth.qm, xv=rho_growth.xv,
                     yv=rho_growth.yv, year=year, network=None, show=False,
                     observed_rho=obs_rho, var_name='$q_{m}$',
                     save_path=f'{dir_path}/', borders=(6, 6),
                     save_var_name='qm')

            plot_var(var=rho_growth.qneg, xv=rho_growth.xv,
                     yv=rho_growth.yv, year=year, network=None, show=False,
                     observed_rho=obs_rho, var_name='$q_{-}$',
                     save_path=f'{dir_path}/', borders=(6, 6),
                     save_var_name='qneg')

            if type(rho_growth.Ygross) == np.ndarray:
                # YGROSS
                plot_var(var=rho_growth.Ygross, xv=rho_growth.xv,
                         yv=rho_growth.yv, year=year, network=None, show=False,
                         observed_rho=obs_rho, var_name='$Y_{gross}$',
                         save_path=f'{dir_path}/', borders=(6, 6),
                         save_var_name='ygross')
                # YNET
                plot_var(var=rho_growth.Ynet, xv=rho_growth.xv,
                         yv=rho_growth.yv, year=year, network=None, show=False,
                         observed_rho=obs_rho, var_name='$Y_{net}$',
                         save_path=f'{dir_path}/', borders=(6, 6),
                         save_var_name='ynet')

        past_decade = copy(rho_growth.rho)

    if year in gi_plot_years and year >= 1941:
        stns = np.where(count_stations(xy_wbc=xy_wbc, xmin=xmin, xmax=xmax,
                                       ymin=ymin, ymax=ymax).flatten()
                        > 0)[0]
        gi = rho_growth.get_bc_nearest().flatten()[stns]
        gm = np.full_like(gi, rho_growth.gm)
        dist_cent = \
            dist_from_point(
                xv=rho_growth.xv.flatten()[stns],
                yv=rho_growth.yv.flatten()[stns],
                centrepoint=(np.round(np.median(rho_growth.xv[0, :])),
                             np.round(np.median(rho_growth.yv[:, 0]))),
                dx=1, dy=1)

        data = {'gmgi': gm - gi, 'gi': gi, 'gm': gm, 'dist': dist_cent}

        data = pd.DataFrame.from_dict(data=data, orient='columns')

        gmgi[year] = data
        ct[year] = rho_growth.ct
        dmin_arr[year] = rho_growth.mindist
        gmgi_arr[year] = (
                    rho_growth.gm - rho_growth.ntwrk_fctr_nearest)

    if year in gi_plot_years or year == model_params['start_yr']:
        rho_results[year] = rho_growth.rho
        eta_dict[year] = rho_growth.eta
        eta_redist[year] = rho_growth.eta_redist
        qm[year] = rho_growth.qm
        qneg[year] = rho_growth.qneg

if make_plots:
    plot_years = np.intersect1d(plot_years, list(rho_obs.keys()))

    plot_radial_years(rho_results, years=plot_years, cmap='autumn',
                      title='Radial profiles of simulated $\\rho$ 1831 - 2011',
                      savepath=f'{dir_path}/radial_years.png', borders=(6, 6),
                      xvar=r'Mean $\rho$ (1000 cap km$^{-2}$)')
    plot_two_radial_years(rho_dict1=rho_results, rho_dict2=smooth_obs,
                          years=plot_years, cmap='autumn', borders=(6, 6),
                          title='Radial profiles of simulated and observed '
                                '(smoothed) $\\rho$ 1831 - 2011',
                          lr_titles=['Simulated', 'Observed'],
                          savepath=f'{dir_path}/radial_profiles_bothsmooth.png')
    plot_two_radial_years(rho_dict1=rho_results, rho_dict2=rho_obs,
                          years=plot_years, cmap='autumn', borders=(6, 6),
                          title='Radial profiles of simulated and observed '
                                '$\\rho$ 1831 - 2011',
                          lr_titles=['Simulated', 'Observed'],
                          savepath=f'{dir_path}/radial_profiles_both.png')

    plot_xy_mean_years(rho_dict=rho_results, years=plot_years, xv=rho_growth.xv,
                       yv=rho_growth.yv, cmaps=('winter', 'spring'),
                       savepath=f'{dir_path}/xy_mean_years.png', borders=(6, 6))
    plot_two_xy_mean_years(rho_dict1=rho_results, rho_dict2=smooth_obs,
                           years=plot_years, xv=rho_growth.xv, yv=rho_growth.yv,
                           cmaps=('winter', 'spring'), borders=(6, 6),
                           title='Mean $\\rho$ in $x$ and $y$ directions',
                           savepath=f'{dir_path}/meanxy_bothsmooth.png',
                           lr_titles=['Simulated', 'Observed'])
    plot_two_xy_mean_years(rho_dict1=rho_results, rho_dict2=rho_obs,
                           years=plot_years, xv=rho_growth.xv, yv=rho_growth.yv,
                           cmaps=('winter', 'spring'), borders=(6, 6),
                           title='Mean $\\rho$ in $x$ and $y$ directions',
                           savepath=f'{dir_path}/meanxy_both.png',
                           lr_titles=['Simulated', 'Observed'])
pickle.dump(rho_results, open(f"{dir_path}/rho_results.p", "wb"))
pickle.dump(errors, open(f"{dir_path}/rmsemaembe.p", "wb"))
if record_details:
    pickle.dump(eta_dict, open(f"{dir_path}/eta.p", "wb"))
    pickle.dump(ct, open(f"{dir_path}/ct_sim.p", "wb"))
    pickle.dump(cr, open(f"{dir_path}/cr_sim.p", "wb"))
    pickle.dump(eta_redist, open(f"{dir_path}/eta_redist.p", "wb"))
    pickle.dump(qm, open(f"{dir_path}/qm.p", "wb"))
    pickle.dump(qneg, open(f"{dir_path}/qneg.p", "wb"))
    pickle.dump(ynet, open(f"{dir_path}/ynet.p", "wb"))

if not os.path.isdir(dir_path + 'rho_txt'):
    os.mkdir(dir_path + '/rho_txt')
if not os.path.isdir(dir_path + 'rho_clusters'):
    os.mkdir(dir_path + '/rho_clusters')

xvyv = pickle.load(open("./growth_results/xvyv.p", "rb"))

for yr in list(rho_obs.keys()):
    rho_to_txt(xv=xvyv['xv'], yv=xvyv['yv'], rho=rho_results[yr],
               savepath=dir_path + f"/rho_txt/rho{yr}.txt")

# Save gm_gi to avoid rerunning each time
if xy_wbc_ready == "record":
    pickle.dump(xy_wbc_dict, open("./core_data/network/xy_wbc_cc.p", "wb"))
pickle.dump(gmgi, open(f"{dir_path}/gmgi.p", "wb"))

if make_plots is True:
    observed_N = [np.nansum(rho_obs[yr]) for yr in np.arange(1831, 2012, 10)]
    plot_obs_sim_pop(observed_N=observed_N, simulated_N=simulated_N,
                     r=rho_growth.r, save_path=f'{dir_path}/'
                                               f'total_population_sum.png')
# %%
