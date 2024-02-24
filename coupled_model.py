"""
Model coupling both the population and transport network growth model.
Initiated with observed population and network

@author: Isabella Capel-Timms
"""
# %%
from graph_functions import RhoGrid, remove_outside_nodes
from graph_functions import create_distance_probability_array, \
    create_edge_probability_array
from plot_growth import *
from analysis_functions import *
from network_growth_main import *
import networkx as nx
import numpy as np
import pickle
import geopandas as gpd
import pandas as pd
from copy import copy
import os
import datetime


os.chdir('')  # Project root filepath

run_name = 'newnetworkrun'

dir_path = f'./Results/CoupledSystem/{run_name}/'
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

ct_year_lim = 1891
if ct_year_lim != 0:
    diff_mult = 4  # multiplier of diffusion during logistic only growth
else:
    diff_mult = 1

int_sink_year = 1921
sink_year = 1951
if sink_year != 0:
    a_i_lim = 0
    rho_lim = 0
    int_sink_rate = 0.00934
    ext_sink_rate = 0.0026  # for linear sink
else:
    a_i_lim = None
    rho_lim = None
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
                      './Data/London/reg_grid_pop_1881_2011_nullfixed_allyears.shp',
                  'rho_yrs_fp': "./Data/London/rho_yrs.p",
                  'thames_fp': './Data/London/thames.p',
                  'network_years_fp': './Data/London/networks_5years_nodup_factcnct.p',
                  'smooth_yrs_fp': "./Data/London/rho_yrs_smooth.p",
                  'empty_grid_fp': './Data/London/empty_regular_grid.shp',
                  'init_network_fp': "./Data/London/1841_init_network.p",
                  'xvyv': "./Data/London/xvyv.p"}

    # %%
    # DOMAIN ONLY PARAMS
    model_params = {'ntwrk_factor': ntwrk_factor, 'cr_fctr': 'rhohist',
                    'start_yr': start_yr, 'ntwrk_start_yr': 1841,
                    'end_yr': end_yr, 'beta': 2.07, 'tau': 0.995,
                    'kappa': 0.00023, 'mu': 0.359, 'nu': 2.43, 'gm': 1.8,
                    'd': 0.03, 'dI': 0.03, 'r': 158.5, 'ygross': 4.89,
                    'ygross0': 4.14, 'ygross_e': 0.099,
                    'int_sink_rate': int_sink_rate,
                    'ext_sink_rate': ext_sink_rate, 'KctIV': 4256.6}

else:
    #  READ NML FROM PATH
    nml_all = f90nml.read(nml_filepath)

    grid_params = dict(nml_all['grid_params'])
    model_params = dict(nml_all['model_params'])
    file_paths = dict(nml_all['file_paths'])

xvyv = pickle.load(open(file_paths['xvyv'], "rb"))
xy_wbc_ready = pickle.load(open(file_paths['xy_wbc_fp'], "rb"))
gen_network = False

pop_grid = gpd.read_file(file_paths['pop_grid_fp']).set_index('num_codes')
xmin, ymin, xmax, ymax = pop_grid.total_bounds / 1000
rho_obs = pickle.load(open(file_paths['rho_yrs_fp'], "rb"))
smooth_obs = pickle.load(open(file_paths['smooth_yrs_fp'], 'rb'))
empty_grid = gpd.read_file(file_paths['empty_grid_fp']).set_index('num_codes')

# # initiate with 1831 population data
grid_params = {'dx': 1, 'dy': 1, 'dt': 1, 'tot_ts': 180,
               'xlim': [xmin, xmax], 'ylim': [ymin, ymax],
               'N0': rho_obs[1831], 'nrows': rho_obs[1831].shape[0],
               'ncols': rho_obs[1831].shape[1]}

if not gen_network:
    if 'network_years_fp' not in file_paths.keys():
        file_paths['network_years_fp'] = \
            './Data/London/networks_5years_nodup_factcnct.p'
    if 'phi' not in model_params.keys() and 'chi' not in model_params.keys():
        model_params['phi'] = None
        model_params['chi'] = None
    ntwrk_years = pickle.load(open(file_paths['network_years_fp'], 'rb'))

thames = pickle.load(open(file_paths['thames_fp'], 'rb'))
thames_top = None

grid_params.update(
    {'xlim': [xmin, xmax], 'ylim': [ymin, ymax], 'topography': thames_top,
     'N0': rho_obs[model_params['start_yr']]})

model_params.update(
    {'Kc': rho_obs[2011].sum() / ((xmax - xmin) * (ymax - ymin)),
     'rho_limit': np.percentile(rho_obs[2011], 98), 'ct_year': ct_year_lim,
     'diff_mult': diff_mult, 'sink_year': sink_year, 'a_i_lim': a_i_lim,
     'rho_lim': rho_lim, 'int_sink_rate': int_sink_rate})

rho_growth = RhoGrid(grid_params=grid_params, model_params=model_params)

# %%
if ct_year_lim is None:
    ct_year_lim = xy['Year'].min()
    model_params['ct_year'] = ct_year_lim

# %%
network_years = 1
ts_per_year = 1  # TODO: eventually sync this with RhoGrid.dt

# %%
plot_years = [1831, 1851, 1891, 1931, 1961, 1991, 2011]
gi_plot_years = np.arange(1841, 2012, 10)


simulated_N, max_ncb = [], []
gmgi, ct, cr, xy_wbc_dict, rho_results, eta_dict, gmgi_arr, dmin_arr, \
eta_redist, ynet, qm, qneg = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}

rho_errors = {'rmse': [], 'mae': [], 'mbe': [], 'rmse_smooth': [],
              'mae_smooth': [], 'mbe_smooth': []}

# NETWORK MODEL INIT
print('Loading network')
# Load the rail network
init_network = pickle.load(open(file_paths['init_network_fp'], 'rb'))
nx.set_node_attributes(init_network,
                       {n: (init_network.nodes[n]['pos'][0] + xmin,
                            init_network.nodes[n]['pos'][1] + ymin)
                        for n in init_network.nodes}, 'pos')

nx.set_node_attributes(init_network, {n: init_network.nodes[n]['pos'][0]
                                      for n in init_network.nodes}, 'lon')
nx.set_node_attributes(init_network, {n: init_network.nodes[n]['pos'][1]
                                      for n in init_network.nodes}, 'lat')
init_network = remove_outside_nodes(G=init_network, xmin=xmin, xmax=xmax,
                                    ymin=ymin, ymax=ymax)
# SET NECESSARY NODE ATTRIBUTES
init_network = set_accessibility(G=init_network)[0]
# Network model parameters
phi = -2.5  # gravity exponent
g = 0.0008
edge_weight = 3000
psi = 1
ntwrk_dt = 10
max_in_grid = 6  # Maximum number of stations
cost_coeff = 4  # lambda in paper

ntwrk_model_params = {'phi': phi, 'g': g, 'edge_weight': edge_weight,
                      'psi': psi, 'dt': ntwrk_dt, 'max_stns': max_in_grid,
                      'cost_coeff': cost_coeff}

if phi == abs(phi):
    phi *= -1

d_origin = create_distance_probability_array(shape=rho_obs[1831].shape,
                                             phi=ntwrk_model_params['phi'])
stn_origin = create_edge_probability_array(shape=rho_obs[1831].shape,
                                           psi=ntwrk_model_params['psi'])
ntwrk_ev_yrs = np.arange(model_params['ntwrk_start_yr'],
                         model_params['end_yr'] + 1, ntwrk_model_params['dt'])
# hold #stations in cell
stn_mat = np.zeros_like(rho_obs[model_params['start_yr']])
stn_graph = nx.empty_graph(n=0, create_using=None)  # nx graph of sim network
total_stns = np.zeros_like(ntwrk_ev_yrs, dtype=float)  # time series stn total
total_pop = np.zeros_like(ntwrk_ev_yrs, dtype=float)  # time series pop total
G_ts = np.zeros_like(ntwrk_ev_yrs, dtype=float)  # time series of G
dPdt_all = []

mean_qj = []

# network nx objects, dataframes through time
ntwrk_dict, ntwrk_df_dict = {}, {}
xy_wbc = None

# if nml_filepath is None:
f90nml.write({'metadata': meta_data,
              'grid_params': {key: val for key, val in grid_params.items()
                              if key not in ['N0', 'topography']},
              'model_params': model_params, 'file_paths': file_paths,
              'ntwrk_model_params': ntwrk_model_params},
             open(f'{dir_path}/{run_name}_settings.nml', 'w'))

for year in np.arange(model_params['start_yr'], model_params['end_yr'],
                      network_years):
    print(year)
    # Load network for year and fix connections
    # Network evolves every 10 years so params stay the same as uncoupled code
    if year == model_params['ntwrk_start_yr']:
        print('INIT NETWORK')
        # USE INIT NETWORK
        stn_graph = nx.Graph(init_network)
        # CONVERT TO DATAFRAME FOR USE BY POPULATION
        xy_wbc = pd.DataFrame.from_dict(dict(stn_graph.degree()),
                                        orient='index', columns=['degree'])
        for k in ['lat', 'lon', 'year', 'a_i', 'a_i_inv']:
            xy_wbc[k] = pd.Series(nx.get_node_attributes(stn_graph, k))

        ntwrk_dict[year] = nx.Graph(stn_graph)

        stn_mat = graph_to_array(stn_graph, stn_mat,
                                 [0, 0, xmax-xmin, ymax-ymin])

        ntwrk_df_dict[year] = xy_wbc

        new_nodes = None

    elif (year - 1) % 10 == 0 and year > model_params['ntwrk_start_yr']:
        print('EVOLVE NETWORK')
        # EVOLVE NETWORK
        # Array of rho values
        city_mat = copy(rho_growth.rho)
        city_mat = np.where(np.isnan(city_mat), 0, city_mat)

        # arrays of edge lengths present and i, j nodes
        if not nx.is_empty(stn_graph):
            scaled_graph = nx.Graph(stn_graph)
            nx.set_node_attributes(scaled_graph,  # scaling km to m
                {n: np.array(stn_graph.nodes[n]['pos']) * 1000
                    for n in nx.get_node_attributes(stn_graph, 'pos').keys()},
                name='pos')
            # create array of length of edges in each grid cell
            edge_lengths, i_nodes, j_nodes = \
                network_to_array(graph=scaled_graph, grid=empty_grid)
            # TODO: find out why arrays are mirrored on x-axis. Fix for now
            edge_lengths = np.flip(edge_lengths, axis=0)
            i_nodes = np.flip(i_nodes, axis=0)
            j_nodes = np.flip(j_nodes, axis=0)
            edge_lengths /= 1000  # convert to km
        else:
            edge_lengths = None
            i_nodes = None
            j_nodes = None
        # Generate random number (0, 1] grid for whole extent
        z_mat = np.random.random(size=(grid_params['nrows'],
                                       grid_params['ncols']))
        # variable for max(qj), used for calclulating G
        # matrix for q_j (zeros)
        dPdt = ((np.nansum(rho_growth.rho) -
                 np.nansum(rho_results[year - ntwrk_model_params['dt']]))) / \
               ntwrk_model_params['dt']
        dPdt_all.append(dPdt)

        rho_k_d_jk = new_station_probability(
            nrows=grid_params['nrows'], ncols=grid_params['ncols'],
            d_origin=d_origin, city_mat=city_mat)

        if edge_lengths is not None:
            rho_k_d_jk = rho_k_d_jk + \
                         edge_weight * np.where(edge_lengths > 0, 1, 0)

        rho_d_max = np.max(rho_k_d_jk)
        # Normalising qj
        qj = copy(rho_k_d_jk)
        qj /= qj.sum()

        mean_qj.append(qj.mean())
        # Actual probability Qj
        Qj = g * dPdt * qj
        new_stns = (Qj > z_mat).astype(int)

        # Restricting total number of stations in each grid cell
        new_stns = np.where((stn_mat + new_stns) <= max_in_grid,
                            new_stns, 0)
        # add boolean matrix to city_mat
        stn_mat += new_stns
        stn_graph, new_nodes = \
            add_new_station_nodes(graph=stn_graph, new_stn_mat=new_stns,
                                  year=year, xmin=xmin, ymin=ymin)
        # return_new_nodes=True,
        add_new_edges(graph=stn_graph, stn_mat=stn_mat,
                      stn_origin=stn_origin, stn_gamma=psi,
                      new_nodes=new_nodes, add_edges=True,
                      i_node=i_nodes, j_node=j_nodes, cost_coeff=cost_coeff,
                      xmin=xmin, ymin=ymin)
        for e in list(stn_graph.edges):
            if e[0] == e[1]:
                print('self loops')

        # SET NECESSARY NODE ATTRIBUTES
        stn_graph = set_accessibility(G=stn_graph)[0]
        # CONVERT TO DATAFRAME FOR USE BY POPULATION
        xy_wbc = pd.DataFrame.from_dict(dict(stn_graph.degree()),
                                        orient='index', columns=['degree'])
        for k in ['lat', 'lon', 'year', 'a_i', 'a_i_inv']:
            xy_wbc[k] = pd.Series(nx.get_node_attributes(stn_graph, k))

        ntwrk_dict[year] = nx.Graph(stn_graph)

        stn_mat = graph_to_array(stn_graph, stn_mat,
                                 [0, 0, xmax-xmin, ymax-ymin])

        ntwrk_df_dict[year] = xy_wbc

    # PHASE I
    if year < ct_year_lim:
        print('No costs')
        rho_growth.set_Ygross_i(average=False)
        rho_growth.run_growth_ts_domain_ygross_only(ts_to_run=1,
                                                    diff_mult=diff_mult)
    # PHASE II
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
    # PHASE III
    elif year >= sink_year and year < regrowth_year:
        # EXTERNAL SINK - Y gross not variable in this phase
        rho_growth.Ygross = model_params['ygross']
        rho_growth.run_growth_ts_domain_sinknegeta(
            ts_to_run=1, remove=False, r_ynet=False,
            sink_rate=model_params['int_sink_rate'], calc_ynet=True,
            ntwrkxy=xy_wbc, cl_fctr=model_params['cr_fctr'],
            ntwrk_fctr=model_params['ntwrk_factor'])
        rho_growth.run_growth_ts_domain_ext_sink_simple(
            ts_to_run=1, sink_rate=model_params['ext_sink_rate'])
        # if q+ also, sink_rate=0.0075
    # PHASE IV
    elif year >= regrowth_year:
        if year == regrowth_year:

            rho_growth.set_Kc(model_params['KctIV'])

        rho_growth.run_growth_ts_domain_r(ts_to_run=1, ntwrkxy=xy_wbc,
            ntwrk_fctr=model_params['ntwrk_factor'],
            cr_fctr=model_params['cr_fctr'])

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
        rho_errors['rmse'].append(rmse(y_pred=rho_growth.rho,
                                       y_true=rho_obs[year]))
        rho_errors['mae'].append(mae(y_pred=rho_growth.rho,
                                     y_true=rho_obs[year]))
        rho_errors['mbe'].append(mbe(y_pred=rho_growth.rho,
                                     y_true=rho_obs[year]))
        rho_errors['rmse_smooth'].append(rmse(y_pred=rho_growth.rho,
                                              y_true=smooth_obs[year]))
        rho_errors['mae_smooth'].append(mae(y_pred=rho_growth.rho,
                                            y_true=smooth_obs[year]))
        rho_errors['mbe_smooth'].append(mbe(y_pred=rho_growth.rho,
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

                if xy_wbc is not None:
                    # NETWORK
                    plot_full_network_rho(stn_graph=stn_graph, xvyv=xvyv,
                        rho=rho_growth.rho, phi=ntwrk_model_params['phi'],
                        g=ntwrk_model_params['g'], N=stn_mat.sum(), year=year,
                        savepath=f"{dir_path}/{year}_simnetwork.png")

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
pickle.dump(rho_errors, open(f"{dir_path}/rmsemaembe.p", "wb"))
pickle.dump(ntwrk_dict, open(f"{dir_path}/sim_network.p", "wb"))
pickle.dump(ntwrk_df_dict, open(f"{dir_path}/sim_network_df.p", "wb"))
if record_details:
    pickle.dump(eta_dict, open(f"{dir_path}/eta.p", "wb"))
    pickle.dump(ct, open(f"{dir_path}/ct_sim.p", "wb"))
    pickle.dump(cr, open(f"{dir_path}/cr_sim.p", "wb"))
    pickle.dump(eta_redist, open(f"{dir_path}/eta_redist.p", "wb"))
    pickle.dump(qm, open(f"{dir_path}/qm.p", "wb"))
    pickle.dump(qneg, open(f"{dir_path}/qneg.p", "wb"))
    pickle.dump(ynet, open(f"{dir_path}/ynet.p", "wb"))

if not os.path.isdir(dir_path + '/rho_txt'):
    os.mkdir(dir_path + '/rho_txt')
if not os.path.isdir(dir_path + '/rho_clusters'):
    os.mkdir(dir_path + '/rho_clusters')

xvyv = pickle.load(open("./growth_results/xvyv.p", "rb"))

for yr in list(rho_obs.keys()):
    rho_to_txt(xv=xvyv['xv'], yv=xvyv['yv'], rho=rho_results[yr],
               savepath=dir_path + f"/rho_txt/rho{yr}.txt")

# Save gm_gi to avoid rerunning each time
if xy_wbc_ready == "record":
    pickle.dump(xy_wbc_dict, open("./core_data/network/xy_wbc_cc.p", "wb"))
pickle.dump(gmgi, open(f"{dir_path}/gmgi.p", "wb"))

if make_plots:
    observed_N = [np.nansum(rho_obs[yr]) for yr in np.arange(1831, 2012, 10)]
    plot_obs_sim_pop(observed_N=observed_N, simulated_N=simulated_N,
                     r=rho_growth.r, save_path=f'{dir_path}/'
                                               f'total_population_sum.png')
