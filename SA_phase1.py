"""
Run for sensitivity analysis of Phase I of the London case, including finding
optimum parameters.
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
import os
import sys


def SA_phase1(ygross0_par, ygrosse_par, d_par, run_no):
    root_dir = '.'
    os.chdir(root_dir)
    print('changed dir')

    run_name = f'SArun_phase1_yg0{np.format_float_scientific(ygross0_par)}_' \
               f'yge{np.format_float_scientific(ygrosse_par)}_' \
               f'D{np.format_float_scientific(d_par)}'

    dir_path = f'/home/ucesica/Scratch/SAphase1saltelli/run_{int(run_no)}'
    print(os.getcwd())
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    # for basic inclusion of regimes. Only logistic growth and income densification
    # before this year
    ct_year_lim = 1891

    sink_year = 1951  # 1951
    if sink_year != 0:
        int_sink_rate = 0.009
        ext_sink_rate = 0.0026  # for linear sink
    else:
        int_sink_rate = 0.009  # 0.013
        ext_sink_rate = 0.0026  # for linear sink

    regrowth_year = 1992

    ygross_var = True

    method = 'domain'
    ntwrk_factor = 'a_i_inv'
    start_yr = 1831
    end_yr = 1902

    # None, "record" or filepath
    file_paths = {'xy_wbc_fp': "./Data/London/xy_wbc_cc.p",
        'pop_grid_fp': './Data/London/reg_grid_pop_1881_2011_'
                       'nullfixed_allyears.shp',
        'rho_yrs_fp': "./Data/London/rho_yrs.p",
        'network_years_fp': './Data/London/networks_5years_nodup_factcnct.p',
        'smooth_yrs_fp': "./Data/London/rho_yrs_smooth.p"}

    # %%
    # # initiate with 1831 population data
    grid_params = {'dx': 1, 'dy': 1, 'dt': 1, 'tot_ts': 180}

    model_params = {'method': method, 'ntwrk_factor': ntwrk_factor,
                    'cr_fctr': 'rhohist', 'start_yr': start_yr,
                    'end_yr': end_yr, 'beta': 2, 'tau': 1, 'kappa': 2e-4,
                    'mu': 0.33, 'nu': 2.33, 'gm': 1.8, 'd': d_par, 'dI': d_par,
                    'r': 158.5, 'ygross': 5, 'ygross0': ygross0_par,
                    'ygross_e': ygrosse_par, 'phi': None, 'chi': None,
                    'int_sink_rate': int_sink_rate,
                    'ext_sink_rate': ext_sink_rate}

    xy_wbc_ready = pickle.load(open(file_paths['xy_wbc_fp'], "rb"))
    gen_network = False

    if not gen_network:
        if 'network_years_fp' not in file_paths.keys():
            file_paths['network_years_fp'] = \
                f'{root_dir}/Data/London/networks_5years_nodup_factcnct.p'
        ntwrk_years = pickle.load(open(file_paths['network_years_fp'], 'rb'))

    rho_obs = pickle.load(open(file_paths['rho_yrs_fp'], "rb"))

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

    # Load the rail network xy of stations
    print('Loading network')
    xy = pd.read_csv(f'{root_dir}/Data/London/ldn_stn_xy_w27700.csv',
                     header=0, index_col=0)[['lon_m', 'lat_m', 'Year']] \
        .rename(columns={'lon_m': 'lon', 'lat_m': 'lat'})

    xy['lon'] = xy['lon'] / 1000.  # m to km
    xy['lat'] = xy['lat'] / 1000.

    if ct_year_lim == None:
        ct_year_lim = xy['Year'].min()
        model_params['ct_year'] = ct_year_lim

    # find source and target nodes for all edges
    ldn_conx = pd.read_csv(f'{root_dir}/Data/London/'
                           f'stations_with_yr_norep.csv', header=0)
    G = nx.from_pandas_edgelist(ldn_conx, source='StationA', target='StationB')

    ldn_coords = {}
    for n in list(G.nodes):
        # converting m to km
        ldn_coords[n] = (xy.loc[n, 'lon'], xy.loc[n, 'lat'])

    ldn_full = NetworkGraph(graph=G, coords=ldn_coords, nodes=G.nodes)
    # %%
    network_years = 1


    xy_wbc_dict = {}
    rho_results = {}

    errors = {}

    for year in np.arange(model_params['start_yr'], model_params['end_yr'],
                          network_years):

        print(year)
        # Load network for year and fix connections
        if gen_network:
            Gyr_nodes = xy[xy['Year'] <= year].index.values
            if year < model_params[
                'ct_year']:  # network avoided until t=ct_year
                xy_wbc = None
            elif type(xy_wbc_ready) == dict and len(Gyr_nodes) > 0:
                xy_wbc = xy_wbc_ready[year]
            elif len(Gyr_nodes) > 0:
                ntwrk_yr = \
                    SubGraph(graph=ldn_full.graph,
                             subgraph=nx.Graph(
                                 ldn_full.graph.subgraph(Gyr_nodes)),
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
            # max_nbc.append(0)
        elif year >= min(list(ntwrk_years.keys())):
            # pre-generated networks every 5 years from 1836
            if year < model_params[
                'ct_year']:  # network avoided until t=ct_year
                xy_wbc = None

            else:
                ntwrk_yr = ntwrk_years[year - (year % 5) + 1]

                xy_wbc = pd.DataFrame.from_dict(dict(ntwrk_yr.degree()),
                                                orient='index',
                                                columns=['degree'])
                for k in ['lat', 'lon', 'year', 'WBC', 'CC', 'a_i', 'a_i_inv']:
                    xy_wbc[k] = pd.Series(nx.get_node_attributes(ntwrk_yr, k))

        else:
            xy_wbc = None
        # xy_wbc = None

        if xy_wbc is None:
            print('No costs')
            if ygross_var:
                rho_growth.set_Ygross_i(average=False)
                rho_growth.run_growth_ts_domain_ygross_only(ts_to_run=1)
            else:
                rho_growth.run_growth_ts_domain_nocost(ts_to_run=1)

        elif year < sink_year or sink_year == 0:
            # INTERNAL SINK - Y gross not variable in this phase
            rho_growth.Ygross = model_params['ygross']
            # REDISTRIBUTION OF POPULATION FROM AREAS WITH -VE ETA
            rho_growth.run_growth_ts_domain_sinknegeta(
                ts_to_run=1, remove=False, r_ynet=False,
                sink_rate=model_params['int_sink_rate'])
            # DISTRIBUTED LOGISTIC GROWTH
            rho_growth.run_growth_ts_domain_r(
                ts_to_run=1, ntwrkxy=xy_wbc,
                ntwrk_fctr=model_params['ntwrk_factor'],
                cr_fctr=model_params['cr_fctr'])
        rho_results[year] = rho_growth.rho

        if year in list(rho_obs.keys()):
            errors[year] = {'rmse': [], 'mae': [], 'r2': [], 'rmse_radial': [],
                            'mae_radial': [], 'r2_radial': [],
                            'mae_central': [], 'rmse_central': [],
                            'r2_central': []}

            errors[year]['rmse'].append(rmse(y_pred=rho_growth.rho,
                                             y_true=rho_obs[year]))
            errors[year]['mae'].append(mae(y_pred=rho_growth.rho,
                                           y_true=rho_obs[year]))
            errors[year]['r2'].append(r_squared(y_pred=rho_growth.rho,
                                                y_true=rho_obs[year]))

            errors[year]['rmse_central'].append(
                rmse(y_pred=rho_growth.rho[15:51, 25:59],
                     y_true=rho_obs[year][15:51, 25:59]))
            errors[year]['mae_central'].append(
                mae(y_pred=rho_growth.rho[15:51, 25:59],
                    y_true=rho_obs[year][15:51, 25:59]))
            errors[year]['r2_central'].append(
                r_squared(y_pred=rho_growth.rho[15:51, 25:59],
                          y_true=rho_obs[year][15:51, 25:59]))

            sim_rad = radial_profile(rho_growth.rho)[0]
            obs_rad = radial_profile(rho_obs[year])[0]
            errors[year]['rmse_radial'].append(rmse(y_pred=sim_rad,
                                                    y_true=obs_rad))
            errors[year]['mae_radial'].append(mae(y_pred=sim_rad,
                                                  y_true=obs_rad))
            errors[year]['r2_radial'].append(r_squared(y_pred=sim_rad,
                                                       y_true=obs_rad))

    pickle.dump(rho_results, open(f"{dir_path}/rho_results.p", "wb"))
    pickle.dump(errors, open(f"{dir_path}/error_results.p", "wb"))


if __name__ == "__main__":
    SA_phase1(ygross0_par=float(sys.argv[1]), ygrosse_par=float(sys.argv[2]),
              d_par=float(sys.argv[3]), run_no=int(sys.argv[4]))
