"""
Run to create all plots for the article (best used as an iPython notebook in
VScode)

@author: Isabella Capel-Timms
"""
#TODO: copy all relevant figures (parts and final) to this workspace
#%%
# %load_ext autoreload
# %autoreload 2
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from all_article_plots import *
from plot_growth import plot_rho_log_log_rad, plot_obssim3_multiple, \
    plot_obs_sim_contours, plot_obs_sim_accuracy, plot_total_region_rho, \
    plot_obssim4_multiple, plot_accuracy_scatter, plot_stns_log_log_rad, \
    plot_multiple_sim_network, plot_obs_sim_contours_diff, \
    plot_multiple_obssim_network, plot_multiple_obssim_network_rho, \
    plot_obssim_stncount_profiles, plot_two_xy_mean_years, \
    plot_twocity_contours_diff
from all_article_plots import plot_two_xy_mean_years_paper, observed_npop_nv, \
    model_concept_rho_ntwrk, obs_rho_network_twoyears
import os
from graph_functions import set_accessibility
import geopandas as gpd

#%%
obs_yrs = pickle.load(open('./Data/London/rho_yrs.p', 'rb'))
obs_smooth = pickle.load(open("./Data/London/rho_yrs_smooth.p", 'rb'))
obs_clstrs = {yr: pd.read_csv(
    f"./Data/London/rhoyr_clusters/gl{yr}_1000.csv")[[
    'long', 'lat', 'add.data']] for yr in np.arange(1831, 2012, 10)}
obs_ntwrk_years = \
    pickle.load(open(f'./Data/London/networks_5years_nodup_factcnct.p', 'rb'))
xvyv = pickle.load(open("./Data/London/xvyv.p", "rb"))

gloutline = pickle.load(open('./Data/London/GLoutline.p', 'rb'))
pop_grid = gpd.read_file("./Data/London/reg_grid_pop_1881_2011.shp")

sim_run = 'LdnPopulationGrowth'
sim_rho = pickle.load(open(f"./Results/{sim_run}/rho_results.p", 'rb'))

sim_ntwrk_yrs = \
    pickle.load(open("./Results/NetworkGrowth/simulated_networks.p", 'rb'))

cluster_rho = 1000
sim_clstrs = {yr: pd.read_csv(f"./Results/{sim_run}/rho_clusters/"
                              f"rho{yr}_{cluster_rho}.txt")[[
    'long', 'lat', 'add.data']] for yr in np.arange(1831, 2012, 10)}

sim_ai_inv = pickle.load(
    open("./NetworkRuns/tryingimprovement19/sim_ai_inv.p", 'rb'))
sim_ntwrk_deg_count = pickle.load(
    open("./NetworkRuns/tryingimprovement19/sim_deg_count.p", 'rb'))
sim_ntwrk_deg_pdf = pickle.load(
    open("./NetworkRuns/tryingimprovement19/sim_deg_pdf.p", 'rb'))
node_deg_root_sim = pickle.load(
    open("./NetworkRuns/tryingimprovement19/node_deg_root_sim.p", 'rb'))

obs_ai_inv = pickle.load(
    open("./plots/paperplots/figure_results/obs_ai_inv_w1891.p", "rb"))
obs_ntwrk_deg_count = pickle.load(
    open("./network_scale_data/obs_deg_count.p", 'rb'))
obs_ntwrk_deg_pdf = pickle.load(
    open("./network_scale_data/obs_deg_pdf.p", 'rb'))
node_deg_root_obs = pickle.load(
    open("./network_scale_data/node_deg_root_obs.p", 'rb'))

#TODO: replicate new_network_results.py from urban_growth
#TODO: copy above results over, rename filepath
cpld_rho = pickle.load(open("./Results/CoupledSystem/rho_results.p", 'rb'))
cpld_ntwrks = \
    pickle.load(open("./Results/CoupledSystem/sim_network.p", 'rb'))
cpld_ntwrk_dfs = \
    pickle.load(open("./Results/CoupledSystem/sim_network_df.p", 'rb'))

syd_run = 'Results/SydPopulationGrowth/optimum_params1881_D1_kappa4'
# syd_sim = pickle.load(open("./SydneyRuns/optimum_params/rho_results.p", 'rb'))
syd_sim = \
    pickle.load(open(f"./{syd_run}/rho_results.p", 'rb'))
syd_obs = pickle.load(open("./Data/Sydney/sydneymet_rho_yrs.p", 'rb'))
xvyv_syd = pickle.load(open("./Data/Sydney/sydney_xvyv.p", "rb"))
xvyv_syd = {k: xvyv_syd[k][12:117, 58:142] for k in xvyv_syd.keys()}
msoutline = \
    pickle.load(open('./Data/Sydney/MSoutline.p', 'rb'))[12:117, 58:142]

sydney_pop_grid = gpd.read_file(
    './Data/Sydney/sydney_reg_pop_grid_1851_2011.shp')

sydney_network = \
    pickle.load(open("./Data/Sydney/sydneymet_ttnetwork_years.p", 'rb'))

sydobs_clstrs = {yr: pd.read_csv(
    f"./Data/Sydney/rhoyr_clusters/ms{yr}_300.csv")[[
    'long', 'lat', 'add.data']] for yr in np.arange(1851, 2012, 10)}
sydsim_clstrs = {yr: pd.read_csv(f"./{syd_run}/"
                                 f"rho_clusters/rho{yr}_300.txt")[[
              'long', 'lat', 'add.data']] for yr in np.arange(1851, 2012, 10)}

save_dir_path = f"./Results/{sim_run}/paper_style_plots/"

all_dir_path = f"./Figures/"
if not os.path.isdir(save_dir_path):
    os.mkdir(save_dir_path)

for yr in [1891, 1951, 1971, 1991, 2011]:
    pos_dict = \
        {n: (obs_ntwrk_years[yr].nodes[n]['lon']*1000,
             obs_ntwrk_years[yr].nodes[n]['lat']*1000)
             for n in obs_ntwrk_years[yr].nodes}
    nx.set_node_attributes(obs_ntwrk_years[yr], pos_dict, name='pos')
#%%
## FIG 1a
obs_rho_network_twoyears(pop_grid=pop_grid, years=[1891, 2011],
    ntwrk=obs_ntwrk_years, xv=xvyv['xv_med'], yv=xvyv['yv_med'],
    savepath=f'{all_dir_path}/figparts/domain_1881_2011_small_rho_centred_axlab.png')

#%%
## FIG 1b
model_concept_rho_ntwrk(pop_grid=pop_grid, ntwrk=obs_ntwrk_years, cbar='rho',
    savepath=f'{all_dir_path}/figparts/modellayers_rho.png')
# %%
## FIG 1c,d
# centre 30:37, 39:46
# suburbs 30:43, 17:33
observed_npop_nv_small(rho_dict=obs_yrs, ntwrk=obs_ntwrk_years,
                 years=np.arange(1831, 2012, 10),
                 ybound=(xvyv['yv'].min(), xvyv['yv'].max()),
                 xbound=(xvyv['xv'].min(), xvyv['xv'].max()),
                 centre=(30,37,39,46), suburbs=(30,43,17,33),
                 savepath=f'{all_dir_path}/figparts/npopnv_small.png')
#%%
## FIG 2a
plot_two_xy_mean_years_paper(
    rho_dict1=sim_rho, rho_dict2=obs_yrs, years=[1891, 1951, 1991, 2011],
    xv=xvyv['xv_med'], yv=xvyv['yv_med'], cmaps=('Set1', 'Set1'),
    savepath=f"{save_dir_path}/figparts/xymeanprofiles_centred.png",
    borders=(6,6))
#%%
## FIG 2b
plot_rho_log_log_rad(
    pop_obs=obs_yrs, pop_sim=sim_rho, years=[1891, 1951, 1991, 2011],
    xvyv=xvyv, city_outline=gloutline, obs_cluster_dict=obs_clstrs,
    sim_cluster_dict=sim_clstrs, borders=(6, 6), radius_line=True,
    log_mode='cumsum', inset=True,
    savepath=f"{save_dir_path}/figparts/rho_spatial_scaling",)

# plawres = pickle.load(open(f"{save_dir_path}/rho_spatial_scaling.p", 'rb'))
#%%
## FIG 2c
npop_nv_simobs_comparison(
    rho_obs=obs_yrs, rho_sim=sim_rho, ntwrk_obs=obs_ntwrk_years,
    ntwrk_sim=sim_ntwrk_yrs, savepath=f"{all_dir_path}/figparts/"\
        "final_totalstns_rho_scatter_obsobs.png")
#%%
## FIG 2d
obs_stn_count_dict = \
    pickle.load(open("./Results/NetworkGrowth/obs_stncount.p", "rb"))
sim_stn_count_dict = \
    pickle.load(open("./Results/NetworkGrowth/sim_stncount.p", 'rb'))

plot_stns_log_log_rad(pop_obs=obs_yrs, stn_obs=obs_stn_count_dict,
    obs_cluster_dict=obs_clstrs, pop_sim=sim_rho, stn_sim=sim_stn_count_dict,
    sim_cluster_dict=sim_clstrs, city_outline=gloutline, borders=(6,6),
    years=[1891, 1951, 1991, 2011], xvyv=xvyv, inset=True,
    radius_top=False, radius_line=True, log_mode='cumsum',
    savepath=f"{all_dir_path}/figparts/stn_spatialscaling")

#%%
## FIG 3
plot_twocity_contours_diff(
    ldn_obs=obs_yrs[2011], ldn_sim=sim_rho[2011], syd_obs=syd_obs[2011],
    syd_sim=syd_sim[2011], levels=[0.3, 1, 1.5, 2, 3, 5, 10], xvyv_ldn=xvyv,
    xvyv_syd=xvyv_syd, figsize=(9,5), borders=(6,6),
    savepath=f"{all_dir_path}/figparts/ldnsyd_diff.png",
    cbar_savepath=f"{all_dir_path}/figparts/ldnsyddiff_cbar.png")
#%%
## FIG 4
#TODO: see above re this plot
plot_spatial_stats_condense( #TODO: copy this function over to plot_growth
    obs_av_data=obs_ai_inv, obs_pdf_data=obs_ntwrk_deg_pdf, 
    obs_hierarchy=node_deg_root_obs, obs_count_data=obs_ntwrk_deg_count, 
    sim_av_data=sim_ai_inv, sim_pdf_data=sim_ntwrk_deg_pdf,
    sim_hierarchy=node_deg_root_sim, sim_count_data=sim_ntwrk_deg_count, 
    yrs=[1891, 1951, 1991, 2011], 
    savepath="./NetworkRuns/tryingimprovement19/final_networkspatialanalysislog_stations_nopl.png")

#%%
## FIG 5 
# TODO: create functions in plot_growth for plot_coupled_results
#%%
two_yr_ntwrk = {1891: sydney_network[1891], 2011: sydney_network[2011]}

ntwrk_pos_2011 = {n: [sydney_network[2011].nodes[n]['lon']/1000 - 9688,
                      sydney_network[2011].nodes[n]['lat']/1000 - 4423]
                    for n in sydney_network[2011].nodes}
ntwrk_pos_1891 = {n: [sydney_network[1891].nodes[n]['lon']/1000 - 9688,
                      sydney_network[1891].nodes[n]['lat']/1000 - 4423]
                    for n in sydney_network[1891].nodes}

nx.set_node_attributes(two_yr_ntwrk[1891], ntwrk_pos_1891, 'pos')
nx.set_node_attributes(two_yr_ntwrk[2011], ntwrk_pos_2011, 'pos')

obs_rho_network_twoyears_array(
    rho_arr=syd_obs, years=[1891,2011], ntwrk=two_yr_ntwrk,
    xv=xvyv_syd['xv_med'], yv=xvyv_syd['yv_med'], vmin=300, cbar_side='right',
    savepath=f"{all_dir_path}/figparts/obs18912011.png")

#%%
metcentre = (38,45,59,65)
metsuburb = (35,42, 52,58)
observed_npop_nv_sydney(
    rho_dict=syd_obs, ntwrk=sydney_network, years=np.arange(1851, 2012, 10),
    ybound=(xvyv_syd['yv'].min(), xvyv_syd['yv'].max()),
    xbound=(xvyv_syd['xv'].min(), xvyv_syd['xv'].max()),
    centre=metcentre, suburbs=metsuburb,
    savepath=f'{all_dir_path}/figparts/npopnv_centresuburbs_metsyd.png')
#%%
plot_two_xy_mean_years_paper(
    rho_dict1=syd_sim, rho_dict2=syd_obs, years=[1891, 1951, 1991, 2011],
    xv=xvyv_syd['xv_med'], yv=xvyv_syd['yv_med'], cmaps=('Set1', 'Set1'),
    savepath=f"{all_dir_path}/figparts/xymeanprofiles_sydneymet_centre.png",
    borders=(6,6))

plot_rho_log_log_rad(
    pop_obs=syd_obs, pop_sim=syd_sim, years=[1891, 1951, 1991, 2011],
    city_outline=msoutline, obs_cluster_dict=sydobs_clstrs, xvyv=xvyv_syd,
    sim_cluster_dict=sydsim_clstrs, borders=(6,6), radius_line=True,
    log_mode='cumsum', inset=True, centre=[62,43], #centre=[120,55],
    savepath=f"{all_dir_path}/figparts/rhoscaling_sydneymet",)
#%%
npop_nv_simobs_comparison(
    rho_obs=syd_obs, rho_sim=syd_sim, ntwrk_obs=sydney_network,
    ntwrk_sim=None, stn_count='Syd', savepath=f"{all_dir_path}/figparts/"\
        "sydney_totalstns_rho_scatter_obsobs.png")
#%%
### COUPLED RESULTS
## FIG SI.8
plot_multiple_obssim_network_rho(
    rho_sim={yr: cpld_rho[yr] for yr in [1891, 1951, 1991, 2011]},
    rho_obs={yr: obs_yrs[yr] for yr in [1891, 1951, 1991, 2011]},
    ntwrk_sim=cpld_ntwrks, ntwrk_obs=obs_ntwrk_years,
    rho_cmap='plasma', av_cmap='Greens', years=[1891, 1951, 1991, 2011],
    borders=(6,6), rho_vmin=1000, xv=xvyv['xv_med'], yv=xvyv['yv_med'],
    savepath=f"{all_dir_path}/figparts/coupled_rhonetworks_centred.png",
    cbar_savepath=f"{all_dir_path}/figparts/coupled_rhonetworks_cbar.png")

# plot_two_xy_mean_years_paper(
#     rho_dict1=cpld_rho, rho_dict2=obs_yrs, years=[1891, 1951, 1991, 2011],
#     xv=xvyv['xv_med'], yv=xvyv['yv_med'], cmaps=('Set1', 'Set1'),
#     savepath=f"./CoupledRuns/newnetworkrun/coupled_xymeanprofiles_centred.png",
#     borders=(6,6))
#%%
## FIG SI.2
plot_total_region_rho(obs_rho=obs_yrs, sim_rho=sim_rho, suburbs=(30,43,17,33),
    years=np.arange(1831, 2012, 10), centre=(30,37,39,46), ymin=1e6,
    savepath=f'{all_dir_path}/figparts/simobstotalcentsuburbs.png')

# plot_total_region_rho(obs_rho=syd_obs, sim_rho=syd_sim, centre=(49,56,117,123),
#     suburbs=(47,54, 110,116), years=np.arange(1851, 2012, 10),
#     savepath=f'./{syd_run}/simobstotalcentsuburbs_sydneymet.png')

#%%
metcentre = (38,45,59,65)
metsuburb = (35,42, 52,58)
# plot_total_region_rho(obs_rho=syd_obs, sim_rho=syd_sim, centre=metcentre,
#     suburbs=metsuburb, years=np.arange(1851, 2012, 10),
#     savepath=f'./{syd_run}/simobstotalcentsuburbs_sydneymet.png')

plot_obs_sim_contours_diff(
    obs_yrs=syd_obs, sim_yrs=syd_sim, xv=xvyv_syd['xv_med'],
    yv=xvyv_syd['yv_med'], years=[1891, 1951, 1991, 2011],
    levels=[0.3, 1, 1.5, 2, 3, 5, 10], borders=(6,6), figsize=(6,9),
    savepath=f"{all_dir_path}/figparts/sydmet_obssimcontours_diff.png",
    cbar_savepath=f"{all_dir_path}/figparts/sydmet_obssimcontours_cbar_diff.png")

#%%
plot_twocity_contours_diff(
    ldn_obs=obs_yrs[2011], ldn_sim=sim_rho[2011], syd_obs=syd_obs[2011],
    syd_sim=syd_sim[2011], levels=[0.3, 1, 1.5, 2, 3, 5, 10], xvyv_ldn=xvyv,
    xvyv_syd=xvyv_syd, figsize=(9,5), borders=(6,6),
    savepath=f"{all_dir_path}/figparts/ldnsyd_diff.png",
    cbar_savepath=f"{all_dir_path}/figparts/ldnsyddiff_cbar.png")
#%%
## FIG SI.3
plot_obssim3_multiple(
    rho_sim=sim_rho, rho_obs=obs_yrs, xv=xvyv['xv'], yv=xvyv['yv'],
    years=[1891, 1951, 1991, 2011], cmap="plasma", cmap_diff="bwr",
    borders=(6, 6), show=True,
    save_path=f"{all_dir_path}/figparts/obssimdiff_years.png",
    cbar_savepath=f"{all_dir_path}/figparts/obssimdiff_years_cbar.png")
#%%
## FIG SI.4
# for yr in sim_ntwrk_yrs.keys():
#     sim_ntwrk_yrs[yr] = set_accessibility(G=sim_ntwrk_yrs[yr])[0]
# ntwrk_plot = {yr: set_accessibility(G=sim_ntwrk_yrs[yr])[0]
#               for yr in [1891, 1951, 1991, 2011]}

plot_multiple_sim_network(
    rho=sim_rho, ntwrk=ntwrk_plot, rho_cmap='plasma', av_cmap='Greens',
    years=[1891, 1951, 1991, 2011], borders=(6, 6), rho_vmin=1000,
    savepath=None)
#%%
for yr in sim_ntwrk_yrs.keys():
    sim_ntwrk_yrs[yr] = set_accessibility(G=sim_ntwrk_yrs[yr])[0]
ntwrk_plot = {yr: set_accessibility(G=sim_ntwrk_yrs[yr])[0]
              for yr in [1891, 1951, 1991, 2011]}

plot_multiple_obssim_network(
    rho={yr: obs_yrs[yr] for yr in [1891, 1951, 1991, 2011]},
    ntwrk_sim=sim_ntwrk_yrs, ntwrk_obs=obs_ntwrk_years, xv=xvyv['xv_med'],
    yv=xvyv['yv_med'], rho_cmap='plasma', av_cmap='Greens',
    years=[1891, 1951, 1991, 2011], borders=(6, 6), rho_vmin=1000,
    savepath=f'{all_dir_path}/figparts/multiobssim_ntwrk_final_centred.png',
    cbar_savepath=f'{all_dir_path}/figparts/multiobssim_ntwrk_final_cbar.png')
#%%
# CONTOURS
plot_obs_sim_contours(
    obs_yrs=obs_yrs, sim_yrs=sim_rho, years=[1891, 1951, 1991, 2011],
    levels=[300, 1000, 1500, 2000, 3000, 5000, 10000], borders=(6,6),
    xv=xvyv['xv_med'], yv=xvyv['yv_med'],
    savepath=f"{all_dir_path}/figparts/obssimcontours.png",
    cbar_savepath=f"{all_dir_path}/figparts/obssimcontours_cbar.png")

#%%
# CONTOURS WITH DIFF
plot_obs_sim_contours_diff(obs_yrs=obs_yrs, sim_yrs=sim_rho, xv=xvyv['xv_med'],
    yv=xvyv['yv_med'], years=[1891, 1951, 1991, 2011],
    levels=[0.3, 1, 1.5, 2, 3, 5, 10], borders=(6,6),
    savepath=f"{all_dir_path}/figparts/obssimcontours_diff_centre.png",
    cbar_savepath=f"{all_dir_path}/figparts/obssimcontours_cbar_diff.png")

#%%
## FIG SI.3 SYDNEY
# syd_sim = pickle.load(open("./SydneyRuns/optimum_params/rho_results.p", 'rb'))
syd_sim = pickle.load(open(f"./Results/Sydney/rho_results.p", 'rb'))
syd_obs = pickle.load(open("./Data/Sydney/sydneymet_rho_yrs.p", 'rb'))
xvyv_syd = pickle.load(open("./Data/Sydney/sydney_xvyv.p", "rb"))
xvyv_syd = {k: xvyv_syd[k][12:117, 58:142] for k in xvyv_syd.keys()}

plot_obs_sim_contours_diff(obs_yrs=syd_obs, sim_yrs=syd_sim, xv=xvyv['xv_med'],
    yv=xvyv['yv_med'], years=[1891, 1951, 1991, 2011],
    levels=[0.3, 1, 1.5, 2, 3, 5, 10], borders=(6,6), figsize=(7.5,9),
    savepath=f"./{all_dir_path}/figparts/sydmet_obssimcontours_diff.png",
    cbar_savepath=f"./{all_dir_path}/figparts/sydmet_obssimcontours_cbar_diff.png")
#%%
## SMOOTH CONTOURS
plot_obs_sim_contours(
    obs_yrs=obs_smooth, sim_yrs=sim_rho, years=[1891, 1951, 1991, 2011],
    levels=[300, 1000, 1500, 2000, 3000, 5000, 10000], borders=(6,6),
    savepath=f"{all_dir_path}/figparts/obssimsmoothcontours.png",)
#%%
# CHOROPLETH ACCURACY
plot_obs_sim_accuracy(obs_yrs=obs_yrs, sim_yrs=sim_rho,
                      years=[1891, 1951, 1991, 2011],
                      savepath=f"{all_dir_path}/figparts/obssim_accuracy")

#%%
# plot_obssim4_multiple(rho_sim=sim_rho, rho_obs=obs_yrs, xv=xvyv['xv'],
#     yv=xvyv['yv'], years=[1891, 1951, 1991, 2011], cmap="viridis",
#     cmap_diff="bwr", perc_diff=False, show=True,
#     save_path=f"{save_dir_path}/obssimdiff4_years.png",
#     cbar_savepath=f"{save_dir_path}/obssimdiff4_years_cbar.png")

#%%
for yr in [1891, 1951, 1991, 2011]:
    plot_accuracy_scatter(obs_data=obs_yrs[yr], sim_data=sim_rho[yr],
        varobs=r'$\rho_{obs}$ (1000 cap km$^{-2}$)',
        varsim=r'$\rho_{sim}$ (1000 cap km$^{-2}$)',
        borders=(6,6), log=True, savepath=f"{save_dir_path}/log11_{yr}.png",
        cmap='plasma', axlim=None, r2_scores=True, year=yr)

#%%
# STATION COUNT PROFILES
# for yr in [1891, 1951, 1991, 2011]:
#     pos_dict = \
#         {n: (obs_ntwrk_years[yr].nodes[n]['lon'] - 490,
#              obs_ntwrk_years[yr].nodes[n]['lat'] - 148)
#              for n in obs_ntwrk_years[yr].nodes}
#     nx.set_node_attributes(obs_ntwrk_years[yr], pos_dict, name='pos')

for yr in [1891, 1991, 2011]:
    plot_obssim_stncount_profiles(obs_ntwrks=obs_ntwrk_years, borders=(6,6),
        sim_ntwrks=sim_ntwrk_yrs, year=yr, xvyv=xvyv, title=yr,
        savepath=f"{all_dir_path}/figparts/xyprofile_paper_{yr}.png")

plot_obssim_stncount_profiles(obs_ntwrks=obs_ntwrk_years, borders=(6,6),
    sim_ntwrks=sim_ntwrk_yrs, year=1951, xvyv=xvyv, title=1951, legend=True,
    savepath=f"{all_dir_path}/figparts/xyprofile_paper_{1951}.png")

#%%
# XY PROFILES MORE YEARS

# plot_two_xy_mean_years(rho_dict1=sim_rho, rho_dict2=obs_yrs,
#     years=[1851, 1891, 1931, 1961, 1991, 2011], xv=xvyv['xv'], yv=xvyv['yv'],
#     cmaps=('winter', 'spring'), borders=(6,6),
#     savepath=f'{all_dir_path}/meanxy_sim.png',
#     lr_titles=['Simulated', 'Observed'])

plot_two_xy_mean_years(rho_dict1=cpld_rho, rho_dict2=obs_yrs,
    years=[1851, 1891, 1931, 1961, 1991, 2011], xv=xvyv['xv_med'],
    yv=xvyv['yv_med'], cmaps=('winter', 'spring'), borders=(6,6),
    savepath=f'{all_dir_path}/figparts/meanxy_cpld_centred.png',
    lr_titles=['Coupled', 'Observed'])