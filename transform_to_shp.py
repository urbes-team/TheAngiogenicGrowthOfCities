import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from copy import copy

grid = gpd.read_file('./Data/London/empty_regular_grid.shp').set_index('num_codes')

grid['x'] = [np.dstack(row.geometry.exterior.coords.xy)[0,0,0]
             for i, row in grid.iterrows()]
grid['y'] = [np.dstack(row.geometry.exterior.coords.xy)[0,0,1]
             for i, row in grid.iterrows()]

coupled = copy(grid)
population = copy(grid)

for yr in np.arange(1831, 2012, 10):
    rho = pd.read_csv(f'./Results/LdnPopulationGrowth/rho_txt/rho{yr}.txt')
    rho.rename(columns={'rho': f'rho{yr}'}, inplace=True)
    rho['xv'] = rho['xv'] * 1000
    rho['yv'] = rho['yv'] * 1000

    population =\
        population.merge(rho, left_on=['x', 'y'], right_on=['xv', 'yv']).drop([
            'xv', 'yv'], axis=1)

print(population.head())

for yr in np.arange(1831, 2012, 10):
    rho = pd.read_csv(f'./Results/LdnPopulationGrowth/rho_txt/rho{yr}.txt')
    rho.rename(columns={'rho': f'rho{yr}'}, inplace=True)
    rho['xv'] = rho['xv'] * 1000
    rho['yv'] = rho['yv'] * 1000

    coupled =\
        coupled.merge(rho, left_on=['x', 'y'], right_on=['xv', 'yv']).drop([
            'xv', 'yv'], axis=1)

print(coupled.head())

population.to_file('./Results/LdnPopulationGrowth/popgrowth.shp')
coupled.to_file('./Results/CoupledSystem/coupledgrowth.shp')
