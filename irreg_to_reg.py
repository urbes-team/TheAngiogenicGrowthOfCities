"""
Generic script for converting population data with irregular spatial units to
a regular grid.

@author: Isabella Capel-Timms
"""
# %%
import geopandas as gpd
import pandas as pd
from matplotlib.pyplot import grid
from shapely.geometry import Polygon
import numpy as np
import math
import matplotlib.pyplot as plt


# %%


def round_down(x, a):
    return math.floor(x / a) * a


def round_nearest(x, a):
    return round(x / a) * a


def round_up(x, a):
    return math.ceil(x / a) * a


def irreg_to_reg(pop_df_path, year, dir_path, broad_extent, pop_grid):
    """
    Generic conversion of irregular population data to regular grid.

    :param pop_df_path: str
        Path to CSV of population data
    :param year: int
        Year of population
    :param dir_path: str
        Path of directory holding irregular shapefiles
    :param broad_extent: geopandas.DataFrame

    :param pop_grid:

    :return:
    """
    pop_df = pd.read_csv(pop_df_path)
    if 'g_unit' in pop_df.columns:
        pop_df.rename(columns={'g_data': "population", 'g_unit': "code"},
                      inplace=True)

    # Import shapefile of irregular spatial units at same CRS and with spatial
    # unit codes as index. This could be different across years e.g. London
    shp_gdf = gpd.read_file(f"{dir_path}/1881_parishes.shp") \
            .to_crs("epsg:27700")[['G_UNIT', 'geometry']] \
            .rename(columns={'G_UNIT': 'label'})

    shp_gdf['Area'] = shp_gdf.area

    # clip census shapefile to broad extent to trim data
    shp_gdf = gpd.overlay(broad_extent, shp_gdf, how="intersection")
    # join population data
    pop_shp_gdf = shp_gdf.merge(pop_df, left_on='label', right_on='code')
    # inter = gpd.overlay(grid, shp_1881, how='intersection')
    inter_gdf = gpd.overlay(pop_grid, pop_shp_gdf, how='intersection')
    # calculate area of new polygons
    inter_gdf["new_areas"] = inter_gdf.area
    # population * (new /original area)
    inter_gdf["re_pop"] = inter_gdf['population'] * \
                          (inter_gdf['new_areas'] / inter_gdf['Area'])
    # sum over each grid space somehow?
    inter_pop_df = inter_gdf.groupby('num_codes')['re_pop'].sum().round(0)
    inter_pop_df.rename(f"{year}_pop", inplace=True)
    # add population to grid space
    pop_grid = pop_grid.merge(inter_pop_df, left_on='num_codes',
                              right_on='num_codes', how='outer')

    return pop_grid


# %%
city_outline = gpd.read_file("./city-extent.shp") \
    .set_index('GSS_CODE')
rail_stns = gpd.read_file("./rail_under_shp/"
                          "rail_ug_dlr_aligned.shp").to_crs("epsg:27700") \
    .set_index('index')

# %%
# boundary increase factor accounts for influence of diffusion and boundary
# conditions later on
bndry_inc_fact = 6
width = 1000  # horizontal resoltuion of grid
height = 1000  # vertical resolution of grid
# btb = boroughs.total_bounds
# Central grid cell is on centre of city for radial profile purposes
cent = city_outline.geometry.bounds  # City of London bounds
centre = [cent[0] + 0.5 * (cent[2] - cent[0]), cent[1] + 0.5 * (cent[3] - cent[1])]

centre = [round_down(centre[0], width), round_down(centre[1], height)]

centreldn = gpd.GeoDataFrame(geometry=gpd.points_from_xy([centre[0]],
                                                         [centre[1]]),
                             crs={'init': 'epsg:27700'})
xmin, ymin, xmax, ymax = rail_stns.total_bounds

xmin = round_down(xmin, width) - bndry_inc_fact * width
ymin = round_down(ymin, width) - bndry_inc_fact * height
xmax = round_up(xmax, height) + bndry_inc_fact * width
ymax = round_up(ymax, height) + bndry_inc_fact * height

if abs(centre[0] - xmin) > abs(xmax - centre[0]):
    xmax = centre[0] + (centre[0] - xmin)
else:
    xmin = centre[0] - (xmax - centre[0])

if abs(centre[1] - ymin) > abs(ymax - centre[1]):
    ymax = centre[1] + (centre[1] - ymin)
else:
    ymin = centre[1] - (ymax - centre[1])

cols = list(np.arange(xmin, xmax + width, width))
rows = list(np.arange(ymin, ymax + height, height))

# %%
polys = []
num_codes = []
for i, x in enumerate(cols[:-1]):
    for j, y in enumerate(rows[:-1]):
        polys.append(Polygon([(x, y), (x + width, y), (x + width, y + height),
                              (x, y + height)]))
        num_codes.append(f"{i}_{j}")
d = {'num_codes': num_codes, 'geometry': polys}
grid = gpd.GeoDataFrame(d).set_crs(epsg=27700)
pop_grid = gpd.GeoDataFrame(d).set_crs(epsg=27700)
# grid.to_file("grid.shp")

# %%
# accounts for preserving pop dens when cutting off irregular gridspaces
brd_fact = 10
# create broad extent +/- 10 km
broad_extent = [Polygon([(xmin - brd_fact * width, ymin - brd_fact * height),
                         (xmax + brd_fact * width, ymin - brd_fact * height),
                         (xmax + brd_fact * width, ymax + brd_fact * height),
                         (xmin - brd_fact * width, ymax + brd_fact * height)])]

broad_extent = gpd.GeoDataFrame({'geometry': broad_extent}).set_crs(epsg=27700)

dir_path = "./dir"
years = [1821, 1831, 1841, 1851, 1881, 1891, 1901, 1911, 1921, 1931, 1951,
         1961, 1971, 1981, 1991, 2001, 2011]
# years = [2001]
# %%
pop_yr = {}
for yr in years:
    print(yr)

    pop_grid[f"{yr}_pop"] = \
        irreg_to_reg(pop_df_path=f"./populationCSVs/{yr}_pop.csv",
                     year=yr, dir_path=dir_path, broad_extent=broad_extent,
                     pop_grid=pop_grid)

# %%
# Might need to fix NaN values

# %%
pop_grid.to_file(
    './Data/city_reg_grid_pop_1881_2011.shp')