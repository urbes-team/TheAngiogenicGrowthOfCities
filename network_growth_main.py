"""
This script simulates network growth for the London case, initiating with
the real network data and allowing the model to respond to the growth of
the observed population.
Population follows observations only (not modelled) and the transport network is
grown stochastically following Li et al..
"""
# %%
from graph_functions import connect_new_subnetworks
from analysis_functions import mean_smooth
import networkx as nx
import numpy as np
import pickle
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from copy import copy
import os
import datetime
from shapely.geometry import Point
from geocube.api.core import make_geocube
import f90nml


# @jit(nopython=True)
def new_station_probability(nrows, ncols, d_origin, city_mat,
                            edge_lengths=None):
    """
    Calculate relative probability of new station appearing as a function of
    proximity to populations and, optionally, existing lines.
    :param nrows: int
    :param ncols: int
    :param d_origin: numpy.ndarray
    :param city_mat: numpy.ndarray
    :param edge_lengths: numpy.ndarray
    :return:
    """
    qj_mat = np.zeros_like(city_mat)
    for i in np.arange(nrows):  # iterate over rows
        for j in np.arange(ncols):
            # Find d_jk matrix, d_origin centre on cell
            d_jk_mat = d_origin[(nrows - 1 - i):(2*nrows - 1 - i),
                                (ncols - 1 - j):(2*ncols - 1 - j)]
            # Multiply d_jk with w matrix and sum across all elements
            w_d_sum = (d_jk_mat * city_mat).flatten().sum()
            # Sum d_jk
            d_sum = d_jk_mat.flatten().sum()
            # Calculating q_j
            q_j = w_d_sum/d_sum
            qj_mat[i, j] = q_j
            # set qj_mat cell to q_j
    if edge_lengths is not None:
        qj_mat += edge_lengths

    return qj_mat


def init_G(nrows, ncols, d_origin, city_mat, g):
    q_mat = np.zeros_like(city_mat)
    for i in np.arange(nrows):  # iterate over rows
        for j in np.arange(ncols):
            # Find d_jk matrix, d_origin centre on cell
            d_jk_mat = d_origin[(nrows - 1 - i):(2*nrows - 1 - i),
                                (ncols - 1 - j):(2*ncols - 1 - j)]
            # Multiply d_jk with w matrix and sum across all elements
            w_d_sum = (d_jk_mat * city_mat).flatten().sum()
            # Sum d_jk
            d_sum = d_jk_mat.flatten().sum()
            # Calculating q_j
            q_j = w_d_sum/d_sum
            # set q_mat cell to q_j
            q_mat[i, j] = q_j
    G = g / (q_mat.max())
    return G


def add_new_station_nodes(graph, new_stn_mat, xmin=0, ymin=0, year=None):
    """
    Add new station nodes to the graph from array of new stations.

    :param graph: networkx.classes.graph.Graph
        Network graph without new nodes
    :param new_stn_mat: numpy.ndarray
        Boolean array of new stations
    :param xmin:
    :param ymin:
    :param year:
    :return:
    """

    j, i = np.where(new_stn_mat > 0)
    grids = list(zip(j, i))
    new_nodes = []
    old_graph = nx.Graph(graph)
    for g in grids:
        if new_stn_mat[g[0], g[1]] > 1:
            graph.add_nodes_from([graph.number_of_nodes() + 1])
            for p in np.arange(new_stn_mat[g[0], g[1]]):
                node_name = graph.number_of_nodes() + 1
                graph.add_nodes_from([
                    (node_name, {'lat': g[0] + np.random.random() + ymin,
                                 'lon': g[1] + np.random.random() + xmin})])
                nx.set_node_attributes(
                    graph, {node_name: (graph.nodes[node_name]['lon'],
                                        graph.nodes[node_name]['lat'])}, 'pos')

                if year is not None:
                    graph.nodes[node_name]['year'] = year
                new_nodes.append(node_name)

        else:
            node_name = graph.number_of_nodes() + 1
            graph.add_nodes_from([
                (node_name, {'lat': g[0] + np.random.random() + ymin,
                             'lon': g[1] + np.random.random() + xmin})])

            nx.set_node_attributes(
                graph, {node_name: (graph.nodes[node_name]['lon'],
                                    graph.nodes[node_name]['lat'])}, 'pos')

            if year is not None:
                graph.nodes[node_name]['year'] = year
            new_nodes.append(node_name)

    return graph, new_nodes


def add_new_edges(graph, stn_mat, stn_origin, stn_gamma, new_nodes, cost_coeff,
                  add_edges=True, i_node=None, j_node=None,
                  xmin=0, ymin=0, old_graph=None):
    """
    Adds edges between new and existing nodes depending on location.
    :param graph: networkx.classes.graph.Graph
        Network graph including new nodes without edges
    :param stn_mat: numpy.ndarray
    :param stn_origin: numpy.ndarray
    :param stn_gamma: float
    :param new_nodes: numpy.ndarray, list
    :param add_edges: bool
    :param i_node: numpy.ndarray
    :param j_node: numpy.ndarray
    :param old_graph: networkx.classes.graph.Graph
        Not used.
    :return:
    """

    if add_edges and len(new_nodes) > 0:
        # graph = add_nearest_edge(old_graph=old_graph, new_graph=graph,
        #                          new_nodes=new_nodes)
        if i_node is not None and j_node is not None:
            graph, new_nodes = add_on_edges(new_graph=graph, i_node=i_node,
                                            j_node=j_node, new_nodes=new_nodes,
                                            xmin=xmin, ymin=ymin)

        add_gravity_edge(graph=graph, new_nodes=new_nodes, stn_mat=stn_mat,
                         stn_origin=stn_origin, stn_gamma=stn_gamma,
                         cost_coeff=cost_coeff, xmin=xmin, ymin=ymin, xvyv=None)

    if len([c for c in nx.connected_components(graph)]) > 1:
        graph = connect_new_subnetworks(G=graph)

    return graph

def add_nearest_edge(old_graph, new_graph, new_nodes):
    """
    New nodes are connected to the existing network via the nearest existing
    nodes.

    :param old_graph: networkx.classes.graph.Graph
        Graph from previous time step (existing nodes and edges)
    :param new_graph: networkx.classes.graph.Graph
        Graph from current time step with new, unconnected nodes
    :param new_nodes: numpy.ndarray
        List of the names of new nodes
    :return:
    """
    old_g_dict = nx.get_node_attributes(G=old_graph, name='pos')
    old_g_nodes = np.array(list(old_g_dict.keys()))
    old_g_pos = np.array(list(old_g_dict.values()))

    graph_dict = nx.get_node_attributes(G=new_graph, name='pos')

    for n in new_nodes:
        coord = new_graph.nodes[n]['pos']
        if len(old_g_nodes) > 0:
            rel = old_g_pos - coord

            dist = np.sqrt(rel[:, 0]**2 + rel[:, 1]**2)
            min_dist = np.min(dist)
            closest = old_g_nodes[np.argmin(dist)]

            if not new_graph.has_edge(n, closest):
                new_graph.add_edge(n, closest, length=min_dist)
        else:
            closest = None

        if len(new_nodes) > 1:
            new_g_dict = {k:graph_dict[k] for k in new_nodes if k != n}
            new_g_nodes = np.array(list(new_g_dict.keys()))
            new_g_pos = np.array(list(new_g_dict.values()))

            new_rel = new_g_pos - coord
            new_dist = np.sqrt(new_rel[:, 0]**2 + new_rel[:, 1]**2)
            new_min_dist = np.min(new_dist)
            new_closest = new_g_nodes[np.argmin(new_dist)]

            if closest is not None:
                if new_closest != closest:
                    if not new_graph.has_edge(n, new_closest):
                        new_graph.add_edge(n, new_closest, length=new_min_dist)
            else:
                if not new_graph.has_edge(n, new_closest):
                    new_graph.add_edge(n, new_closest, length=new_min_dist)

    return new_graph

def add_gravity_edge(graph, new_nodes, stn_origin, stn_mat, stn_gamma,
                     cost_coeff, xmin=0, ymin=0, xvyv=None):
    """
    New nodes are connected to existing nodes as a function of network density,
    using the 'gravity' of nearby areas.

    :param graph: networkx.classes.graph.Graph
        Graph from current time step with new, unconnected nodes
    :param new_nodes: numpy.ndarray
        List of the names of new nodes
    :param stn_origin: numpy.ndarray
        Matrix of origin that may move around to cover each cell
    :param stn_mat: numpy.ndarray
        Boolean array of new stns
    :param stn_gamma: float
        Exponent for controlling influence of distance for stn_origin
    :param cost_coeff:
        Scalar reflecting cost efficiency of added direct line instead of
        connecting to existing line
    :param xvyv: dict
        x and y mesh grids for stn arrays
    :return:
    """


    stn_df = pd.DataFrame.from_dict(
        nx.get_node_attributes(G=graph, name='pos'), orient='index')\
        .rename(columns={0:'x', 1:'y'})
    stn_df['x'] = stn_df['x'] - xmin
    stn_df['y'] = stn_df['y'] - ymin

    # ensure new edges added to new nodes in order of those closest to centre
    stn_df['centre_dist'] = \
        np.sqrt((stn_df['x'] - (stn_mat.shape[1] / 2)) ** 2 +
                (stn_df['y'] - (stn_mat.shape[0] / 2)) ** 2)
    stn_df = stn_df.sort_values('centre_dist')
    new_nodes = stn_df.index.values[np.isin(stn_df.index.values, new_nodes)]

    graph_dict = nx.get_node_attributes(G=graph, name='pos')
    nrows, ncols = stn_mat.shape
    # FOR EACH NEW NODE N:
    for n in new_nodes:
        n_loc = stn_df.loc[n, ['y', 'x']].values.astype(int)
    # TARGET WHERE MAX (STN_ORIGIN ** STN_GAMMA) * STN_MEAN_SMOOTH * OLD_STN_MAT
        stn_jk_mat = \
            stn_origin[(nrows - 1 - n_loc[0]):(2 * nrows - 1 - n_loc[0]),
                       (ncols - 1 - n_loc[1]):(2 * ncols - 1 - n_loc[1])]

        with np.errstate(divide='ignore'):
            gravity = (stn_jk_mat ** -stn_gamma) * \
                             mean_smooth(stn_mat, 3, average = False) * stn_mat
        gravity = np.where(np.isinf(gravity), 0, gravity)
        t_loc = np.nanargmax(gravity)
        t_loc = np.unravel_index(t_loc, stn_mat.shape)
        t_node = stn_df[(stn_df.astype(int)['x'] - t_loc[1] == 0) &
                        (stn_df.astype(int)['y'] - t_loc[0] == 0)
        ].index.values[0]
        if graph.degree(t_node) >= 10:
            # Another node is chosen if deg(T_node) > 10
            t_loc = np.argsort(gravity.flatten())[-2]
            t_loc = np.unravel_index(t_loc, stn_mat.shape)
            t_node = stn_df[(stn_df.astype(int)['x'] - t_loc[1] == 0) &
                            (stn_df.astype(int)['y'] - t_loc[0] == 0)
                            ].index.values[0]

        # SELECT NODES INSIDE BOUNDING AREA THAT INCLUDES TARGET
        # bounds outside of xmin, ymin to zero
        # TODO: When network is on the real grid, consider diff between np array
        # indices and lon/lat station locations. For now this is shape of array
        if t_loc[1] > n_loc[1]:
            bnd_l_i = np.clip(n_loc[1] - 3, 0, stn_mat.shape[1])
            bnd_r_i = np.clip(t_loc[1] + 1, 0, stn_mat.shape[1])
        else:
            bnd_l_i = np.clip(t_loc[1] - 1, 0, stn_mat.shape[1])
            bnd_r_i = np.clip(n_loc[1] + 3, 0, stn_mat.shape[1])
    # DIRECT PATH LENGTH N -> TARGET
        n_t_dist = np.sqrt((t_loc[0] - n_loc[0])**2 + (t_loc[1] - n_loc[1])**2)

        if t_loc[0] > n_loc[0]:
            bnd_b_j = np.clip(n_loc[0] - 3, 0, stn_mat.shape[0])
            bnd_t_j = np.clip(t_loc[0] + 1, 0, stn_mat.shape[0])
        else:
            bnd_b_j = np.clip(t_loc[0] - 1, 0, stn_mat.shape[0]-1)
            bnd_t_j = np.clip(n_loc[0] + 3, 0, stn_mat.shape[0]-1)

        stns_in_bnds = stn_df[(stn_df.astype(int)['x'] >= bnd_l_i) &
                              (stn_df.astype(int)['x'] <= bnd_r_i) &
                              (stn_df.astype(int)['y'] >= bnd_b_j) &
                              (stn_df.astype(int)['y'] <= bnd_t_j) &
                              (~stn_df.index.isin(new_nodes)) &
                              (~stn_df.index.isin([t_node]))].copy()

        if len(stns_in_bnds) == 0:
            graph.add_edge(n, t_node, length=n_t_dist)
        else:
            # Distance to n in df
            stns_in_bnds.loc[:, 'dist'] = \
                np.sqrt((stns_in_bnds.loc[:, 'x'] - n_loc[1]) ** 2 +
                        (stns_in_bnds.loc[:, 'y'] - n_loc[0]) ** 2)

            paths = np.zeros_like(stns_in_bnds.index.values, dtype=float)
            i = 0
            # Find existing paths to target
            for index, stn in list(stns_in_bnds.iterrows()):
                try:
                    # Prioritise closer nodes by "lowering" Euclidean distance
                    paths[i] = stn['dist'] + \
                               0.75 * nx.shortest_path_length(graph, index, t_node,
                                                       weight='length')
                except:
                # For when there are multiple components (this is fixed later
                # anyway). Gravity is only for general direction in this case
                # Prioritise closer nodes by "lowering" Euclidean distance
                    paths[i] = stn['dist'] + \
                        0.75 * np.sqrt((stn['x'] - t_loc[1]) ** 2 +
                                (stn['y'] - t_loc[0]) ** 2)
                    paths[i] = 9999 + i  # fill value for missing path
                i += 1

            stns_in_bnds['path'] = paths
            # Potential connecting node with the shortest path
            via_node = stns_in_bnds['path'].idxmin()
            # Path via another node must be less than cost_coeffx direct new
            # edge
            if stns_in_bnds.loc[via_node, 'path'] < cost_coeff * n_t_dist:
                if not graph.has_edge(n, via_node):
                    graph.add_edge(n, via_node,
                                   length=stns_in_bnds.loc[via_node, 'dist'])
                elif not graph.has_edge(n, t_node):
                    graph.add_edge(n, t_node, length=n_t_dist)
            else:
                if graph.degree(t_node) >= 10:
                    graph.add_edge(n, via_node,
                                   length=stns_in_bnds.loc[via_node, 'dist'])
                else:
                    graph.add_edge(n, t_node, length=n_t_dist)

        new_nodes = new_nodes[new_nodes != n]

    return graph

def add_on_edges(new_graph, i_node, j_node, new_nodes, xmin=0, ymin=0):
    """

    :param new_graph:
    :param i_node:
    :param j_node:
    :param new_nodes:
    :return:
    """
    stn_df = pd.DataFrame.from_dict(
        nx.get_node_attributes(G=new_graph, name='pos'), orient='index')\
        .rename(columns={0:'x', 1:'y'})
    if new_nodes is not None:
        n_not_added = []
        n_on_edge = {}
        for n in new_nodes:
            n_pos = tuple(np.array(new_graph.nodes[n]['pos'])[::-1].astype(
                int) - np.array([ymin, xmin]).astype(int))
            try:
                if i_node[n_pos] > 0:
                    old_s_t = \
                        np.sort([i_node[n_pos], j_node[n_pos]])
                    n_on_edge[n] = old_s_t
                else:
                    n_not_added.append(n)
            except:
                print(n)
                print(n_pos)
                pass
        if len(n_on_edge) > 0:
            edges_dict = \
                pd.DataFrame.from_dict(n_on_edge, orient='index').astype(int)

            #Can only insert one node on each edge for now
            dup_edge = edges_dict.duplicated(keep=False)
            brk_e = edges_dict[~dup_edge]
            dup_brk_e = edges_dict[dup_edge]
            dup_brk_e = dup_brk_e.assign(
                cat=dup_brk_e[0].where(dup_brk_e.duplicated(keep=False), '')
                    .astype('category').cat.codes)
            if len(edges_dict[dup_edge].index.values) > 0:
                for n in edges_dict[dup_edge].index.values:
                    n_not_added.append(n)

            for n, e in brk_e.iterrows():
                new_graph.remove_edge(e[0], e[1])

                pos_n = np.array(new_graph.nodes[n]['pos'])
                pos_e0 = np.array(new_graph.nodes[e[0]]['pos'])
                pos_e1 = np.array(new_graph.nodes[e[1]]['pos'])

                rel0 = pos_n - pos_e0
                rel1 = pos_n - pos_e1
                dist0 = np.sqrt(rel0[0]**2 + rel0[1]**2)
                dist1 = np.sqrt(rel1[0]**2 + rel1[1]**2)

                if not new_graph.has_edge(n, e[0]):
                    new_graph.add_edge(n, e[0], length=dist0)
                if not new_graph.has_edge(n, e[1]):
                    new_graph.add_edge(n, e[1], length=dist1)

            for cat in np.unique(dup_brk_e['cat'].values):
                group = dup_brk_e[dup_brk_e['cat'] == cat]
                e = stn_df.loc[list(group.iloc[0, [0,1]])]

                group_nodes = list(group.index.values) + list(e.index.values)
                group_pos = stn_df.loc[group_nodes]
                group_pos['dist'] = \
                    np.sqrt((stn_df['x'] - e.loc[e.index[0], 'x']) ** 2 +
                            (stn_df['y'] - e.loc[e.index[0], 'y']) ** 2)
                group_pos = group_pos.sort_values('dist')

                new_graph.remove_edge(e.index[0], e.index[1])

                for i, n in enumerate(group_pos.index.values[:-1]):
                    n2 = group_pos.index.values[i+1]
                    dist = \
                        np.sqrt((group_pos.loc[n, 'x'] -
                                 group_pos.loc[n2, 'x']) ** 2 +
                                (group_pos.loc[n, 'y'] -
                                 group_pos.loc[n2, 'y']) ** 2)
                    new_graph.add_edge(n, n2, length=dist)

        return new_graph, n_not_added

def network_to_gdf(graph):
    """
    Convert a networkx graph into two GeoDataFrames: one for nodes (Points) and
    the other for edges (LineStrings)
    :param graph:
    :return:
    """


    node_gdf = \
        gpd.GeoDataFrame.from_dict(nx.get_node_attributes(graph, 'pos')).T

    node_xy = [Point(stn[0], stn[1])
               for index, stn in list(node_gdf.iterrows())]
    node_gdf['geometry'] = node_xy
    node_gdf.set_geometry(col='geometry', inplace=True)
    node_gdf.set_crs(epsg=27700)

    lines = []
    g_edges = graph.edges()
    for edge in g_edges:
        i_x, i_y = node_gdf.loc[edge[0], 'geometry'].x, \
                   node_gdf.loc[edge[0], 'geometry'].y
        j_x, j_y = node_gdf.loc[edge[1], 'geometry'].x, \
                   node_gdf.loc[edge[1], 'geometry'].y
        lines.append(f"LINESTRING({i_x} {i_y}, {j_x} {j_y})")

    edge_dict = {'edge': g_edges, 'geometry': lines}
    edge_gdf = gpd.GeoDataFrame.from_dict(edge_dict)
    edge_gdf['geometry'] = gpd.GeoSeries.from_wkt(edge_gdf['geometry'])
    edge_gdf.set_geometry(col='geometry', inplace=True)
    edge_gdf.set_crs(epsg=27700, inplace=True)

    return node_gdf, edge_gdf

def network_to_array(graph, grid):
    """
    Converts the edges of a network graph to an array reflective of the grid,
    including the length of all lines in each grid space.
    :param graph:
    :param grid:
    :return:
    """

    node_gdf, edge_gdf = network_to_gdf(graph=graph)
    # buff_edge = edge_gdf.buffer(1000)
    # buff_edge = buff_edge.to_frame(name='geometry')
    # buff_edge['edge'] = edge_gdf['edge']

    merged_gdf = gpd.sjoin(grid, edge_gdf, how='left', op='intersects')
    clipped_gdf = gpd.clip(grid, edge_gdf)

    # clipped_buff = gpd.clip(grid, buff_edge)
    # length = clipped_buff.length

    length = clipped_gdf.geometry.length

    merged_gdf['edge_length'] = length
    merged_gdf['edge_length'] = \
        np.where(np.isnan(merged_gdf['edge_length']), 0,
                 merged_gdf['edge_length'])
    merged_gdf['i_node'] = [i[0] if i is not np.nan else np.nan
                            for i in merged_gdf['edge'].values]
    merged_gdf['j_node'] = [i[1] if i is not np.nan else np.nan
                            for i in merged_gdf['edge'].values]

    length_raster = make_geocube(vector_data=merged_gdf, resolution=(-1e3, 1e3),
                                 fill=0, measurements=["edge_length"])
    length_array = length_raster.edge_length.values

    inode_raster = make_geocube(vector_data=merged_gdf, measurements=["i_node"],
                                resolution=(-1e3, 1e3), fill=np.nan)
    inode_array = inode_raster.i_node.values

    jnode_raster = make_geocube(vector_data=merged_gdf, measurements=["j_node"],
                                resolution=(-1e3, 1e3), fill=np.nan)
    jnode_array = jnode_raster.j_node.values

    return length_array, inode_array, jnode_array

def graph_to_array(graph, stn_mat, xylim):
    """
    Converts nodes positions to array detailing count of nodes in each grid
    cell.

    :param graph: networkx.classes.graph.Graph
    :param stn_mat: numpy.ndarray
    :param xylim: numpy.ndarray
        x and y limits of stn_mat, in order xmin, ymin, xmax, ymax
    :return:
    """
    stn_df = pd.DataFrame.from_dict(
        nx.get_node_attributes(G=graph, name='pos'), orient='index') \
        .rename(columns={0: 'x', 1: 'y'}).astype(int)

    for i, stn in stn_df.iterrows():
        if stn['x'] >= xylim[0] and stn['x'] < xylim[2] and \
           stn['y'] >= xylim[1] and stn['y'] < xylim[3]:
            stn_mat[stn['y'] - xylim[1], stn['x'] -xylim[0]] += 1

    return stn_mat

def obs_graph_to_array(graph, stn_mat, xylim):
    """
    Converts nodes positions to array detailing count of nodes in each grid
    cell.

    :param graph: networkx.classes.graph.Graph
    :param stn_mat: numpy.ndarray
    :param xylim: numpy.ndarray
        x and y limits of stn_mat, in order xmin, ymin, xmax, ymax
    :return:
    """
    stn_df = pd.concat([pd.DataFrame.from_dict(
        nx.get_node_attributes(G=graph, name='lon'), orient='index') \
        .rename(columns={'lon': 'x'}).astype(int),
                        pd.DataFrame.from_dict(
        nx.get_node_attributes(G=graph, name='lat'), orient='index') \
        .rename(columns={'lat': 'y'}).astype(int)])

    stn_df['lat'] = stn_df['lat'] - xylim[0]

    for i, stn in stn_df.iterrows():
        if stn['x']-xylim[0] >= xylim[0] and stn['x']-xylim[0] < xylim[2] and \
           stn['y']-xylim[1]  >= xylim[1] and stn['y']-xylim[1] < xylim[3]:
            stn_mat[stn['y'] -xylim[0], stn['x']] += 1

    return stn_mat
# %%
if __name__ == '__main__':
    os.chdir('.')

    run_name = 'tryingimprovement20'

    dir_path = f'./Results/NetworkGrowth/{run_name}'
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    notes = ''
    meta_data = {'Timestamp': str(datetime.datetime.now()), 'Notes': notes}
    plot_figs = True
    rho_start_yr = 1831
    ntwrk_start_yr = 1836

    # None, "record" or filepath
    file_paths = {'rho_yrs_fp': "./Data/London/rho_yrs.p",
                  'network_years_fp': './Data/London/networks_5years_nodup_factcnct.p',
                  'empty_grid_fp': './Data/London/empty_regular_grid.shp',
                  'init_network_fp': "./Data/London/1841_init_network.p"}

    rho_obs = pickle.load(open(file_paths['rho_yrs_fp'], "rb"))
    ntwrk_yrs = pickle.load(open(file_paths['network_years_fp'], 'rb'))
    empty_grid = gpd.read_file(file_paths['empty_grid_fp']).set_index('num_codes')
    xmin, ymin, xmax, ymax = empty_grid.total_bounds / 1000
    # Set the domain spatial, temporal params and initial conditions
    grid_params = {'dx': 1, 'dy': 1, 'dt': 1, 'tot_ts': 180,
                   'xlim': [xmin, xmax], 'ylim': [ymin, ymax],
                   'N0': rho_obs[1831], 'nrows': rho_obs[1831].shape[0],
                   'ncols': rho_obs[1831].shape[1]}
    model_params ={}
    # Only used for plotting
    ntwrk_pos = {n: [ntwrk_yrs[2011].nodes[n]['lon'] - xmin,
                     ntwrk_yrs[2011].nodes[n]['lat'] - ymin]
                 for n in ntwrk_yrs[2011].nodes}

    # Load the rail network
    # xy of stations
    print('Loading network')
    init_network = pickle.load(open(file_paths['init_network_fp'], 'rb'))
    nx.set_node_attributes(init_network, {n: init_network.nodes[n]['pos'][0]
                                          for n in init_network.nodes}, 'lon')
    nx.set_node_attributes(init_network, {n: init_network.nodes[n]['pos'][1]
                                          for n in init_network.nodes}, 'lat')
    xy = pd.read_csv('./Data/London/ldn_stn_xy_w27700.csv', header=0,
                     index_col=0)[['lon_m', 'lat_m', 'Year']]\
        .rename(columns={'lon_m': 'lon', 'lat_m': 'lat'})

    xy['lon'] = xy['lon'] / 1000.  # m to km
    xy['lat'] = xy['lat'] / 1000.

    # %%
    network_years = 10
    ts_per_year = 10  # TODO: eventually sync this with RhoGrid.dt

    # %%
    plot_years = [1831, 1851, 1891, 1931, 1961, 1991, 2011]
    # Initially running every ten years so no need to interpolate rho for now
    # run_years = np.arange(1841, 2012, 10)
    run_years = np.sort(list(rho_obs.keys()))

    model_params = {'phi': -2.5, 'g': 0.0008, 'edge_weight': 3000,
                    'psi': 1, 'dt' : 10, 'max_stns':6, 'cost_coeff': 4}

    if model_params['phi'] == abs(model_params['phi']):
        model_params['phi'] *= -1

    # Matrix of d**-phi that may move around to cover each cell
    max_dim = np.max(rho_obs[run_years[0]].shape)
    y_d = np.arange(max_dim-1, -max_dim, -1)[np.newaxis] * \
          np.full(shape=[2*max_dim-1, 2*max_dim-1],
                  fill_value=1)
    x_d = y_d.T
    d_origin = np.sqrt(x_d**2 + y_d**2) ** model_params['phi']
    d_origin[max_dim-1, max_dim-1] = 0

    stn_origin = np.sqrt(x_d**2 + y_d**2) ** model_params['psi']
    stn_origin[max_dim - 1, max_dim - 1] = 0
    # Initiating G as g/max(qj)
    G = init_G(nrows=grid_params['nrows'], ncols=grid_params['ncols'],
               d_origin=d_origin, city_mat=rho_obs[run_years[0]],
               g=model_params['g'])

    # trimming distance gravity matrix
    shape_diff = np.abs(rho_obs[run_years[0]].shape[1] -
                        rho_obs[run_years[0]].shape[0])
    if rho_obs[run_years[0]].shape[1] > rho_obs[run_years[0]].shape[0]:
        d_origin = d_origin[shape_diff:-shape_diff,:]
        stn_origin = stn_origin[shape_diff:-shape_diff, :]
    else:
        d_origin = d_origin[:, shape_diff:-shape_diff]
        stn_origin = stn_origin[:, shape_diff:-shape_diff]

    stn_mat = np.zeros_like(rho_obs[run_years[0]]) # hold #stations in cell
    stn_graph = nx.empty_graph(n=0, create_using=None) # nx graph of sim network
    total_stns = np.zeros_like(run_years, dtype=float) # time series stn total
    total_pop = np.zeros_like(run_years, dtype=float) # time series pop total
    G_ts = np.zeros_like(run_years, dtype=float) # time series of G
    dPdt_all = np.zeros_like(run_years, dtype=float) # time series dP/dt

    mean_qj = np.zeros_like(run_years, dtype=float) # time series mean qj

    ntwrk_dict = {} # network nx objects through time
    for i, yr in enumerate(run_years):
        print(yr)
        if i == 0:
            pass
        elif yr == 1841:
            print('Loading initial network')
            stn_graph = nx.Graph(init_network)
            stn_mat = graph_to_array(stn_graph, stn_mat,
                                     [0, 0, xmax-xmin, ymax-ymin])
        else:
            # Array of rho values
            city_mat = copy(rho_obs[yr])
            city_mat = np.where(np.isnan(city_mat), 0, city_mat)

            # arrays of edge lengths present and i, j nodes
            if not nx.is_empty(stn_graph):
                scaled_graph = nx.Graph(stn_graph)
                nx.set_node_attributes(scaled_graph,  # scaling km to m
                    {n:
                    (stn_graph.nodes[n]['pos'] + np.array([xmin, ymin])) * 1000
                     for n in nx.get_node_attributes(
                        stn_graph, 'pos').keys()},
                    name='pos')
                # create array of length of edges in each grid cell
                edge_lengths, i_nodes, j_nodes = \
                    network_to_array(graph=scaled_graph, grid=empty_grid)
                # TODO: find out why arrays are mirrored on x-axis. Fix for now
                edge_lengths = np.flip(edge_lengths, axis=0)
                i_nodes = np.flip(i_nodes, axis=0)
                j_nodes = np.flip(j_nodes, axis=0)
                edge_lengths /= 1000 # convert to km
            else:
                edge_lengths = None
                i_nodes = None
                j_nodes = None
            # Generate random number (0, 1] grid for whole extent
            z_mat = np.random.random(size=(grid_params['nrows'],
                                           grid_params['ncols']))
            # variable for max(qj), used for calclulating G
            # matrix for q_j (zeros)
            dPdt = ((np.nansum(rho_obs[yr]) -
                     np.nansum(rho_obs[run_years[i - 1]]))) / model_params['dt']
            dPdt_all[i] = dPdt

            rho_k_d_jk = new_station_probability(
                nrows=grid_params['nrows'], ncols=grid_params['ncols'],
                d_origin=d_origin, city_mat=city_mat)

            if edge_lengths is not None:
                rho_k_d_jk = rho_k_d_jk + \
                             model_params['edge_weight'] * \
                             np.where(edge_lengths > 0, 1, 0)

            rho_d_max = np.max(rho_k_d_jk)
            # Normalising qj
            qj = copy(rho_k_d_jk)
            qj /= qj.sum()

            mean_qj[i] = qj.mean()
            # Actual probability Qj
            Qj = model_params['g'] * dPdt * qj
            new_stns = (Qj > z_mat).astype(int)

            # Restricting total number of stations in each grid cell
            new_stns = np.where(
                (stn_mat + new_stns) <=  model_params['max_stns'],
                 new_stns, 0)
            # add boolean matrix to city_mat
            stn_mat += new_stns
            stn_graph, new_nodes = \
                add_new_station_nodes(graph=stn_graph, new_stn_mat=new_stns,
                                      year=yr) # return_new_nodes=True,
            add_new_edges(graph=stn_graph, stn_mat=stn_mat,
                          stn_origin=stn_origin, stn_gamma=model_params['psi'],
                          new_nodes=new_nodes, add_edges=True,
                          i_node=i_nodes, j_node=j_nodes,
                          cost_coeff=model_params['cost_coeff'])
            for e in list(stn_graph.edges):
                if e[0] == e[1]:
                    print('self loops')

            if plot_figs:
                node_pos = nx.get_node_attributes(stn_graph, 'pos')
                fig, ax = plt.subplots()
                qj_plot = ax.pcolor(qj)
                cbar = ax.figure.colorbar(qj_plot)
                cbar.ax.set_ylabel('$Q_{j}$')
                ax.set_title(
                    f"{yr}: $\phi$ = {model_params['phi']}, "
                    f"g = {model_params['g']}, N = {stn_mat.sum()}")
                nx.draw(G=stn_graph, node_size=5, node_color='r', pos=node_pos)
                nx.draw(G=stn_graph, node_size=5, node_color='cyan',
                        nodelist=new_nodes, pos=node_pos)
                ax.set_aspect('equal')
                limits = plt.axis('on')  # turns on axis
                ax.tick_params(left=True, bottom=True, labelleft=True,
                               labelbottom=True)
                plt.tight_layout()
                plt.savefig(f"./{dir_path}/"
                            f"rybskili_network_{yr}_{abs(model_params['phi'])}"
                            f"_{int(model_params['g'] * 100)}.png")
                # plt.show()

                ntwrk = ntwrk_yrs[yr]

                fig, (ax1, ax2) = plt.subplots(ncols=2)
                city = ax1.pcolor(city_mat / 1000.)
                cbar = ax1.figure.colorbar(city)
                cbar.ax.set_ylabel('Population (1000s)')
                ax1.set_title(
                    f"{yr}: $\phi$ = {model_params['phi']}, "
                    f"g = {model_params['g']}, N = {stn_mat.sum()}")
                nx.draw(G=stn_graph, node_size=3, node_color='r',
                        pos=node_pos, ax=ax1)

                ax1.set_xlim(0)
                ax1.set_aspect('equal')
                city2 = ax2.pcolor(city_mat / 1000.)
                ax2.set_title(f"N = {len(ntwrk.nodes)}")
                nx.draw(G=ntwrk, node_size=3, node_color='r',
                        pos=ntwrk_pos, ax=ax2)
                ax2.set_xlim(0)
                ax2.set_aspect('equal')
                plt.tight_layout()
                plt.savefig(f"./{dir_path}/rybskili_network_obs_{yr}_"
                            f"{abs(model_params['phi'])}"
                            f"_{int(model_params['g'] * 100)}.png")
                # plt.show()
                nx.set_node_attributes(ntwrk, {n: (ntwrk.nodes[n]['lon'] - 490,
                                                   ntwrk.nodes[n]['lat'] - 148)
                                       for n in ntwrk.nodes}, 'pos')
                obs_stn_mat = graph_to_array(
                    graph=ntwrk, stn_mat=np.zeros_like(rho_obs[run_years[0]]),
                    xylim=[0, 0, xmax-xmin, ymax-ymin])
                obs_x_profile = obs_stn_mat.sum(axis=0)
                obs_y_profile = obs_stn_mat.sum(axis=1)

                sim_x_profile = stn_mat.sum(axis=0)
                sim_y_profile = stn_mat.sum(axis=1)

                fig, (ax1, ax2) = plt.subplots(nrows=2)
                ax1.plot(obs_x_profile, color='b')
                ax1.plot(sim_x_profile, color='r')
                ax1.set_title(f'{yr} x profile')

                ax2.plot(obs_y_profile, color='b')
                ax2.plot(sim_y_profile, color='r')
                ax2.set_title('y profile')

                plt.tight_layout()
                plt.savefig(f"./{dir_path}/xyprofiles_{yr}.png")

            total_pop[i] = city_mat.sum()
            total_stns[i] = stn_mat.sum()
            # Calculate G
            print(model_params['g'], model_params['g'] * dPdt * rho_d_max /
                  np.mean(rho_k_d_jk))

            G_ts[i] = G

        ntwrk_dict[yr] = nx.Graph(stn_graph)

    pickle.dump(ntwrk_dict, open(f'./{dir_path}/simulated_networks.p', 'wb'))

    f90nml.write({'metadata': meta_data,
                  'grid_params': {key: val for key, val in grid_params.items()
                                  if key not in ['N0', 'topography']},
                  'model_params': model_params, 'file_paths': file_paths},
                 open(f'{dir_path}/{run_name}_settings.nml', 'w'))

    if plot_figs:
        fig, ax = plt.subplots()
        ax.plot(dPdt_all)
        ax.set_title('dPdt')
        plt.show()

        fig, ax = plt.subplots()
        stns = ax.pcolor(stn_mat)
        cbar = ax.figure.colorbar(stns)
        cbar.ax.set_ylabel('No. of stations')
        ax.set_title(f"$\phi$ = {model_params['phi']}, g = "
                     f"{model_params['g']}, "
                     f"N = {stn_mat.sum()}")

        ax.set_aspect('equal')
        plt.tight_layout()
        plt.savefig(f"./{dir_path}/rybskili_{abs(model_params['phi'])}"
                    f"_{int(model_params['g'] * 100)}.png")
        plt.show()

        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        ax.plot(total_pop, color='red')
        ax2.plot(total_stns, color='blue')
        ax.set_ylabel('Population within extent', color='red')
        ax2.set_ylabel('Modelled no. of stations in extent', color='blue')
        ax.set_xlabel('Year')
        ax.set_xticks(np.arange(len(run_years), step=2))
        ax.set_xticklabels(run_years[::2])
        plt.savefig(f'./{dir_path}/rybskili_stations.png')
        plt.show()

        fig, ax = plt.subplots()
        ax.plot(mean_qj, color='red')
        ax.set_ylabel('Mean $q_{j}$')
        ax.set_xlabel('Year')
        ax.set_xticks(np.arange(len(run_years[1:]), step=2))
        ax.set_xticklabels(run_years[1::2])
        plt.tight_layout()
        plt.savefig(f'./{dir_path}/rybskili_stations_meanqj.png')
        plt.show()