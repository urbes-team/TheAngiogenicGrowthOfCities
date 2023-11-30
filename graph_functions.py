"""
Script holding the classes that direct population and network response, and some
pre-processing functions.
"""
import networkx as nx
import numpy as np
from numba import jit
import pandas as pd
import matplotlib.pyplot as plt
from copy import copy
from analysis_functions import mean_smooth


@jit(nopython=True)
def rho_diffusion(rho, f, dx, dy, dt, D, topo=None):
    rho_new = np.zeros_like(rho)
    # limits of i and j stop the domain acting like a manifold wrt diffusion,
    # i.e. people fall off the boundaries instead of shifting to the opposite.
    # Diffusion of pop density only works for a regular grid as 1. it requires
    # neighbouring grid spaces and 2. it models the movement of population
    # density not population
    if topo is not None:
        for i in np.arange(1, rho.shape[0] - 1):
            for j in np.arange(1, rho.shape[1] - 1):
                # TODO: should it be the case that it isn't divided by dx * dy
                #  as this is then double divided by area?
                rho_new[i, j] = D * dt * \
                                (topo[i, j] * (rho[i - 1, j] + rho[i + 1, j] +
                                               rho[i, j - 1] + rho[i, j + 1])
                                 - (topo[i - 1, j] + topo[i + 1, j] + topo[
                                            i, j - 1] +
                                    topo[i, j + 1]) * rho[i, j]) / \
                                (dx * dy)
    else:
        for i in np.arange(1, rho.shape[0] - 1):
            for j in np.arange(1, rho.shape[1] - 1):
                # TODO: should it be the case that it isn't divided by dx * dy
                #  as this is then double divided by area?
                rho_new[i, j] = D * dt * (rho[i - 1, j] - 2 * rho[i, j] + \
                                          rho[i + 1, j] + rho[i, j - 1] - 2 *
                                          rho[i, j] + rho[i, j + 1]) / \
                                (dx * dy)

    return rho_new


@jit(nopython=True)
def round_lonlat(arr):
    return (arr * 2).round() / 2


def create_distance_probability_array(shape, phi):
    max_dim = np.max(shape)
    y_d = np.arange(max_dim - 1, -max_dim, -1)[np.newaxis] * \
          np.full(shape=[2 * max_dim - 1, 2 * max_dim - 1],
                  fill_value=1)
    x_d = y_d.T
    d_origin = np.sqrt(x_d ** 2 + y_d ** 2) ** phi
    d_origin[max_dim - 1, max_dim - 1] = 0

    shape_diff = np.abs(shape[1] - shape[0])
    if shape[1] > shape[0]:
        d_origin = d_origin[shape_diff:-shape_diff, :]
    else:
        d_origin = d_origin[:, shape_diff:-shape_diff]

    return d_origin

def create_edge_probability_array(shape, psi):
    max_dim = np.max(shape)
    y_d = np.arange(max_dim - 1, -max_dim, -1)[np.newaxis] * \
          np.full(shape=[2 * max_dim - 1, 2 * max_dim - 1],
                  fill_value=1)
    x_d = y_d.T

    stn_origin = np.sqrt(x_d ** 2 + y_d ** 2) ** psi
    stn_origin[max_dim - 1, max_dim - 1] = 0

    shape_diff = np.abs(shape[1] - shape[0])
    if shape[1] > shape[0]:
        stn_origin = stn_origin[shape_diff:-shape_diff, :]
    else:
        stn_origin = stn_origin[:, shape_diff:-shape_diff]

    return stn_origin

def check_endpoints(graph, subgraph):
    """
    Checks if new endpoints exist in subgraph where they do not exist in
    complete graph.

    :param graph: networkx.classes.graph.Graph
        Complete graph from which the subgraph is derived
    :param subgraph: networkx.classes.graph.Graph
        Subgraph of complete graph
    :return:
    """
    grph_deg = dict(graph.degree)
    dscnt_ends = []
    for node, deg in dict(subgraph.degree).items():
        if deg == 1:
            if grph_deg[node] > 1:
                dscnt_ends.append(node)

    return dscnt_ends


def add_edges(graph, subgraph, distance):
    """
    Adds edges between disconnected subgraphs, assuming that an edge will exist
    between a given number of consecutive nodes.

    :param graph: networkx.classes.graph.Graph
        Complete graph from which the subgraph is derived
    :param subgraph: networkx.classes.graph.Graph
        Subgraph of complete graph
    :param distance: int
        Maximum number of connected nodes in complete graph from which we can
        assume new endpoints in subgraph have a connection
    :return:
    """

    if nx.is_frozen(subgraph):
        raise nx.NetworkXError("Frozen graph can't be modified")
    if len(list(nx.connected_components(graph))) == \
            len(list(nx.connected_components(subgraph))):
        # if subgraph is as disconnected as graph, then no new endpoints
        return subgraph
    else:
        # find disconnected components of subgraph
        cnt_comp = list(nx.connected_components(subgraph))
        # Find new endpoints of disconnected subgraph components
        dscnt_ends = check_endpoints(graph=graph, subgraph=subgraph)

        if len(dscnt_ends) == 0:
            raise ValueError("Subgraph is disconnected but no new endpoints "
                             "found")

        # Collect potential connections between disconnected components
        for node in dscnt_ends:
            # Find potential new nodes from whole graph
            pot_nodes = []
            short_path = nx.single_source_shortest_path(graph, source=node,
                                                        cutoff=distance)
            for v in short_path:
                for n in v:
                    if (n == node) or (n not in subgraph.nodes) or \
                            subgraph.has_edge(node, n):
                        pass
                    elif n in pot_nodes:
                        # Stops any nodes past already viable node being added
                        # works as v in order of distance from node
                        break
                    else:
                        pot_nodes.append(n)

            for pn in pot_nodes:
                print('Adding pot nodes')
                subgraph = create_connecting_edge(graph=graph,
                                                  subgraph=subgraph, src=node,
                                                  trgt=pn)

        return subgraph


def create_connecting_edge(graph, subgraph, src, trgt):
    """
    Adds an edge a subgraph, with summed weight of the original path from the
    complete graph.

    :param graph: networkx.classes.graph.Graph
        Complete graph from which the subgraph is derived
    :param subgraph: networkx.classes.graph.Graph
        Subgraph of complete graph
    :param src: str, int, float
        Node identifier
    :param trgt: str, int, float
        Node identifier
    :return:
    """
    edge_weight = nx.shortest_path_length(graph, source=src, target=trgt,
                                          weight='weight')

    subgraph.add_edge(src, trgt)
    subgraph[src][trgt]['weight'] = edge_weight

    return subgraph

def remaining_nodes(arr):
    """
    Creates array of possible connections between nodes.

    :param arr: numpy.ndarray
         Array of node IDs which are possibly all connected to each other
         through every combination
    """
    out = []
    for i, x in enumerate(arr[:-1]):
        for j, y in enumerate(arr[i+1:]):
            out.append([arr[i], arr[i+j+1]])
    return(np.array(out))

def rebuild_later_nodes(g_edges, later_nodes):
    """

    :param g_edges: numpy.ndarray
        Edges of the time-limited graph, as N node pairs, shape: (N, 2)
    :param later_nodes: numpy.ndarray
        List of nodes that appear later then the intended network
    :return:

    """
    new_links = np.empty((0, 2), int)

    # Edges with one later node. Avoids edges where neither node exists
    lxor_edges = \
        g_edges[
            np.where(
                np.logical_xor(np.isin(g_edges[:, 0], later_nodes),
                               np.isin(g_edges[:, 1], later_nodes)))[0]]

    for ln in later_nodes:
        if ln in lxor_edges.flatten():  # checking node is xor
            ln_edges = \
                lxor_edges[np.logical_or(lxor_edges[:, 0] == ln,
                                         lxor_edges[:, 1] == ln)].flatten()
            ln_edges = ln_edges[np.where(ln_edges != ln)]
            if len(
                    ln_edges) == 1:
                pass  # either terminal nodes or part of missing chains
            else:
                links = remaining_nodes(ln_edges)
                new_links = np.append(new_links, links, axis=0)
        else:
            pass

    lboth_edges = \
        g_edges[
            np.where(
                np.logical_and(np.isin(g_edges[:, 0], later_nodes),
                               np.isin(g_edges[:, 1], later_nodes)))[0]]

    g_chain = nx.from_edgelist(
        lboth_edges)  # subgraph of edges with two late nodes
    # subgraphs of chains
    sg_chain = [g_chain.subgraph(c).copy() for c in
                nx.connected_components(g_chain)]

    end_chain = []
    for g in sg_chain:  #
        end_chain.append([node for (node, deg) in list(g.degree()) if deg == 1])

    for ec in end_chain:
        cnnct_nodes = \
            lxor_edges[np.where(np.logical_xor(lxor_edges == ec[0],
                                               lxor_edges == ec[1]))[
                0]].flatten()
        cnnct_nodes = cnnct_nodes[np.where(
            np.logical_and(cnnct_nodes != ec[0], cnnct_nodes != ec[1]))]

        if len(cnnct_nodes) == 1:
            pass  # these are either terminal nodes or part of missing chains
        else:
            links = remaining_nodes(cnnct_nodes)
            new_links = np.append(new_links, links, axis=0)

    if len(new_links) > 0:
        return new_links
    else:
        return None


def set_edge_node_attributes(G, stns_df):
    """

    :param G: networkx.classes.graph.Graph
    :param stns_df: pandas.core.frame.DataFrame
    :param lines_df: pandas.core.frame.DataFrame
    :return:
    """

    coords, lon, lat, year = {}, {}, {}, {}

    for n in list(G.nodes):
        # converting m to km
        coords[n] = (stns_df.loc[n, 'lon'], stns_df.loc[n, 'lat'])
        lon[n] = stns_df.loc[n, 'lon']
        lat[n] = stns_df.loc[n, 'lat']
        year[n] = stns_df.loc[n, 'YEAR']

    nx.set_node_attributes(G, lon, name='lon')
    nx.set_node_attributes(G, lat, name='lat')
    nx.set_node_attributes(G, year, name='year')

    edge_len, edge_year = {}, {}
    for e in list(G.edges):
        edge_len[e] = np.hypot(G.nodes[e[0]]['lon'] - G.nodes[e[1]]['lon'],
                               G.nodes[e[0]]['lat'] - G.nodes[e[1]]['lat'])

    nx.set_edge_attributes(G, edge_len, name='length')

    return G, coords


def add_and_remove_edges_nodes(G, later_nodes, new_links):
    """
    Adds new edges to the network graph such that later nodes are excluded but
    total edge length is preserved. Removes edges from later, non-existent
    nodes, and corresponding nodes.

    :param G: networkx.classes.graph.Graph
        Current network graph
    :param later_nodes: numpy.ndarray
        Array of nodes that are appear later than the intended network year
    :param new_links: numpy.ndarray
        Array of new edges
    :return: networkx.classes.graph.Graph
        Graph with necessary edges added and removed
    """
    g_edges = np.array(G.edges)
    late_edges = \
        g_edges[
            np.where(
                np.logical_or(np.isin(g_edges[:, 0], later_nodes),
                              np.isin(g_edges[:, 1], later_nodes)))[0]]

    for e in new_links:
        G.add_edge(e[0], e[1],
                   length=nx.shortest_path_length(G, source=e[0], target=e[1],
                                                  weight='length'))

    G.remove_edges_from(late_edges)
    G.remove_nodes_from(later_nodes)

    return G

def graph_to_year(lines_df, stns_df, year, lineclose=True, stnclose=True):
    """
    Creates a network graph representative of the specified year from existing
    edges whilst removing non-existent nodes.

    :param lines_df: pandas.core.frame.DataFrame
        Dataframe detailing source ('ID_I') and target ('ID_J') node IDs as well
        as the years they were established (LINEYEAR)
    :param stns_df: pandas.core.frame.DataFrame
        Dataframe detailing station IDs (index), year opened ('YEAR'), longitude
        ('lon', km) and latitude ('lat', km)
    :param year: int
        Year of intended network
    :param lineclose: bool
        Whether or not to include line closure dates
    :param stnclose: bool
        Whether or not to include station closure dates
    :return: tuple
        Network graph representative of year and dictionary of node longitude
        and latitude
    """
    # Graph for the year from edges (this includes non-existent nodes)
    if lineclose:
        G_yr = nx.from_pandas_edgelist(
            lines_df[(lines_df['LINEYEAR'] <= year) &
                    (lines_df['line_close'] >= year)],
            source='ID_I', target='ID_J', edge_attr='LINEYEAR')
    else:
        G_yr = nx.from_pandas_edgelist(
            lines_df[(lines_df['LINEYEAR'] <= year)],
            source='ID_I', target='ID_J', edge_attr='LINEYEAR')
    # Set attributes for the nodes and edges
    G_yr, coords = set_edge_node_attributes(G=G_yr, stns_df=stns_df,
                                            stnclose=stnclose)
    # Find nodes that do not exist on the network
    if stnclose:
        later_nodes = np.setdiff1d(
            np.array(G_yr.nodes),
            stns_df[(stns_df['YEAR'] <= year) &
                    (stns_df['stn_close'] >= year)].index.values)
    else:
        later_nodes = np.setdiff1d(
            np.array(G_yr.nodes),
            stns_df[(stns_df['YEAR'] <= year)].index.values)
    # Create possible new links
    new_links = rebuild_later_nodes(g_edges=np.array(G_yr.edges),
                                    later_nodes=later_nodes)
    # Add new links and remove false ones
    if new_links is not None:
        G_yr = add_and_remove_edges_nodes(G_yr, later_nodes, new_links)

    return G_yr, coords

def set_centrality(G, G_alt_connect=None):
    """
    Calculates weighted node betweenness centrality and closeness centrality and
    sets the network characteristics as node attributes

    :param G: networkx.classes.graph.Graph
        Current network graph
    :param G_alt_connect: networkx.classes.graph.Graph
        Fully connected version of G
    :return:
    """
    if G_alt_connect is not None:
        wbc = nx.betweenness_centrality(G, normalized=True, endpoints=True,
                                        weight='length')
        cc = nx.closeness_centrality(G, wf_improved=True, distance='length')
    else:
        wbc = nx.betweenness_centrality(G, normalized=True, endpoints=True,
                                        weight='length')
        cc = nx.closeness_centrality(G, wf_improved=True, distance='length')

    nx.set_node_attributes(G, wbc, name='WBC')
    nx.set_node_attributes(G, cc, name='CC')

    return G

def set_accessibility(G, G_alt_connect=None):
    """
    Calculates accessibility and total transportation distance of each node,
    setting these characteristics as node attributes (Wang et al. 2009).

    :param G: networkx.classes.graph.Graph
        Current network graph
    :param G_alt_connect: networkx.classes.graph.Graph
        Fully connected version of G
    """
    # Chooses the graph from which to calculate (not set) accessibility
    if G_alt_connect is None:
        G_op, removed_nodes = remove_unconnected_nodes(G)
    else:
        G_op = G_alt_connect
        removed_nodes = None

    d_i = {}
    for n in list(G_op.nodes):
        # shortest weighted path lengths from node n
        d_i[n] = \
            np.sum(list(dict(
                nx.single_source_dijkstra_path_length(
                    G_op, source=n, weight='length')).values()))

    total_d = np.sum(list(d_i.values()))

    v = nx.number_of_nodes(G_op)
    a_i = {}
    a_i_inv = {}
    for n in list(G_op.nodes):
        a_i[n] = d_i[n] / (total_d / v)
        a_i_inv[n] = a_i[n] ** -1

    if removed_nodes is not None:
        for n in removed_nodes:
            a_i[n] = 0
            d_i[n] = 0
            a_i_inv[n] = 0
    # Sets accessibility factors to original network
    nx.set_node_attributes(G, a_i, name='a_i')
    nx.set_node_attributes(G, d_i, name='d_i')
    nx.set_node_attributes(G, a_i_inv, name='a_i_inv')

    return G, total_d

# def set_accessibility(G):

def get_network_connectivities(G):
    """
    Calculates overall network connectivities: beta index, cyclomatic index,
    alpha index, gamma index (Black 2003; Wang et al. 2009).

    :param G: networkx.classes.graph.Graph
        Current network graph
    """
    G_copy, removed_nodes = remove_unconnected_nodes(G)

    e = nx.number_of_edges(G_copy)
    v = nx.number_of_nodes(G_copy)
    p = nx.number_connected_components(G_copy)

    # BETA INDEX
    beta_index = e / v
    # CYCLOMATIC NUMBER
    cyclo_num = e - v + p
    # ALPHA INDEX
    alpha_index = cyclo_num / (2 * v - 5 * p)
    # GAMMA INDEX
    gamma_index = e / (3 * (v - 2))

    return beta_index, cyclo_num, alpha_index, gamma_index

def remove_unconnected_nodes(G):
    """
    Removes single unconnected nodes from a graph. Single unconnected nodes are
    a result of unavailable network time data.

    :param G: networkx.classes.graph.Graph
        Current network graph
    :return:
    """
    # Remove single unconnected nodes.
    G_copy = copy(G)
    to_be_removed = [x for x in G_copy.nodes() if G_copy.degree(x) < 1]

    for x in to_be_removed:
        G_copy.remove_node(x)

    return G_copy, to_be_removed

def remove_outside_nodes(G, xmin, xmax, ymin, ymax):
    """
    Removes nodes which are outside of the domain
    :param G:
    :return:
    """
    lonlat = pd.DataFrame.from_dict(dict(G.degree()),
                                    orient='index', columns=['degree'])
    for k in ['lat', 'lon']:
        lonlat[k] = pd.Series(nx.get_node_attributes(G, k))

    outside_nodes = lonlat[(lonlat['lon'] < xmin) | (lonlat['lon'] > xmax) |
                           (lonlat['lat'] < ymin) |
                           (lonlat['lat'] > ymax)].index.values

    if len(outside_nodes) > 0:
        G.remove_nodes_from(outside_nodes)

    return G


def connect_subnetworks(G):
    """
    Naively connect all subnetworks in G so that G is entirely connected

    :param G: networkx.classes.graph.Graph
        Network graph comprised of multiple subnetworks
    :return:
    """
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    new_edges = []  # FOR TESTING ONLY
    # iterate
    while len([c for c in nx.connected_components(G)]) > 1:
        # Find "centre" of each component using min/max lon and lat
        centres = []
        for c in Gcc:
            Gc = G.subgraph(c)
            cent_x = \
                np.min(list(nx.get_node_attributes(Gc, 'lon').values())) + \
                0.5 * (np.max(
                    list(nx.get_node_attributes(Gc, 'lon').values())) - \
                       np.min(list(nx.get_node_attributes(Gc, 'lon').values())))
            cent_y = \
                np.min(list(nx.get_node_attributes(Gc, 'lat').values())) + \
                0.5 * (np.max(
                    list(nx.get_node_attributes(Gc, 'lat').values())) - \
                       np.min(list(nx.get_node_attributes(Gc, 'lat').values())))
            centres.append([cent_x, cent_y])

        centres = np.array(centres)
        # Choose other network (B) centre closest to centre of chosen network (A)
        clst_i = \
            np.argmin(np.sqrt(((centres[1:] - centres[0])**2).sum(axis=1))) + 1
        B = G.subgraph(Gcc[clst_i])
        # Choose B node (b) closest to centre of A
        Blonlat = \
            np.vstack((np.array(list(B.nodes)),
                       np.array(list(nx.get_node_attributes(
                           B, 'lon').values())),
                       np.array(list(nx.get_node_attributes(
                           B, 'lat').values())))) \
                .transpose()

        b = \
            np.argmin(np.sqrt(((Blonlat[:, 1:] - centres[0]) ** 2).sum(axis=1)))

        # Find A node (a) closest to b
        A = G.subgraph(Gcc[0])
        Alonlat = \
            np.vstack((np.array(list(A.nodes)),
                       np.array(list(nx.get_node_attributes(
                           A, 'lon').values())),
                       np.array(list(nx.get_node_attributes(
                           A, 'lat').values())))) \
                .transpose()

        a = \
            np.argmin(np.sqrt(((Alonlat[:, 1:] - Blonlat[b, 1:]) ** 2) \
                              .sum(axis=1)))

        # Create edge with length 33/5x Euclidean (walking speed)
        G.add_edge(Alonlat[a, 0], Blonlat[b, 0])
        edge_length = \
            np.sqrt((Alonlat[a, 1] - Blonlat[b, 1]) ** 2 +
                    (Alonlat[a, 2] - Blonlat[b, 2]) ** 2)
        new_edges.append((int(Alonlat[a, 0]), int(Blonlat[b, 0])))
        # new edge is assumed to involve walking between the two nodes so
        # length is distance * (ave tube speed / ave walking speed)
        G[Alonlat[a, 0]][Blonlat[b, 0]]['length'] = edge_length * (33 / 5)

        Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    return G

def connect_new_subnetworks(G):
    """
    Naively connect all subnetworks in G so that G is entirely connected. Same
    as connect_subnetworks() but coordinates have a different structure.

    :param G: networkx.classes.graph.Graph
        Network graph comprised of multiple subnetworks
    :return:
    """
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    new_edges = []  # FOR TESTING ONLY
    # iterate
    while len([c for c in nx.connected_components(G)]) > 1:
        # Find "centre" of each component using min/max lon and lat
        centres = []
        for c in Gcc:
            Gc = G.subgraph(c)
            pos = np.array(list(nx.get_node_attributes(Gc, 'pos').values()))
            cent_x = \
                np.min(pos[:, 0]) + 0.5 * (np.max(pos[:, 0]) -
                                           np.min(pos[:, 0]))
            cent_y = \
                np.min(pos[:, 1]) + 0.5 * (np.max(pos[:, 1]) -
                                           np.min(pos[:, 1]))
            centres.append([cent_x, cent_y])

        centres = np.array(centres)
        # Choose other network (B) centre closest to centre of chosen network (A)
        clst_i = \
            np.argmin(
                np.sqrt(((centres[1:] - centres[0]) ** 2).sum(axis=1))) + 1
        B = G.subgraph(Gcc[clst_i])
        # Choose B node (b) closest to centre of A
        Blonlat = np.vstack((np.array(list(B.nodes)),
                             np.array(list(nx.get_node_attributes(
                                 B, 'pos').values()))[:, 0],
                             np.array(list(nx.get_node_attributes(
                                 B, 'pos').values()))[:, 1])) \
            .transpose()

        b = \
            np.argmin(np.sqrt(((Blonlat[:, 1:] - centres[0]) ** 2).sum(axis=1)))

        # Find A node (a) closest to b
        A = G.subgraph(Gcc[0])
        Alonlat = np.vstack((np.array(list(A.nodes)),
                             np.array(list(nx.get_node_attributes(
                                 A, 'pos').values()))[:, 0],
                             np.array(list(nx.get_node_attributes(
                                 A, 'pos').values()))[:, 1])) \
            .transpose()

        a = \
            np.argmin(np.sqrt(((Alonlat[:, 1:] - Blonlat[b, 1:]) ** 2) \
                              .sum(axis=1)))

        # Create edge with length 33/5x Euclidean (walking speed)
        G.add_edge(Alonlat[a, 0], Blonlat[b, 0])
        edge_length = \
            np.sqrt((Alonlat[a, 1] - Blonlat[b, 1]) ** 2 +
                    (Alonlat[a, 2] - Blonlat[b, 2]) ** 2)
        new_edges.append((int(Alonlat[a, 0]), int(Blonlat[b, 0])))
        # new edge is assumed to involve walking between the two nodes so
        # length is distance * (ave tube speed / ave walking speed)
        G[Alonlat[a, 0]][Blonlat[b, 0]]['weight'] = edge_length * (33 / 5)

        Gcc = sorted(nx.connected_components(G), key=len, reverse=True)

        pass
    return G

class RhoGrid:

    def __init__(self, grid_params: dict, model_params:dict) -> None:
        # grid cell dimensions
        self.dx = grid_params['dx']
        self.dy = grid_params['dy']
        x = np.arange(grid_params['xlim'][0],
                      grid_params['xlim'][1], self.dx)  # km
        y = np.arange(grid_params['ylim'][0],
                      grid_params['ylim'][1], self.dy)  # km

        self.xv, self.yv = np.meshgrid(x, y)
        self.DX = grid_params['xlim'][1] - grid_params['xlim'][0]
        self.DY = grid_params['ylim'][1] - grid_params['ylim'][0]
        # Time
        self.dt = grid_params['dt']
        self.tot_ts = grid_params['tot_ts']
        self.ts = 0
        # Topography
        self.top = grid_params['topography']
        # set initial population across extent
        self.N0 = grid_params['N0']
        # For storing total population for each ts
        self.N = np.zeros(shape=self.tot_ts + 1)
        # population density per unit area
        if type(self.N0) in [int, float]:
            self.rho = np.zeros_like(self.xv) + \
                       ((self.N0 / (self.xv.shape[0] * self.xv.shape[1])) /
                        self.dx * self.dy)
        elif type(self.N0) == np.ndarray:
            self.rho = self.N0 / (self.dx * self.dy)

        self.beta = model_params['beta']
        self.tau = model_params['tau']
        self.kappa = model_params['kappa']  # applied to Cr
        self.mu = model_params['mu']  # applied to Ct
        self.nu = model_params['nu']  # applied to Ct
        self.y0 = model_params['ygross0']
        self.y_e = model_params['ygross_e']

        self.gm = model_params['gm']
        # Diffusion/dispersion
        self.D = model_params['d']
        self.DI = model_params['dI']
        # Exogenous growth (logistic)
        self.r = model_params['r']  # accounts for out-of-system growth
        self.Kc = model_params['Kc']  # Population carrying capacity
        self.Ygross = model_params['ygross']
        self.rho_lim = model_params['rho_limit']
        self.int_sink_rate = model_params['int_sink_rate']
        self.ext_sink_rate = model_params['ext_sink_rate']

        self.mindist = np.zeros_like(self.rho)
        self.ntwrk_fctr_nearest = np.zeros_like(self.rho)
        self.cc_nearest = np.zeros_like(self.rho)
        self.node_imp = np.zeros_like(self.rho)
        self.rho_hist, self.last_rho = None, None
        self.ct = np.zeros_like(self.rho)
        self.cr = np.zeros_like(self.rho)

        self.xi = 0
        self.Ynet = np.zeros_like(self.rho)
        self.eta = np.zeros_like(self.rho)
        self.eta_redist = np.zeros_like(self.rho) # only needed for testing
        self.qm = np.zeros_like(self.rho) # only needed for testing
        self.qneg = np.zeros_like(self.rho) # for testing

    def calc_ct_factors(self, xy, factor):
        """
        Finds the network factor of and minimum distance from the
        nearest network node for each space on the population grid

        :param xy: pandas.core.frame.DataFrame
            Dataframe holding the longitude ('lon'), latitude ('lat') and
            weighted betweenness centrality ('WBC') of each node on the network
        :param factor: str
            Centrality factor considered. Options: betwenness centrality ('WBC')
            closeness centrality ('CC') or inverse of accessibility ('a_i_inv')
        :return:
        """
        # finding minimum distance from station for each grid (assuming centre
        # of gridspace, different from .m script) and NBC of nearest station
        # (mean in case multiple stations at same distance)
        for i in np.arange(0, self.rho.shape[0]):
            for j in np.arange(0, self.rho.shape[1]):
                # nodes in the nearest gridspace(s) with nodes rather than just
                # single nearest node. Can
                d = np.hypot((self.xv[i, j] + 0.5 * self.dx) -
                             np.around(xy['lon'].values * 2) / 2,
                             (self.yv[i, j] + 0.5 * self.dy) -
                             np.around(xy['lat'].values * 2) / 2)
                self.mindist[i, j] = min(d)
                self.ntwrk_fctr_nearest[i, j] = \
                    np.mean(xy.iloc[np.where(d == min(d))][factor])
                # * 100 + 1e-3)

    def calc_rho_domain(self):
        """
        Calculates the population density across the domain.
        :return:
        """
        return np.nansum(self.rho * self.dx * self.dy) / (self.DX * self.DY)

    def calc_cc(self, xy):
        """
        Finds the closeness centrality of and minimum distance from the
        nearest network node for each space on the population grid

        :param xy: pandas.core.frame.DataFrame
            Dataframe holding the longitude ('lon'), latitude ('lat') and
            weighted betweenness centrality ('WBC') of each node on the network
        :return:
        """
        # finding minimum distance from station for each grid (assuming centre
        # of gridspace, different from .m script) and NBC of nearest station
        # (mean in case multiple stations at same distance)
        for i in np.arange(0, self.rho.shape[0]):
            for j in np.arange(0, self.rho.shape[1]):
                # nodes in the nearest gridspace(s) with nodes rather than just
                # single nearest node. Can
                d = np.hypot((self.xv[i, j] + 0.5 * self.dx) -
                             np.around(xy['lon'].values * 2) / 2,
                             (self.yv[i, j] + 0.5 * self.dy) -
                             np.around(xy['lat'].values * 2) / 2)
                self.cc_nearest[i, j] = \
                    np.mean(xy.iloc[np.where(d == min(d))]['CC'])
                # * 100 + 1e-3)

    def run_growth(self, networkxy):

        if networkxy:  # Option for no network
            self.calc_ct_factors(networkxy)

        for t in np.arange(0, self.tot_ts, self.dt):
            # if self.ts % 1000 == 0:
            #     print(self.ts)

            Cr = self.kappa * self.rho ** self.tau
            if networkxy:
                Ct = self.mu * self.mindist + self.nu * \
                     (self.gm - self.ntwrk_fctr_nearest)
            else:  # Ct is 0 across domain if no network
                Ct = np.zeros_like(self.mindist)
            K = self.Ygross - Cr - Ct
            q = np.exp(self.beta * K)

            sumq = q.sum()

            z = np.random.random(size=q.shape)

            eta = np.where(z < q/sumq, 1, 0)

            # pop/m2/d - logistic? or see Verbavatz and Barthelemy 2020
            f = eta * self.rho * self.r * (self.Kc - self.rho)

            rho_new = rho_diffusion(rho=self.rho, f=f, dx=self.dx, dy=self.dy,
                                    dt=self.dt, D=self.D)

            rho_new += q * self.dt

            self.rho += rho_new

            # in case diffusion has caused a negative population density
            self.rho = np.where(self.rho < 0, 0, self.rho)

            # Total population of extent
            self.N[self.ts] = self.rho.sum() * self.dx * self.dy

        self.ct = Ct
        self.cr = Cr
        self.Ynet = K

    def run_growth_ts(self, ts_to_run, networkxy=None):
        """

        :param networkxy:
        :param ts_to_run:
        :return:
        """
        if networkxy is not None:  # Option for no network
            self.calc_ct_factors(networkxy)

        for ts in np.arange(0, ts_to_run, self.dt):
            self.ts += 1
            # if ts % 1000 == 0:
            #     print(ts)

            Cr = self.kappa * self.rho ** self.tau
            if networkxy is not None:
                Ct = self.mu * self.mindist + self.nu * \
                     (self.gm - self.ntwrk_fctr_nearest)
            else:  # Ct is 0 across domain if no network
                Ct = np.zeros_like(self.mindist)
            K = self.Ygross - Cr - Ct
            q = np.exp(self.beta * K)

            sumq = q.sum()

            z = np.random.random(size=q.shape)

            eta = np.where(z < q / sumq, 1, 0)

            f = eta * self.rho * self.r * (self.Kc - self.rho)

            rho_new = rho_diffusion(rho=self.rho, f=f, dx=self.dx, dy=self.dy,
                                    dt=self.dt, D=self.D)
            rho_new += f * self.dt

            if self.top is not None:
                self.rho += rho_new * self.top
            else:
                self.rho += rho_new

            # in case diffusion has caused a negative population density
            self.rho = np.where(self.rho < 0, 0, self.rho)

            # Total population of extent
            self.N[ts] = self.rho.sum() * self.dx * self.dy

        self.ct = Ct
        self.cr = Cr
        self.Ynet = K

    def run_growth_ts_deterministic(self, ts_to_run, networkxy=None):
        """
        Run the diffusion and source/sink logistic function for the population
        density in a way that considers only the total population in the domain
        growth
        :param networkxy: pandas.core.frame.DataFrame
            Dataframe holding the transport network details, including the
            betweenness centrality, longitude and latitude of each node
        :param ts_to_run: int
            Time step
        """
        if networkxy is not None:  # Option for no network
            self.calc_ct_factors(networkxy)

        for ts in np.arange(0, ts_to_run, self.dt):
            # if ts % 1000 == 0:
            #     print(ts)
            Cr = self.kappa * self.rho ** self.tau
            if networkxy is not None:
                Ct = self.mu * self.mindist + self.nu * \
                     (self.gm - self.ntwrk_fctr_nearest)
            else:  # Ct is 0 across domain if no network
                Ct = np.zeros_like(self.mindist)

            if np.where(Ct < 0, 1, 0).sum() >= 1:
                print('Ct negative in some cases')
            K = self.Ygross - Cr - Ct
            Kexp = np.exp(self.beta * K)
            # Kexp /= Kexp.mean()
            Kexp /= Kexp.sum()
            # r is growth rate increase of the population due to natural growth
            # and external migration. Kc is carrying capacity.
            q = Kexp * self.rho.sum() * self.r * self.DX * self.DY * \
                (self.Kc - self.rho.sum()) / self.Kc

            rho_new = rho_diffusion(rho=self.rho, f=q, dx=self.dx, dy=self.dy,
                                    dt=self.dt, D=self.D)
            rho_new += q * self.dt

            if self.top is not None:
                self.rho += rho_new * self.top
            else:
                self.rho += rho_new

            # in case diffusion has caused a negative population density
            self.rho = np.where(self.rho < 0, 0, self.rho)

            # Total population of extent
            self.N[self.ts] = self.rho.sum() * self.dx * self.dy

        self.ct = Ct
        self.cr = Cr
        self.Ynet = K
        self.eta = Kexp

        self.ts += 1

    def run_growth_ts_rhodomain(self, ts_to_run, ntwrkxy=None, ntwrk_fctr='WBC',
                                limit_rho=False, year=None, ct_rho=False):
        """
        Run the diffusion and source/sink logistic function for the population
        density in a way that considers only the domain population density for
        the domain growth
        :param ntwrkxy: pandas.core.frame.DataFrame
            Dataframe holding the transport network details, including the
            betweenness centrality, longitude and latitude of each node
        :param ts_to_run: int
            Time step
        :param ntwrk_fctr: str
            Centrality factor considered. Options: betwenness centrality ('WBC')
            or closeness centrality ('CC')
        :param limit_rho: bool
            Switch to limit population growth based on a maximum population
            density
        """
        if ntwrkxy is not None:  # Option for no network
            self.calc_ct_factors(ntwrkxy, factor=ntwrk_fctr)

        for ts in np.arange(0, ts_to_run, self.dt):
            # if ts % 1000 == 0:
            #     print(ts)
            Cr = self.kappa * self.rho ** self.tau
            if ntwrkxy is not None:
                if ntwrk_fctr == "a_i":
                    self.node_imp = (self.ntwrk_fctr_nearest)
                else:
                    self.node_imp = (self.gm - self.ntwrk_fctr_nearest)
                Ct = self.mu * self.mindist + self.nu * self.node_imp
            else:
                Ct = np.zeros_like(self.mindist)

            if np.where(Ct < 0, 1, 0).sum() >= 1:
                print('Ct negative in some cases')
            K = self.Ygross - Cr - Ct
            if limit_rho:
                # Limiting growth in areas that exceed rho limit
                K = np.where(self.rho > self.rho_lim, np.nan, K)
            Kexp = np.exp(self.beta * K)
            # Kexp /= Kexp.mean()
            Kexp /= np.nansum(Kexp)
            # r is growth rate increase of the population due to natural growth
            # and external migration. Kc is carrying capacity.
            domain_rho = self.calc_rho_domain()
            q = Kexp * self.r * domain_rho * \
                (self.Kc - domain_rho) / self.Kc
            if limit_rho:
                # Zero growth in areas that exceed rho limit
                q = np.where(np.isnan(q), 0, q)

            if self.top is not None:
                q *= self.top

            rho_diff = rho_diffusion(rho=self.rho, f=q, dx=self.dx, dy=self.dy,
                                     dt=self.dt, D=self.D, topo=self.top)

            rho_new = (q * self.dt) + rho_diff

            if self.top is not None:
                self.rho += rho_new * self.top
            else:
                self.rho += rho_new

            # in case diffusion has caused a negative population density
            self.rho = np.where(self.rho < 0, 0, self.rho)

            # Total population of extent
            self.N[self.ts] = self.rho.sum() * self.dx * self.dy

        self.ct = Ct
        self.cr = Cr
        self.Ynet = K
        self.eta = Kexp

        self.ts += 1

    def run_growth_ts_domain_nocost(self, ts_to_run, diff_mult=1):
        """
        Run the diffusion and source/sink logistic function for the population
        density without considering living and transport costs in a way that
        considers both the local and domain population densities for the domain
        growth
        :param ntwrkxy: pandas.core.frame.DataFrame
            Dataframe holding the transport network details, including the
            betweenness centrality, longitude and latitude of each node
        :param ts_to_run: int
            Time step
        :param ntwrk_fctr: str
            Centrality factor considered. Options: betwenness centrality ('WBC')
        """

        for ts in np.arange(0, ts_to_run, self.dt):
            domain_rho = self.calc_rho_domain()
            eta = np.full_like(self.rho, fill_value=1) / self.rho.size
            xi = self.calc_xi(P=domain_rho, r=self.r, K=self.Kc, c=-1)
            q = eta * xi
            rho_new = rho_diffusion(rho=self.rho, f=q, dx=self.dx, dy=self.dy,
                                    dt=self.dt, D=self.D,
                                    topo=self.top)
            rho_new += q * self.dt

            if self.top is not None:
                self.rho += rho_new * self.top
            else:
                self.rho += rho_new

            # in case diffusion has caused a negative population density
            self.rho = np.where(self.rho < 0, 0, self.rho)

            # Total population of extent
            self.N[self.ts] = self.rho.sum() * self.dx * self.dy

            if np.where(rho_new < 0, 1, 0).sum() > 1:
                pass
            elif np.where(self.rho < 0, 1, 0).sum() > 1:
                pass

        self.ts += 1

    def run_growth_ts_domain_r(self, ts_to_run, ntwrkxy=None, ntwrk_fctr='WBC',
                               cr_fctr='rho', growth_rate=None):
        """
        Run the diffusion and source/sink logistic function for the population
        density in a way that considers only the domain population density for
        the domain growth.

        :param ntwrkxy: pandas.core.frame.DataFrame
            Dataframe holding the transport network details, including the
            betweenness centrality, longitude and latitude of each node
        :param ts_to_run: int
            Time step
        :param ntwrk_fctr: str
            Centrality factor considered. Options: inverse accessibility
            ('a_i_inv'), betweenness centrality ('WBC') or closeness centrality
            ('CC').
        :param limit_rho: bool
            Switch to limit population growth based on a maximum population
            density
        :param cr_factor: str
            Factor used to calculate living cost
        """

        if cr_fctr == 'rhohist':  # TODO: does this need to be in ts for-loop?
            self.last_rho = copy(self.rho)
            if self.rho_hist is None:
                self.rho_hist = copy(self.rho)

        if ntwrkxy is not None:  # Option for no network
            self.calc_ct_factors(ntwrkxy, factor=ntwrk_fctr)

        for ts in np.arange(0, ts_to_run, self.dt):
            self.calc_ynet(ntwrkxy=ntwrkxy, cl_fctr=cr_fctr)
            self.eta = np.exp(self.beta * self.Ynet)
            if self.top is not None:
                self.eta *= self.top

            self.eta /= np.nansum(self.eta)
            # r is growth rate increase of the population due to natural growth
            # and external migration. Kc is carrying capacity.
            domain_rho = self.calc_rho_domain()
            if growth_rate is not None:
                self.xi = \
                    self.calc_xi(P=domain_rho, r=growth_rate, K=self.Kc, c=-1)
            else:
                self.xi = self.calc_xi(P=domain_rho, r=self.r, K=self.Kc, c=-1)
            q = self.eta * self.xi

            rho_new = rho_diffusion(rho=self.rho, f=q, dx=self.dx, dy=self.dy,
                                    dt=self.dt, D=self.D, topo=self.top)
            rho_new += q * self.dt

            if self.top is not None:
                self.rho += rho_new * self.top
            else:
                self.rho += rho_new

            # in case diffusion has caused a negative population density
            self.rho = np.where(self.rho < 0, 0, self.rho)

            # Total population of extent
            self.N[self.ts] = self.rho.sum() * self.dx * self.dy

        if cr_fctr == 'rhohist':
            self.accumulate_rho()

        self.ts += 1

    def run_growth_ts_domain_ygross_only(self, ts_to_run):
        """
        Run the diffusion and source/sink logistic function for the population
        density in a way that considers only the domain population in the
        logistic growth and only variable Ygross in the cost calculation.

        :param ntwrkxy: pandas.core.frame.DataFrame
            Dataframe holding the transport network details, including the
            betweenness centrality, longitude and latitude of each node
        :param ts_to_run: int
            Time step
        :param ntwrk_fctr: str
            Centrality factor considered. Options: betwenness centrality ('WBC')
            or closeness centrality ('CC').
        """

        if type(self.Ygross) is not np.ndarray:
            raise TypeError(f'Ygross must be of type numpy.ndarray. Current '
                            f'type is {type(self.Ygross)}')

        for ts in np.arange(0, ts_to_run, self.dt):
            K = self.Ygross
            # beta is superfluous, encoded in Y0
            eta = np.exp(K)

            if self.top is not None:
                eta *= self.top
            # eta = K - np.nanmin(K)
            # eta /= eta.mean()
            eta /= np.nansum(eta)
            # r is growth rate increase of the population due to natural growth
            # and external migration. Kc is carrying capacity.
            domain_rho = self.calc_rho_domain()

            self.xi = self.calc_xi(P=domain_rho, r=self.r, K=self.Kc, c=-1)

            q = eta * self.xi

            rho_new = rho_diffusion(rho=self.rho, f=q, dx=self.dx, dy=self.dy,
                                    dt=self.dt, D=self.DI,
                                    # D=self.D*diff_mult,
                                    topo=self.top)

            rho_new += q * self.dt

            if self.top is not None:
                self.rho += rho_new * self.top
            else:
                self.rho += rho_new

            # in case diffusion has caused a negative population density
            self.rho = np.where(self.rho < 0, 0, self.rho)

            # Total population of extent
            self.N[self.ts] = self.rho.sum() * self.dx * self.dy

        self.Ynet = K
        self.eta = eta

        self.ts += 1

    def run_growth_ts_domain_ygrosscl(self, ts_to_run, diff_mult=1):
        """
        Run the diffusion and source/sink logistic function for the population
        density in a way that considers only the domain population in the
        logistic growth and only variable Ygross and CL in the cost calculation.

        :param ntwrkxy: pandas.core.frame.DataFrame
            Dataframe holding the transport network details, including the
            betweenness centrality, longitude and latitude of each node
        :param ts_to_run: int
            Time step
        :param ntwrk_fctr: str
            Centrality factor considered. Options: betwenness centrality ('WBC')
            or closeness centrality ('CC').
        """

        if type(self.Ygross) is not np.ndarray:
            raise TypeError(f'Ygross must be of type numpy.ndarray. Current '
                            f'type is {type(self.Ygross)}')

        for ts in np.arange(0, ts_to_run, self.dt):
            Cr = self.kappa * self.rho ** self.tau

            K = self.Ygross - Cr
            eta = np.exp(self.beta * K)
            # eta /= eta.mean()
            eta /= np.nansum(eta)
            # r is growth rate increase of the population due to natural growth
            # and external migration. Kc is carrying capacity.
            domain_rho = self.calc_rho_domain()
            q = eta * self.r * domain_rho * (self.Kc - domain_rho) / self.Kc

            rho_new = rho_diffusion(rho=self.rho, f=q, dx=self.dx, dy=self.dy,
                                    dt=self.dt, D=self.D * diff_mult,
                                    topo=self.top)

            rho_new += q * self.dt

            if self.top is not None:
                self.rho += rho_new * self.top
            else:
                self.rho += rho_new

            # in case diffusion has caused a negative population density
            self.rho = np.where(self.rho < 0, 0, self.rho)

            # Total population of extent
            self.N[self.ts] = self.rho.sum() * self.dx * self.dy

        self.cr = Cr
        self.Ynet = K
        self.eta = eta

        self.ts += 1

    def run_growth_ts_domain_sinketa(self, ts_to_run, ntwrk_fctr, a_i_lim=None,
                                     rho_lim=None, sink_rate=None,
                                     ntwrkxy=None):
        """
        Run the diffusion and source/sink logistic function for the population
        density in a way that considers both the local and domain population
        densities for the domain growth
        :param ntwrkxy: pandas.core.frame.DataFrame
            Dataframe holding the transport network details, including the
            betweenness centrality, longitude and latitude of each node
        :param ts_to_run: int
            Time step
        :param ntwrk_fctr: str
            Centrality factor considered. Options: betwenness centrality ('WBC')
            or closeness centrality ('CC').
        :param limit_rho: bool
            Switch to limit population growth based on a maximum population
            density
        """
        if ntwrkxy is not None:  # Option for no network
            self.calc_ct_factors(ntwrkxy, factor=ntwrk_fctr)

        for ts in np.arange(0, ts_to_run, self.dt):
            # if ts % 1000 == 0:
            #     print(ts)
            Cr = self.kappa * self.rho ** self.tau

            if ntwrkxy is not None:
                self.node_imp = (self.gm - self.ntwrk_fctr_nearest)
                Ct = self.mu * self.mindist + self.nu * self.node_imp

            else:  # Ct is 0 or across domain if no network
                rho_no_zero = np.where(self.rho == 0, 1, self.rho)
                Ct = np.zeros_like(self.mindist)

            if np.where(Ct < 0, 1, 0).sum() >= 1:
                print('Ct negative in some cases')
            K = self.Ygross - Cr - Ct
            eta = np.exp(self.beta * K)
            eta /= np.nansum(eta)
            # r is growth rate increase of the population due to natural growth
            # and external migration. Kc is carrying capacity.
            domain_rho = self.calc_rho_domain()
            # FAVOUR AREAS WITH HIGH CR AND LOW CT
            sink_cr = \
                np.where(np.logical_and(self.ntwrk_fctr_nearest >= a_i_lim,
                                        self.rho >= rho_lim), Cr - Ct, 0)
            if sink_cr.sum() > 0:
                # CALC SUM OF POPULATION THAT IS MOVING
                sink_pop = \
                    np.where(np.logical_and(self.ntwrk_fctr_nearest >= a_i_lim,
                                            self.rho >= rho_lim), self.rho, 0)
                sink_cr_eta = sink_cr / sink_cr.sum()
                sink_cr_eta /= np.nanmax(sink_cr_eta)
                sink_q = sink_cr_eta * sink_rate * sink_pop
                # REMOVE POPULATION MOVING FROM RHO
                self.rho -= sink_q
            q = eta * self.r * domain_rho * (self.Kc - domain_rho) / self.Kc
            if sink_cr.sum() > 0:
                # ADD EXTRA POPULATION MOVING OUT OF SINK
                move_eta = \
                    np.where(np.logical_and(self.ntwrk_fctr_nearest > a_i_lim,
                                            self.rho > rho_lim), 0, eta)
                move_eta /= np.nansum(move_eta)
                q += move_eta * sink_q.sum()
            # DISTRIBUTING BY ETA
            rho_new = rho_diffusion(rho=self.rho, f=q, dx=self.dx, dy=self.dy,
                                    dt=self.dt, D=self.D, topo=self.top)
            rho_new += q * self.dt

            if self.top is not None:
                self.rho += rho_new * self.top
            else:
                self.rho += rho_new

            # in case diffusion has caused a negative population density
            self.rho = np.where(self.rho < 0, 0, self.rho)

            # Total population of extent
            self.N[self.ts] = self.rho.sum() * self.dx * self.dy

        self.ct = Ct
        self.cr = Cr
        self.Ynet = K
        self.eta = eta

        self.ts += 1

    def run_growth_ts_domain_sinknegeta(self, ts_to_run, sink_rate=None,
                                        remove=False, r_ynet=False,
                                        calc_ynet=False, ntwrkxy=None,
                                        cl_fctr=None, ntwrk_fctr=None):
        """
        Removes sink population from areas with negative Ynet. This sink
        population may be redistributed within the domain or removed from the
        domain.

        :param ts_to_run: int
            Time step
        :param sink_rate: float, int
            Rate of internal population decrease
        :param remove: bool
            If True, sink population is removed from the domain entirely,
            otherwise the sink population is redistributed within the domain
        :param r_ynet: bool
            If True, the sink rate is proportional to lcoal Ynet
        """

        if calc_ynet:
            if cl_fctr is not None:
                if cl_fctr == 'rhohist':  # TODO: does this need to be in ts for-loop?
                    self.last_rho = copy(self.rho)
                    if self.rho_hist is None:
                        self.rho_hist = copy(self.rho)

                if ntwrkxy is not None:  # Option for no network
                    self.calc_ct_factors(ntwrkxy, factor=ntwrk_fctr)
                self.calc_ynet(ntwrkxy=ntwrkxy, cl_fctr=cl_fctr, dt=ts_to_run)
            else:
                raise ValueError('Network or CL factor are not specified')

        for ts in np.arange(0, ts_to_run, self.dt):
            eta = np.exp(self.beta * self.Ynet) - 1

            eta /= np.nansum(eta)
            # r is growth rate increase of the population due to natural growth
            # and external migration. Kc is carrying capacity.
            domain_rho = self.calc_rho_domain()
            # FAVOUR AREAS WITH HIGH CR AND LOW CT
            sink = np.where(eta <= 0, 1, 0)
            if sink.sum() > 0:
                # CALC SUM OF POPULATION THAT IS MOVING
                sink_pop = np.where(sink, self.rho, 0)

                if r_ynet:
                    # AREAS WITH GREATER YNET EXPERIENCE MORE SINK
                    ynet_neg = np.where(self.Ynet < 0, self.Ynet, np.nan)
                    ynet_neg /= np.nansum(ynet_neg)
                    ynet_neg = np.where(np.isnan(ynet_neg), 0, ynet_neg)
                    sink_q = sink * ynet_neg * sink_rate * sink_pop
                else:
                    sink_q = sink * sink_rate * sink_pop
                # REMOVE POPULATION MOVING FROM RHO
                self.rho -= sink_q

            if remove:
                # Sink population is removed from the domain entirely
                pass
            elif sink.sum() > 0:
                # Sink population is redistributed across the domain
                # Find eta where there is no sink
                q = self.eta * sink_q.sum()

                self.rho += q

                self.eta_redist = self.eta
                self.qm = q

                self.qneg = sink_q

            # if sink has caused a negative population density
            self.rho = np.where(self.rho < 0, 0, self.rho)

            # Total population of extent
            self.N[self.ts] = self.rho.sum() * self.dx * self.dy

    def run_growth_ts_domain_nosink(self, ts_to_run, calc_ynet=False,
                                    ntwrkxy=None,
                                    cl_fctr=None, ntwrk_fctr=None):
        """
        Removes sink population from areas with negative Ynet. This sink
        population may be redistributed within the domain or removed from the
        domain.

        :param ts_to_run: int
            Time step
        :param sink_rate: float, int
            Rate of internal population decrease
        :param remove: bool
            If True, sink population is removed from the domain entirely,
            otherwise the sink population is redistributed within the domain
        :param r_ynet: bool
            If True, the sink rate is proportional to lcoal Ynet
        """

        if calc_ynet:
            if cl_fctr is not None:
                if cl_fctr == 'rhohist':  # TODO: does this need to be in ts for-loop?
                    self.last_rho = copy(self.rho)
                    if self.rho_hist is None:
                        self.rho_hist = copy(self.rho)

                if ntwrkxy is not None:  # Option for no network
                    self.calc_ct_factors(ntwrkxy, factor=ntwrk_fctr)
                self.calc_ynet(ntwrkxy=ntwrkxy, cl_fctr=cl_fctr)
            else:
                raise ValueError('Network or CL factor are not specified')

        for ts in np.arange(0, ts_to_run, self.dt):
            eta = np.exp(self.beta * self.Ynet) - 1

            eta /= np.nansum(eta)
            # r is growth rate increase of the population due to natural growth
            # and external migration. Kc is carrying capacity.
            domain_rho = self.calc_rho_domain()

            q = self.eta
            self.rho += q

            self.eta_redist = self.eta
            self.qm = q

            # if sink has caused a negative population density
            self.rho = np.where(self.rho < 0, 0, self.rho)

            # Total population of extent
            self.N[self.ts] = self.rho.sum() * self.dx * self.dy

    def run_growth_ts_domain_ext_sink_simple(self, ts_to_run, sink_rate):
        """
        Removes sink population from areas with negative Ynet. This sink
        population may be redistributed within the domain or removed from the
        domain.

        :param ts_to_run: int
            Time step
        :param sink_rate: float, int
            Rate of external population decrease
        """

        for ts in np.arange(0, ts_to_run, self.dt):
            # r is growth rate increase of the population due to natural growth
            # and external migration. Kc is carrying capacity.
            sink_q = sink_rate * self.rho

            self.rho -= sink_q
            self.qneg = sink_q

            # if sink has caused a negative population density
            self.rho = np.where(self.rho < 0, 0, self.rho)

            # Total population of extent #TODO Decide what to do with this

    def run_growth_ts_domain_ext_sink_eta(self, ts_to_run, sink_rate, beta,
                                          K=None, r=None, c=None):
        """
        Removes sink population from areas with negative Ynet. This sink
        population may be redistributed within the domain or removed from the
        domain.

        :param ts_to_run: int
            Time step
        :param sink_rate: float, int
            Rate of external population decrease
        """

        for ts in np.arange(0, ts_to_run, self.dt):

            # r is growth rate increase of the population due to natural growth
            # and external migration. Kc is carrying capacity.

            # Total P that will be removed from the domain
            sink_xi = sink_rate * self.N[self.ts - 1]

            if K is not None and r is not None and c is not None:
                domain_rho = self.calc_rho_domain()
                sink_xi = self.calc_xi(P=domain_rho, r=r, K=K, c=c)
            # Inverting Ynet so higher eta in high cost areas
            eta_sink = np.exp(beta * (-1 * self.Ynet))
            eta_sink /= eta_sink.sum()

            sink_q = eta_sink * sink_xi

            self.rho -= sink_q
            self.qneg = sink_q

            # if sink has caused a negative population density
            self.rho = np.where(self.rho < 0, 0, self.rho)

            # Total population of extent TODO uncomment
            # self.N[self.ts] = self.rho.sum() * self.dx * self.dy

    def run_growth_ts_rhodomain_cronly(self, ts_to_run, limit_rho=False,
                                       year=None):
        """
        Run the diffusion and source/sink logistic function for the population
        density in a way that considers only the domain population density for
        the domain growth
        :param ntwrkxy: pandas.core.frame.DataFrame
            Dataframe holding the transport network details, including the
            betweenness centrality, longitude and latitude of each node
        :param ts_to_run: int
            Time step
        :param limit_rho: bool
            Switch to limit population growth based on a maximum population
            density
        """

        for ts in np.arange(0, ts_to_run, self.dt):
            Cr = self.kappa * self.rho ** self.tau

            K = self.Ygross - Cr
            if limit_rho:
                # Limiting growth in areas that exceed rho limit
                K = np.where(self.rho > self.rho_lim, np.nan, K)
            Kexp = np.exp(self.beta * K)
            Kexp /= np.nansum(Kexp)
            # r is growth rate increase of the population due to natural growth
            # and external migration. Kc is carrying capacity.
            domain_rho = self.calc_rho_domain()
            q = Kexp * self.r * domain_rho * \
                (self.Kc - domain_rho) / self.Kc
            if limit_rho:
                # Zero growth in areas that exceed rho limit
                q = np.where(np.isnan(q), 0, q)
            rho_new = rho_diffusion(rho=self.rho, f=q, dx=self.dx, dy=self.dy,
                                    dt=self.dt, D=self.D)
            rho_new += q * self.dt

            if self.top is not None:
                self.rho += rho_new * self.top
            else:
                self.rho += rho_new

            # in case diffusion has caused a negative population density
            self.rho = np.where(self.rho < 0, 0, self.rho)

            # Total population of extent
            self.N[self.ts] = self.rho.sum() * self.dx * self.dy

        self.cr = Cr
        self.Ynet = K
        self.eta = Kexp

        self.ts += 1

    def run_growth_ts_nocost(self, ts_to_run):
        # TODO: remove year after testing
        """

        :param networkxy:
        :param ts_to_run:
        :return:
        """
        for ts in np.arange(0, ts_to_run, self.dt):
            # r is growth rate increase of the population due to natural growth
            # and external migration. Kc is carrying capacity.
            q = self.rho * self.r * \
                (self.Kc - self.rho) / self.Kc

            rho_new = rho_diffusion(rho=self.rho, f=q, dx=self.dx, dy=self.dy,
                                    dt=self.dt, D=self.D)
            rho_new += q * self.dt

            self.rho += rho_new

            # in case diffusion has caused a negative population density
            self.rho = np.where(self.rho < 0, 0, self.rho)

            # Total population of extent
            self.N[self.ts] = self.rho.sum() * self.dx * self.dy

        self.ts += 1

    def run_growth_ts_spatialr(self, ts_to_run, networkxy=None):
        """
        Run the diffusion and source/sink logistic function for the population
        density in a way that considers only the local population density for
        the domain growth
        :param networkxy: pandas.core.frame.DataFrame
            Dataframe holding the transport network details, including the
            betweenness centrality, longitude and latitude of each node
        :param ts_to_run: int
            Time step
        """
        if networkxy is not None:  # Option for no network
            self.calc_ct_factors(networkxy)

        for ts in np.arange(0, ts_to_run, self.dt):
            Cr = self.kappa * self.rho ** self.tau
            if networkxy is not None:
                Ct = self.mu * self.mindist + self.nu * \
                     (self.gm - self.ntwrk_fctr_nearest)
            else:  # Ct is 0 across domain if no network
                Ct = np.zeros_like(self.mindist)

            if np.where(Ct < 0, 1, 0).sum() >= 1:
                print('Ct negative in some cases')
            eta = self.Ygross - Cr - Ct
            eta_exp = np.exp(self.beta * eta)
            eta_exp /= eta_exp.sum()
            # r is growth rate increase of the population due to natural growth
            # and external migration. Kc is carrying capacity.
            q = (eta_exp + self.r) * self.rho * \
                (self.Kc - self.rho) / self.Kc

            rho_new = rho_diffusion(rho=self.rho, f=q, dx=self.dx, dy=self.dy,
                                    dt=self.dt, D=self.D)
            rho_new += q * self.dt

            self.rho += rho_new

            # in case diffusion has caused a negative population density
            self.rho = np.where(self.rho < 0, 0, self.rho)

            # Total population of extent
            self.N[self.ts] = self.rho.sum() * self.dx * self.dy

        self.ts += 1

    def run_growth_ts_domainlocal_r(self, ts_to_run, ntwrkxy=None,
                                    ntwrk_fctr='WBC', limit_rho=False,
                                    cr_fctr='rho', ct_rho=False):
        """
        Run the diffusion and source/sink logistic function for the population
        density in a way that considers both the local and domain population
        densities for the domain growth
        :param ntwrkxy: pandas.core.frame.DataFrame
            Dataframe holding the transport network details, including the
            betweenness centrality, longitude and latitude of each node
        :param ts_to_run: int
            Time step
        :param ntwrk_fctr: str
            Centrality factor considered. Options: betwenness centrality ('WBC')
            or closeness centrality ('CC').
        :param limit_rho: bool
            Switch to limit population growth based on a maximum population
            density
        """
        if ntwrkxy is not None:  # Option for no network
            self.calc_ct_factors(ntwrkxy, factor=ntwrk_fctr)

        for ts in np.arange(0, ts_to_run, self.dt):
            if cr_fctr == 'rho':
                Cr = self.kappa * self.rho ** self.tau
            elif cr_fctr == 'CC':
                self.calc_cc(xy=ntwrkxy)
                Cr = self.kappa * self.cc_nearest ** self.tau
            if ntwrkxy is not None:
                self.node_imp = (self.gm - self.ntwrk_fctr_nearest)
                Ct = self.mu * self.mindist + self.nu * self.node_imp
            elif self.phi is not None and self.chi is not None and ct_rho:
                Ct = self.phi * self.rho ** self.chi
            else:  # Ct is 0 or across domain if no network
                Ct = np.zeros_like(self.mindist)

            if np.where(Ct < 0, 1, 0).sum() >= 1:
                print('Ct negative in some cases')
            K = self.Ygross - Cr - Ct
            if limit_rho:
                # Limiting growth in areas that exceed rho limit
                K = np.where(self.rho > self.rho_lim, np.nan, K)
            eta = np.exp(self.beta * K)
            # eta /= eta.mean()
            eta /= np.nansum(eta)
            # r is growth rate increase of the population due to natural growth
            # and external migration. Kc is carrying capacity.
            domain_rho = self.calc_rho_domain()
            q = eta * self.r * self.rho * \
                (self.Kc - domain_rho) / self.Kc
            if limit_rho:
                # Zero growth in areas that exceed rho limit
                q = np.where(np.isnan(q), 0, q)
            rho_new = rho_diffusion(rho=self.rho, f=q, dx=self.dx, dy=self.dy,
                                    dt=self.dt, D=self.D, topo=self.top)
            rho_new += q * self.dt

            if self.top is not None:
                self.rho += rho_new * self.top
            else:
                self.rho += rho_new

            # in case diffusion has caused a negative population density
            self.rho = np.where(self.rho < 0, 0, self.rho)

            # Total population of extent
            self.N[self.ts] = self.rho.sum() * self.dx * self.dy

        self.ct = Ct
        self.cr = Cr
        self.Ynet = K
        self.eta = eta

        self.ts += 1

    def run_growth_ts_domainlocal_sink(self, ts_to_run, ntwrk_fctr,
                                       a_i_lim=None, rho_lim=None,
                                       sink_rate=None, ntwrkxy=None):
        """
        Run the diffusion and source/sink logistic function for the population
        density in a way that considers both the local and domain population
        densities for the domain growth
        :param ntwrkxy: pandas.core.frame.DataFrame
            Dataframe holding the transport network details, including the
            betweenness centrality, longitude and latitude of each node
        :param ts_to_run: int
            Time step
        :param ntwrk_fctr: str
            Centrality factor considered. Options: betwenness centrality ('WBC')
            or closeness centrality ('CC').
        :param limit_rho: bool
            Switch to limit population growth based on a maximum population
            density
        """
        if ntwrkxy is not None:  # Option for no network
            self.calc_ct_factors(ntwrkxy, factor=ntwrk_fctr)

        for ts in np.arange(0, ts_to_run, self.dt):
            Cr = self.kappa * self.rho ** self.tau

            if ntwrkxy is not None:
                self.node_imp = (self.gm - self.ntwrk_fctr_nearest)  # / self.gm
                Ct = self.mu * self.mindist + self.nu * self.node_imp

            else:  # Ct is 0 or across domain if no network
                Ct = np.zeros_like(self.mindist)

            if np.where(Ct < 0, 1, 0).sum() >= 1:
                print('Ct negative in some cases')
            K = self.Ygross - Cr - Ct
            eta = np.exp(self.beta * K)
            # eta /= eta.mean()
            eta /= np.nansum(eta)
            # r is growth rate increase of the population due to natural growth
            # and external migration. Kc is carrying capacity.
            domain_rho = self.calc_rho_domain()
            # CALC SUM OF POPULATION THAT IS MOVING
            sink_pop = \
                np.where(np.logical_and(self.ntwrk_fctr_nearest >= a_i_lim,
                                        self.rho >= rho_lim),
                         self.rho, 0) * sink_rate
            sink_sum = sink_pop.sum()
            # REMOVE POPULATION MOVING FROM RHO
            self.rho -= sink_pop
            q = eta * self.r * self.rho * \
                (self.Kc - domain_rho) / self.Kc
            # ADD EXTRA POPULATION MOVING OUT OF SINK
            sink_eta = np.where(self.ntwrk_fctr_nearest > a_i_lim, 0, eta)
            sink_eta /= np.nansum(sink_eta)
            q += sink_eta * sink_sum
            # DISTRIBUTING BY ETA
            rho_new = rho_diffusion(rho=self.rho, f=q, dx=self.dx, dy=self.dy,
                                    dt=self.dt, D=self.D, topo=self.top)
            rho_new += q * self.dt

            if self.top is not None:
                self.rho += rho_new * self.top
            else:
                self.rho += rho_new

            # in case diffusion has caused a negative population density
            self.rho = np.where(self.rho < 0, 0, self.rho)

            # Total population of extent
            self.N[self.ts] = self.rho.sum() * self.dx * self.dy

        self.ct = Ct
        self.cr = Cr
        self.Ynet = K
        self.eta = eta

        self.ts += 1

    def run_growth_ts_domainlocal_sinketa(self, ts_to_run, ntwrk_fctr,
                                          a_i_lim=None, rho_lim=None,
                                          sink_rate=None, ntwrkxy=None):
        """
        Run the diffusion and source/sink logistic function for the population
        density in a way that considers both the local and domain population
        densities for the domain growth
        :param ntwrkxy: pandas.core.frame.DataFrame
            Dataframe holding the transport network details, including the
            betweenness centrality, longitude and latitude of each node
        :param ts_to_run: int
            Time step
        :param ntwrk_fctr: str
            Centrality factor considered. Options: betwenness centrality ('WBC')
            or closeness centrality ('CC').
        :param limit_rho: bool
            Switch to limit population growth based on a maximum population
            density
        """
        if ntwrkxy is not None:  # Option for no network
            self.calc_ct_factors(ntwrkxy, factor=ntwrk_fctr)

        for ts in np.arange(0, ts_to_run, self.dt):
            # if ts % 1000 == 0:
            #     print(ts)
            Cr = self.kappa * self.rho ** self.tau

            if ntwrkxy is not None:
                self.node_imp = (self.gm - self.ntwrk_fctr_nearest)  # / self.gm
                Ct = self.mu * self.mindist + self.nu * self.node_imp

            else:  # Ct is 0 or across domain if no network
                Ct = np.zeros_like(self.mindist)

            if np.where(Ct < 0, 1, 0).sum() >= 1:
                print('Ct negative in some cases')
            K = self.Ygross - Cr - Ct
            eta = np.exp(self.beta * K)
            eta /= np.nansum(eta)
            # r is growth rate increase of the population due to natural growth
            # and external migration. Kc is carrying capacity.
            domain_rho = self.calc_rho_domain()
            # FAVOUR AREAS WITH HIGH CR AND LOW CT
            sink_cr = \
                np.where(np.logical_and(self.ntwrk_fctr_nearest >= a_i_lim,
                                        self.rho >= rho_lim), Cr - Ct, 0)
            if sink_cr.sum() > 0:
                # CALC SUM OF POPULATION THAT IS MOVING
                sink_pop = \
                    np.where(np.logical_and(self.ntwrk_fctr_nearest >= a_i_lim,
                                            self.rho >= rho_lim), self.rho, 0)
                sink_cr_eta = sink_cr / sink_cr.sum()
                sink_cr_eta /= np.nanmax(sink_cr_eta)
                sink_q = sink_cr_eta * sink_rate * sink_pop
                # REMOVE POPULATION MOVING FROM RHO
                self.rho -= sink_q
            q = eta * self.r * self.rho * \
                (self.Kc - domain_rho) / self.Kc
            if sink_cr.sum() > 0:
                # ADD EXTRA POPULATION MOVING OUT OF SINK
                move_eta = \
                    np.where(np.logical_and(self.ntwrk_fctr_nearest > a_i_lim,
                                            self.rho > rho_lim), 0, eta)
                move_eta /= np.nansum(move_eta)
                q += move_eta * sink_q.sum()
            # DISTRIBUTING BY ETA
            rho_new = rho_diffusion(rho=self.rho, f=q, dx=self.dx, dy=self.dy,
                                    dt=self.dt, D=self.D, topo=self.top)
            rho_new += q * self.dt

            if self.top is not None:
                self.rho += rho_new * self.top
            else:
                self.rho += rho_new

            # in case diffusion has caused a negative population density
            self.rho = np.where(self.rho < 0, 0, self.rho)

            # Total population of extent
            self.N[self.ts] = self.rho.sum() * self.dx * self.dy

        self.ct = Ct
        self.cr = Cr
        self.Ynet = K
        self.eta = eta

        self.ts += 1

    def run_growth_ts_domainlocal_nocost(self, ts_to_run, diff_mult=1):
        """
        Run the diffusion and source/sink logistic function for the population
        density without considering living and transport costs in a way that
        considers both the local and domain population densities for the domain
        growth
        :param ntwrkxy: pandas.core.frame.DataFrame
            Dataframe holding the transport network details, including the
            betweenness centrality, longitude and latitude of each node
        :param ts_to_run: int
            Time step
        :param ntwrk_fctr: str
            Centrality factor considered. Options: betwenness centrality ('WBC')
        """

        for ts in np.arange(0, ts_to_run, self.dt):
            domain_rho = self.calc_rho_domain()
            eta = np.full_like(self.rho, fill_value=1) / self.rho.size
            q = eta * self.r * self.rho * \
                (self.Kc - domain_rho) / self.Kc
            rho_new = rho_diffusion(rho=self.rho, f=q, dx=self.dx, dy=self.dy,
                                    dt=self.dt, D=self.D * diff_mult,
                                    topo=self.top)
            rho_new += q * self.dt

            if self.top is not None:
                self.rho += rho_new * self.top
            else:
                self.rho += rho_new

            # in case diffusion has caused a negative population density
            self.rho = np.where(self.rho < 0, 0, self.rho)

            # Total population of extent
            self.N[self.ts] = self.rho.sum() * self.dx * self.dy

            if np.where(rho_new < 0, 1, 0).sum() > 1:
                pass
            elif np.where(self.rho < 0, 1, 0).sum() > 1:
                pass

        self.ts += 1

    def run_diffusion_ts(self, ts_to_run):
        """

        :param networkxy:
        :param ts_to_run:
        :return:
        """

        for ts in np.arange(0, ts_to_run, self.dt):
            self.ts += 1
            # pop/m2/d - logistic? or see Verbavatz and Barthelemy 2020
            f = np.zeros_like(self.rho)

            Cr = self.kappa * self.rho ** self.tau

            rho_new = rho_diffusion(rho=self.rho, f=f, dx=self.dx, dy=self.dy,
                                    dt=self.dt, D=self.D)

            self.rho += rho_new

            # Total population of extent
            self.N[ts] = self.rho.sum() * self.dx * self.dy

    def run_diffusion_simple_growth_ts(self, ts_to_run):
        """

        :param networkxy:
        :param ts_to_run:
        :return:
        """

        for ts in np.arange(0, ts_to_run, self.dt):
            self.ts += 1
            # pop/m2/d - logistic? or see Verbavatz and Barthelemy 2020
            f = np.zeros_like(self.rho)

            rho_new = rho_diffusion(rho=self.rho, f=f, dx=self.dx, dy=self.dy,
                                    dt=self.dt, D=self.D)
            rho_new += q * self.dt
            rho_new = np.where(rho_new < 0, 0, rho_new)

            self.rho += rho_new

            # Total population of extent
            self.N[ts] = self.rho.sum() * self.dx * self.dy

    def calc_xi(self, P, r, K, c):
        """
        Calculate xi using the modified logistic function from Fulda (1981)

        :param P: float, int
            Domain population density
        :param r: float, int
            Growth rate
        :param K: float, int
            Carrying capacity
        :param c: float, int
            Modifier for flexible logistic func., c = -1 for growth, 0 < c < 1
            for population decline
        :return:
        """
        xi = r * P * (K + (c * P)) / K

        return xi

    def calc_ynet(self, ntwrkxy, cl_fctr):
        """
        Calculates Ynet under different scenarios (different CL factors, network
         existence).

        :param ntwrkxy:
        :param cl_fctr:
        :return:
        """
        if cl_fctr == 'rhohist':
            self.last_rho = copy(self.rho)
            if self.rho_hist is None:
                self.rho_hist = copy(self.rho)

        if cl_fctr == 'rho':  # local rho only
            self.cr = self.kappa * self.rho ** self.tau
        elif cl_fctr == 'ccrho' and self.phi is not None and \
                self.chi is not None:  # closeness centrality and local rho
            self.calc_cc(xy=ntwrkxy)
            self.cr = self.phi * self.cc_nearest ** self.chi + \
                      self.kappa * self.rho ** self.tau
        elif cl_fctr == 'cc' and self.phi is not None and \
                self.chi is not None:  # closeness centrality only
            self.calc_cc(xy=ntwrkxy)
            self.cr = self.phi * self.cc_nearest ** self.chi
        elif cl_fctr == 'rhohist':  # historical accumulated rho
            self.cr = self.kappa * self.rho_hist ** self.tau

        if ntwrkxy is not None:
            self.node_imp = (self.gm - self.ntwrk_fctr_nearest)  # / self.gm
            self.ct = self.mu * self.mindist + self.nu * self.node_imp
        else:  # Ct is 0 or across domain if no network
            self.ct = np.zeros_like(self.mindist)

        if np.where(self.ct < 0, 1, 0).sum() >= 1:
            print('Ct negative in some cases')
        # Set Ynet for use in parent functions
        self.Ynet = self.Ygross - self.cr - self.ct

    def accumulate_rho(self):
        """
        Accumulates the positive difference in rho.
        """
        rho_diff = self.rho - self.last_rho
        rho_diff = np.where(rho_diff > 0, rho_diff, 0)

        self.rho_hist += rho_diff
        pass

    def set_Ygross_i(self, average=False):
        Ygross_arr = self.y0 * (self.rho ** self.y_e)
        if average:
            Ygross_arr = mean_smooth(Ygross_arr, 3, average=False)
        self.Ygross = Ygross_arr

    def set_gm(self, new_gm):
        """
        Set gm value
        :param new_gm: float, int
            New gm value
        """
        self.gm = new_gm

    def set_Ygross(self, ygross):
        """
        Set gm value
        :param new_gm: float, int
            New gm value
        """
        self.Ygross = ygross

    def set_Kc(self, kc):
        """
        Set carrying capacity Kc
        :return:
        """

        self.Kc = kc

    def get_bc_nearest(self, extent_mask=None):

        return self.ntwrk_fctr_nearest

    def get_wbc_nearest(self, border=None):

        return self.ntwrk_fctr_nearest[border:-border, border:-border]

    def get_CT(self, border=None):

        return self.ntwrk_fctr_nearest[border:-border, border:-border]

    def get_CR(self, border=None):

        pass

    def get_K(self, border=None):

        pass

    def get_rho(self, border=None):

        pass

    def get_N(self, border=None):

        pass

    def get_mindist(self, border=None):

        pass


class NetworkGraph:

    def __init__(self, graph, coords, nodes):
        self.graph = graph  # networkx graph object
        self.coords = coords  # dict of k (node id), v (tuple(lat,lon))
        self.nodes = nodes  # list?
        self.wbc = None  # dict
        self.wbc_cols = None  # list
        self.cc = None
        self.cc_cols = None

        self.set_edge_weights()

    def set_wbc(self):
        """
        Calculates the betweenness centrality of the graph and sets colour
        values for later plotting
        :return:
        """
        self.wbc = nx.betweenness_centrality(self.graph, normalized=True,
                                             endpoints=True)
        self.wbc_cols = [self.wbc.get(node) for node in self.graph.nodes]

    def set_cc(self):
        """
        Calculates the betweenness centrality of the graph and sets colour
        values for later plotting
        :return:
        """
        self.cc = nx.closeness_centrality(self.graph, wf_improved=True,
                                          distance='weight')
        self.cc_cols = [self.cc.get(node) for node in self.graph.nodes]

    def set_edge_weights(self):
        """
        Sets the graph's edge weights as the Euclidean distance between node
        coordinates
        :return:
        """
        for e in list(self.graph.edges):
            self.graph[e[0]][e[1]]['weight'] = np.linalg.norm(
                np.array(self.coords[e[0]]) -
                np.array(self.coords[e[1]]))

    def get_xy_wbc(self):
        networkdf = pd.DataFrame.from_dict(self.coords, orient='index')
        networkdf.columns = ['lon', 'lat']

        if self.wbc is None:
            self.set_wbc()

        wbcdf = pd.DataFrame.from_dict(self.wbc, orient='index')
        wbcdf.columns = ['WBC']

        if len(np.intersect1d(networkdf.index, wbcdf.index)) == \
                len(networkdf.index):
            networkdf = networkdf.join(other=wbcdf, how='left')

            return networkdf

        else:
            raise KeyError('networkdf and wbcdf index not the same')

    def get_xy_wbc_cc(self):
        networkdf = pd.DataFrame.from_dict(self.coords, orient='index')
        networkdf.columns = ['lon', 'lat']

        if self.wbc is None:
            self.set_wbc()

        wbcdf = pd.DataFrame.from_dict(self.wbc, orient='index')
        wbcdf.columns = ['WBC']

        ccdf = pd.DataFrame.from_dict(self.cc, orient='index')
        ccdf.columns = ['CC']

        if (len(np.intersect1d(networkdf.index, wbcdf.index)) == \
            len(networkdf.index)) and \
                len(np.intersect1d(networkdf.index, ccdf.index)) == \
                len(networkdf.index):
            networkdf = networkdf.join(other=wbcdf, how='left')

            return networkdf

        else:
            raise KeyError('networkdf and wbcdf index not the same')


class SubGraph():

    def __init__(self, graph, coords, nodes, subgraph):
        self.graph = graph  # networkx graph object
        self.coords = coords  # dict of k (node id), v (tuple(lat,lon))
        self.nodes = nodes  # list?
        self.wbc = None  # dict
        self.wbc_cols = None  # list

        self.subgraph = subgraph

        self.set_sub_edge_weights()

    def add_new_edges(self, distance, hub_limit):
        """
        Adds edges between disconnected subgraphs, assuming that an edge will
        exist between a given number of consecutive nodes.

        :param graph: networkx.classes.graph.Graph
            Complete graph from which the subgraph is derived
        :param subgraph: networkx.classes.graph.Graph
            Subgraph of complete graph
        :param distance: int
            Maximum number of connected nodes in complete graph from which we
            can assume new endpoints in subgraph have a connection
        :param hub_limit: int
            Minimum number of degrees for a node to be considered a hub
        :return:
        """

        if nx.is_frozen(self.subgraph):
            raise nx.NetworkXError("Frozen graph can't be modified")
        if len(list(nx.connected_components(self.graph))) == \
                len(list(nx.connected_components(self.subgraph))):
            # if subgraph is as disconnected as graph, then no new endpoints
            print('Not modifying')
            pass
        else:
            # find disconnected components of subgraph
            cnt_comp = list(nx.connected_components(self.subgraph))
            # Find new endpoints of disconnected subgraph components
            dscnt_ends = self.check_endpoints()
            if len(dscnt_ends) == 0 and len(self.subgraph) > 0:
                raise ValueError("Subgraph is disconnected but no new "
                                 "endpoints found")

            blcklst = []
            # Collect potential connections between disconnected components
            for node in dscnt_ends:
                # Find potential new nodes from whole graph
                pot_nodes = []
                short_path = nx.single_source_shortest_path(self.graph,
                                                            source=node,
                                                            cutoff=distance)

                for v in short_path.values():
                    for n in v:
                        if (n == node) or (n not in self.subgraph.nodes) or \
                                self.subgraph.has_edge(node, n) or \
                                (n in nx.single_source_shortest_path( \
                                        self.subgraph, source=node,
                                    cutoff=distance).keys()):
                            pass
                        elif (n in pot_nodes):
                            # Stops any nodes past already viable node being
                            # added. Works as v in order of distance from node
                            break

                        else:
                            pot_nodes.append(n)

                for pn in pot_nodes:
                    self.create_connecting_edge(src=node, trgt=pn)

    def check_endpoints(self):
        """
        Checks if new endpoints exist in subgraph where they do not exist in
        complete graph.

        :param graph: networkx.classes.graph.Graph
            Complete graph from which the subgraph is derived
        :param subgraph: networkx.classes.graph.Graph
            Subgraph of complete graph
        :return:
        """
        grph_deg = dict(self.graph.degree)
        dscnt_ends = []
        for node, deg in dict(self.subgraph.degree).items():
            if (deg == 1) and (grph_deg[node] > 1):
                dscnt_ends.append(node)
            elif (deg == 0) and grph_deg[node] > 0:
                dscnt_ends.append(node)

        return dscnt_ends

    def create_connecting_edge(self, src, trgt):
        """
        Adds an edge a subgraph, with summed weight of the original path from
        the complete graph.

        :param graph: networkx.classes.graph.Graph
            Complete graph from which the subgraph is derived
        :param subgraph: networkx.classes.graph.Graph
            Subgraph of complete graph
        :param src: str, int, float
            Node identifier
        :param trgt: str, int, float
            Node identifier
        :return:
        """
        edge_weight = nx.shortest_path_length(self.graph, source=src,
                                              target=trgt, weight='weight')

        self.subgraph.add_edge(src, trgt)
        self.subgraph[src][trgt]['weight'] = edge_weight

    def plot_subnetwork(self, text=None, facecol=None, xlim=None, ylim=None,
                        cbar=None, save_path=None):

        fig, ax = plt.subplots()
        if self.wbc_cols is not None:
            nx.draw(self.subgraph, pos=self.coords, node_size=5, ax=ax,
                    node_color=self.wbc_cols, cmap='jet')
            if cbar:
                sm = plt.cm.ScalarMappable(cmap='jet',
                                           norm=plt.Normalize(
                                               vmin=min(self.wbc_cols),
                                               vmax=max(self.wbc_cols)))
                sm._A = []
                plt.colorbar(sm, label='WBC')
        else:
            nx.draw(self.subgraph, pos=self.coords, node_size=5, ax=ax,
                    cmap='jet')

        if text:
            ax.text(s=text['text'], x=text['x'], y=text['y'],
                    transform=ax.transAxes)

        if facecol:
            fig.set_facecolor(facecol)

        if xlim:
            ax.set_xlim(xlim)

        if ylim:
            ax.set_ylim(ylim)

        if save_path:
            plt.savefig(save_path)
        plt.show()

    def set_sub_edge_weights(self):
        """
        Sets the subgraph's edge weights as the Euclidean distance between node
        coordinates
        :return:
        """
        for e in list(self.subgraph.edges):
            self.subgraph[e[0]][e[1]]['weight'] = np.linalg.norm(
                np.array(self.coords[e[0]]) -
                np.array(self.coords[e[1]]))

    def set_wbc(self):
        """
        Calculates the betweenness centrality of the graph and sets colour
        values for later plotting
        :return:
        """
        self.wbc = nx.betweenness_centrality(self.subgraph, normalized=True,
                                             endpoints=True)
        self.wbc_cols = [self.wbc.get(node) for node in self.subgraph.nodes]

    def set_cc(self):
        """
        Calculates the closeness centrality of the graph and sets colour
        values for later plotting
        :return:
        """
        self.cc = nx.closeness_centrality(self.subgraph, wf_improved=True,
                                          distance='weight')
        self.cc_cols = [self.cc.get(node) for node in self.subgraph.nodes]

    def set_wbc_cc(self):
        """
        Calculates the betweenness and closeness centrality of the graph and
        sets colour values for later plotting
        :return:
        :return:
        """

        self.wbc = nx.betweenness_centrality(self.subgraph, normalized=True,
                                             endpoints=True)
        self.wbc_cols = [self.wbc.get(node) for node in self.subgraph.nodes]

        self.cc = nx.closeness_centrality(self.subgraph, wf_improved=True,
                                          distance='weight')
        self.cc_cols = [self.cc.get(node) for node in self.subgraph.nodes]

    def get_xy_wbc(self):
        networkdf = pd.DataFrame.from_dict(self.coords, orient='index')
        networkdf.columns = ['lon', 'lat']

        if self.wbc is None:
            self.set_wbc()

        wbcdf = pd.DataFrame.from_dict(self.wbc, orient='index',
                                       columns=['WBC'])

        degdf = pd.DataFrame.from_dict(dict(self.subgraph.degree),
                                       orient='index', columns=['degrees'])

        wbcdf = wbcdf.join(degdf, how='left')

        if len(np.intersect1d(networkdf.index, wbcdf.index)) == \
                len(networkdf.index):
            networkdf = networkdf.join(other=wbcdf, how='left')

            return networkdf

        else:
            raise KeyError('networkdf and wbcdf index not the same')

    def get_xy_wbc_cc(self):
        networkdf = pd.DataFrame.from_dict(self.coords, orient='index')
        networkdf.columns = ['lon', 'lat']

        if self.wbc is None:
            self.set_wbc()

        wbcdf = pd.DataFrame.from_dict(self.wbc, orient='index')
        wbcdf.columns = ['WBC']

        ccdf = pd.DataFrame.from_dict(self.cc, orient='index')
        ccdf.columns = ['CC']

        degdf = pd.DataFrame.from_dict(dict(self.subgraph.degree),
                                       orient='index', columns=['degrees'])

        wbcdf = wbcdf.join(degdf, how='left')

        wbcdf = wbcdf.join(ccdf, how='left')

        if (len(np.intersect1d(networkdf.index, wbcdf.index)) ==
                len(networkdf.index)):
            networkdf = networkdf.join(other=wbcdf, how='left')

            return networkdf

        else:
            raise KeyError('networkdf and wbcdf index not the same')
