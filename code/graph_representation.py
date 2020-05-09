#!/usr/bin/env python
# coding: utf-8

# # Libs

# In[1]:


import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# # Create Grid Network

# In[2]:

def create_grid_network(size, show_plot=False):
    '''
    Create a square grid network 
    size: network shape --> (size, size)
    '''

    G = nx.DiGraph()
    
    grid_size = (size,size)

    node_list = []
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            node_list.append((i,j))
    
    # add nodes
    for node in node_list:
        G.add_node(node)

    # add edges
    max_idx = size - 1
    for node in node_list:
        row, col = node
        if row > 0:
            G.add_edge(node,(row - 1,col))
            G.add_edge((row - 1,col),node)
        if col > 0:
            G.add_edge(node,(row,col - 1))
            G.add_edge((row,col - 1),node)
        if row < max_idx:
            G.add_edge(node,(row + 1,col))
            G.add_edge((row + 1,col),node)
        if col < max_idx:
            G.add_edge(node,(row,col + 1))
            G.add_edge((row,col + 1),node)

    # network vis
    if show_plot:
        nx.draw(G,node_size=100,node_color='b',edge_color='k')
        plt.show()
    
    return G


# # Transform into Movement Network


def cal_direction(edge_idx):
    '''
    edge_idx: [row_diff, col_diff]
    # [1,0] --> heading South
    # [-1,0] --> heading North
    # [0,1] --> heading East
    # [0,-1] --> heading West
    '''
    if edge_idx == [1,0]:
        return 'South'
    elif edge_idx == [-1,0]:
        return 'North'
    elif edge_idx == [0,1]:
        return 'East'
    elif edge_idx == [0,-1]:
        return 'West'
    else:
        print('Wrong edge index!')


# In[8]:

def movement_network_transform(G, show_plot=False):

    # margin cases
    size = int(np.sqrt(len(G.nodes)))
    max_idx = size - 1
    margin_edges = []
    for row in range(max_idx + 1):
        margin_edges.append(((row, -1),(row, 0)))
        margin_edges.append(((row,max_idx + 1),(row,max_idx)))
    for col in range(max_idx + 1):
        margin_edges.append(((-1,col),(0,col)))
        margin_edges.append(((max_idx + 1,col),(max_idx,col)))

    # add nodes
    G_mv = nx.DiGraph()
    for new_node in list(G.edges) + margin_edges:
        for mv in [-1,0,1]:
            G_mv.add_node(tuple(list(new_node) + [mv]))

    # direction flow functions
    fn_row_minus = lambda a:(a[0] - 1,a[1])
    fn_row_plus = lambda a:(a[0] + 1,a[1])
    fn_col_minus = lambda a:(a[0],a[1] - 1)
    fn_col_plus = lambda a:(a[0],a[1] + 1)


    # add edges
    for node in G_mv.nodes:
        
        direction = cal_direction([node[1][i] - node[0][i] for i in range(2)])
        
        # South, -1 --> col + 1; South, 0 --> row + 1; South, 1 --> col - 1
        # North, -1 --> col - 1; North, 0 --> row - 1; North, 1 --> col + 1
        # East, -1 --> row - 1; East, 0 --> col + 1; East, 1 --> row + 1
        # West, -1 --> row + 1; West, 0 --> col - 1; West, 1 --> row - 1

        from_node = node[1]
        if (direction,node[-1]) in [('South',0),('East',1),('West',-1)]:
            to_node = fn_row_plus(node[1])
        elif (direction,node[-1]) in [('North',0),('East',-1),('West',1)]:
            to_node = fn_row_minus(node[1])
        elif (direction,node[-1]) in [('South',-1),('North',1),('East',0)]:
            to_node = fn_col_plus(node[1])
        elif (direction,node[-1]) in [('South',1),('North',-1),('West',0)]:
            to_node = fn_col_minus(node[1])

        if to_node not in G.nodes:
            continue
        else:
            connecting_node = []
            for mv in [-1,0,1]:
                trial_node = tuple([from_node, to_node] + [mv])
                if trial_node in G_mv.nodes:
                    connecting_node.append(trial_node)
            for i in range(len(connecting_node)):
                G_mv.add_edge(node,connecting_node[i],weight=0.1*np.random.rand())

    # remove 0-degree nodes
    degree_dict = dict(G_mv.degree)
    rmv_node = []
    for key in degree_dict.keys():
        if degree_dict[key] == 0:
            rmv_node.append(key)
            G_mv.remove_node(key)

    # network vis
    if show_plot:
        plt.figure(figsize=(8,8))
        nx.draw(G_mv,node_size=30,node_color='r',edge_color='k',pos=nx.kamada_kawai_layout(G_mv))
        plt.show()

    return G_mv



if __name__ == "__main__":
    G = create_grid_network(size=size)
    G_mv = movement_network_transform(G)