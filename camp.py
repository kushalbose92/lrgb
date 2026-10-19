import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import math
import os 
import numpy as np
import dgl
import networkx as nx
import sys

from torch_geometric.nn import GCNConv, GINConv, global_mean_pool
# from ginconv import GINConv
from torch_geometric.utils import degree, to_networkx
# from torch.nn.parameter import Parameter
import torch_geometric.transforms as T 
from torch_geometric.data import Data

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import scipy.sparse as sp


'''
centrality --- node degree / betweeness / closeness / pagrank / load
nodes with higher degree are highly likely to cause oversquashing. 
nodes with higher centrality should communicate in the earlier layers than the nodes with lower centrality 
'''

'''
node_counts --- number of nodes in each graph in the batch
ptr --- pointer to the start and end of each graph in the batch
centrality --- list of centrality values for all the nodes in the batch
layer --- current layer number 
x --- node features for all the nodes in the batch
prev_embed --- node features from the previous layer for all the nodes in the batch
'''
def camp_update(node_counts, batch_frac, ptr, centrality, layer, x, prev_embed, num_layers):
    device = x.device
    batch_mask = mask_generation_centrality(node_counts, batch_frac, ptr, centrality, layer, num_layers, device)
    x, prev_embed = async_update(batch_mask, x, prev_embed)
    return x, prev_embed


def async_update(batch_mask, x, prev_embed):
    index_mask = np.hstack(batch_mask)
    x, prev_embed = rand_async_msg_passing(x, prev_embed, index_mask)
    return x, prev_embed

# Asynchronous Message Passing for CAMP
def rand_async_msg_passing(x, prev_embed, batch_mask):
    update_mask = torch.from_numpy(batch_mask).to(x.device)
    update_mask = update_mask.unsqueeze(1)
    x = torch.where(update_mask == 1, x, prev_embed)
    prev_embed = x
    return x, prev_embed

'''
Generate a mask based on centrality for a single graph in the batch.
-- centrality: centrality values for the nodes in the graph
-- num_nodes: number of nodes in the graph
-- layer: current layer number
'''
def mask_based_on_centrality(centrality, batch_frac, num_nodes, layer, num_layers, device):
    sorted_indices = centrality
    chunk_size = math.floor(num_nodes / num_layers)
    p = 0.5
    st = layer*chunk_size - math.floor((p*chunk_size))
    en = (layer+1)*chunk_size + math.floor((p*chunk_size))
    if st < 0:
        st = 0
    if en > num_nodes:
        en = num_nodes
    if layer == num_layers - 1:
        candidate_nodes = sorted_indices[st:en]
    else:
        candidate_nodes = sorted_indices[st:en]
    sample_size = min(math.floor(batch_frac*len(candidate_nodes)), len(candidate_nodes))
    rand_indices = torch.randperm(candidate_nodes.shape[0])
    selected_nodes = candidate_nodes[rand_indices[:sample_size]]
    node_idx = torch.zeros(num_nodes).to(device)
    node_idx[selected_nodes] = 1
    node_idx = node_idx.detach().cpu().numpy()
    return node_idx


'''
This function computes the batch of masks for all the graphs 
in the batch based on their centrality values. For each graph the function "mask_based_on_centrality()"
is called. 
-- node_counts: list of number of nodes in each graph in the batch
-- ptr: pointer to the start and end of each graph in the batch
-- centrality: list of centrality values for all the nodes in the batch
-- layer: current layer number
return a batch of masks for all the graphs in the batch. 
Each mask is a binary vector of length equal to the number of nodes in the graph, 
where 1 indicates that the node is selected for message passing 
and 0 indicates that it is not selected.
'''
def mask_generation_centrality(node_counts, batch_frac, ptr, centrality, layer, num_layers, device):
    batch_mask = []
    num_graphs = node_counts.shape[0]
    for g in range(num_graphs):
        start, end = int(ptr[g]), int(ptr[g+1])
        centrality_values = centrality[start:end]
        graph_wise_mask = mask_based_on_centrality(centrality_values, batch_frac, node_counts[g], layer, num_layers, device)
        batch_mask.append(graph_wise_mask)
    return batch_mask
            

