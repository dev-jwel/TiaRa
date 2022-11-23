import dgl
import torch
import random
import numpy as np
from copy import deepcopy
from loguru import logger

SEED = None

def fix_seed(seed):
	"""Fix random seed"""
	global SEED
	if SEED is None and seed is not None:
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		dgl.seed(seed)
		SEED = seed

def sparse_eye(N, device):
	arr = torch.arange(N, device=device)
	indices = torch.stack([arr, arr])
	values = torch.ones_like(arr, dtype=torch.float)
	return torch.sparse_coo_tensor(indices, values, (N, N))

def sparse_diag(input):
	N = input.shape[0]
	arr = torch.arange(N, device=input.device)
	indices = torch.stack([arr, arr])
	return torch.sparse_coo_tensor(indices, input, (N, N))

def sparse_filter(input, eps):
	input = input.coalesce()
	indices = input.indices()
	values = input.values()
	idx = torch.where(values > eps, True, False)
	indices = indices[:, idx]
	values = values[idx]
	return torch.sparse_coo_tensor(indices, values)

def graph_to_normalized_adjacency(graph):
	normalized_graph = dgl.GCNNorm()(graph.add_self_loop())
	adj = normalized_graph.adj(ctx=graph.device)
	new_adj = torch.sparse_coo_tensor(
		adj.coalesce().indices(), normalized_graph.edata['w'], tuple(adj.shape)
	)
	return new_adj

def weighted_adjacency_to_graph(adj):
	adj = adj.coalesce()
	indices = adj.indices()
	graph = dgl.graph((indices[0, :], indices[1, :]))
	graph.edata['w'] = adj.values()
	return graph

def normalize_graph(graph, ord='sym'):
	adj = graph.adj(ctx=graph.device).coalesce()

	if ord == 'row':
		norm = torch.sparse.sum(adj, dim=1).to_dense()
		norm[norm<=0] = 1
		inv_D = sparse_diag(1 / norm)
		new_adj = inv_D @ adj
	elif ord == 'col':
		norm = torch.sparse.sum(adj, dim=0).to_dense()
		norm[norm<=0] = 1
		inv_D = sparse_diag(1 / norm)
		new_adj = adj @ inv_D
	elif ord == 'sym':
		norm = torch.sparse.sum(adj, dim=1).to_dense()
		norm[norm<=0] = 1
		inv_D = sparse_diag(norm ** (-0.5))
		new_adj = inv_D @ adj @ inv_D

	new_adj = new_adj.coalesce()
	indices = new_adj.indices()
	new_graph = dgl.graph((indices[0, :], indices[1, :]))
	new_graph.edata['w'] = new_adj.values()
	return new_graph
