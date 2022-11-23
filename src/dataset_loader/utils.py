import math
import random
import torch
from loguru import logger

def aggregate_by_time(raw_edges, time_aggregation):
	"""
	Parameters
	----------
	raw_edges
		list of the edges
	time_aggregation
		time step size in seconds

	Returns
	-------
	list of edges in the single time step
	"""
	times = [int(re['time'] // time_aggregation) for re in raw_edges]

	min_time, max_time = min(times), max(times)
	times = [t - min_time for t in times]
	time_steps = max_time - min_time + 1
	seperated_edges = [[] for _ in range(time_steps)]

	for i, edge in enumerate(raw_edges):
		t = times[i]
		seperated_edges[t].append(edge)

	return seperated_edges

# this function guarantees unique edges
# use latest edge in single time step
def generate_undirected_edges(directed_edges):
	"""
	Parameters
	----------
	directed_edges
		directional edges

	Returns
	-------
	undirectional edges
	"""

	edges_dict = {}
	for edge in directed_edges:
		e = (edge['from'], edge['to'])
		wt = edges_dict.get(e)

		if wt:
			_, t = wt
			if edge['time'] > t:
				edges_dict[e] = edge['weight'], edge['time']
		else:
			edges_dict[e] = edge['weight'], edge['time']

	undirected_edges = []
	for edge in edges_dict:
		if (edge[1], edge[0]) in edges_dict:
			if edge[0] > edge[1]:
				continue

			weight1, time1 = edges_dict[edge]
			weight2, time2 = edges_dict[(edge[1], edge[0])]

			weight = weight1 if time1 >= time2 else weight2

			undirected_edges.append({
				'from': edge[0],
				'to': edge[1],
				'weight': weight,
				'original': time1 >= time2
			})
			undirected_edges.append({
				'from': edge[1],
				'to': edge[0],
				'weight': weight,
				'original': time1 < time2
			})

		else:
			weight, _ = edges_dict[edge]
			undirected_edges.append({
				'from': edge[0],
				'to': edge[1],
				'weight': weight,
				'original': True
			})
			undirected_edges.append({
				'from': edge[1],
				'to': edge[0],
				'weight': weight,
				'original': False
			})

	return undirected_edges

def negative_sampling(adj_list):
	"""
	Parameters
	----------
	adj_list
		directed adjacency list on single time step

	Returns
	-------
	sampled undirected non_edge list from given adjacency list
	"""
	num_nodes = len(adj_list)
	non_edges = []

	for node, neighbors in enumerate(adj_list):
		neg_dests = []
		while len(neg_dests) < len(neighbors):
			dest = random.randint(0, num_nodes-1)
			if dest in [node] + neighbors + neg_dests:
				continue
			neg_dests.append(dest)
			non_edges.append([node, dest])

	return non_edges + [non_edge[::-1] for non_edge in non_edges]
