import torch, dgl
from tqdm.autonotebook import tqdm
from loguru import logger
import utils
import os, multiprocessing

class Tiara():
	"""
	Approximate augmentation of temporal random walk diffusion

	Parameters
	----------
	adjacencies
		list(or iterator) of dgl dataset
	alpha
		restart probability
	beta
		time travel probability
	eps
		filtering threshold
	K
		number of iteration for inverting H at time t
	symmetric_trick
		method to generate normalized symmetric adjacency matrix
	device
		device name
	dense
		use dense adjacency matrix
	verbose
		print additional information

	Returns
	-------
	list of augmented dgl dataset

	Examples
	--------
	>>> tiara = Tiara(alpha, beta, device=device)
	>>> augmented_graphs = tiara(dataset)
	"""
	def __init__(
		self,
		alpha=0.2,
		beta=0.3,
		eps=1e-3,
		K=100,
		symmetric_trick=True,
		device='cuda',
		dense=False,
		verbose=False
	):
		assert 0 <= alpha and alpha <= 1
		assert 0 <= beta and beta <= 1
		assert 0 <= alpha + beta and alpha + beta <= 1
		assert 0 < eps

		self.alpha = alpha
		self.beta = beta
		self.eps = eps
		self.K = K
		self.symmetric_trick = symmetric_trick
		self.device = device
		self.dense = dense
		self.verbose = verbose

	def __call__(self, dataset):
		N = dataset[0].num_nodes()
		for graph in dataset:
			assert N == graph.num_nodes()

		if self.dense:
			I = torch.eye(N, device=self.device)
		else:
			I = utils.sparse_eye(N, self.device)

		Xt_1 = I
		new_graphs = list()

		for graph in tqdm(dataset, desc='augmentation'):
			At = graph.adj(ctx=graph.device)
			if self.dense:
				At = At.to_dense()
			At = At + I
			At = self.normalize(At, ord='row')

			inv_Ht = self.approx_inv_Ht(At, self.alpha, self.beta, self.K)

			Xt = inv_Ht @ (self.alpha * I + self.beta * Xt_1)
			# when alpha/beta is small, small K can lead to a large approximate error in Xt
			# for considering this case, we normalize X so that the column sum of X is 1
			Xt = self.normalize(Xt, ord='col')

			Xt = self.filter_matrix(Xt, self.eps)
			Xt_1 = Xt

			if self.symmetric_trick:
				Xt = (Xt + Xt.transpose(1, 0)) / 2

			if self.dense:
				A = Xt.to_sparse().transpose(0, 1).coalesce()
			else:
				A = Xt.transpose(1, 0).coalesce()

			if self.symmetric_trick:
				ones = torch.ones(A._nnz(), device=self.device)
				A = torch.sparse_coo_tensor(A.indices(), ones, A.shape, device=self.device)
				A = self.normalize(A, ord='sym', dense=False)
			else:
				A = self.normalize(A, ord='row', dense=False)

			new_graph = utils.weighted_adjacency_to_graph(A)
			new_graphs.append(new_graph)

			if self.verbose:
				logger.info('number of edge in this time step: {}'.format(new_graph.num_edges()))

		return new_graphs

	def row_sum(self, A, dense=None):
		if dense is None:
			dense = self.dense
		if dense:
			return A.sum(dim=1)
		else:
			return torch.sparse.sum(A, dim=1).to_dense()

	def normalize(self, A, ord='row', dense=None):
		if dense is None:
			dense = self.dense

		N = A.shape[0]
		A = A if ord == 'row' else A.transpose(0, 1)

		norm = self.row_sum(A, dense=dense)
		norm[norm<=0] = 1
		if ord == 'sym':
			norm = norm ** 0.5

		if dense:
			inv_D = torch.diag(1 / norm)
		else:
			inv_D = utils.sparse_diag(1 / norm)

		if ord == 'sym':
			nA = inv_D @ A @ inv_D
		else:
			nA = inv_D @ A
		return nA if ord == 'row' else nA.transpose(0, 1)

	def approx_inv_Ht(self, A, alpha, beta, K=10):
		if self.dense:
			I = torch.eye(A.shape[0], device=self.device)
		else:
			I = utils.sparse_eye(A.shape[0], self.device)

		inv_H_k = I
		c = 1.0 - alpha - beta
		cAT = c * A.transpose(0, 1)

		for i in range(K):
			inv_H_k = I + cAT @ inv_H_k

		return inv_H_k

	def filter_matrix(self, X, eps):
		assert eps < 1.0
		if self.dense:
			X[X<eps] = 0.0
		else:
			X = utils.sparse_filter(X, eps)
		return self.normalize(X, ord='col')

class Merge():
	def __init__(self, device):
		self.device = device

	def __call__(self, dataset):
		merged_graphs = [dataset[0]]

		for graph in dataset[1:]:
			merged_graph = dgl.merge([merged_graphs[-1], graph])
			merged_graph = merged_graph.cpu().to_simple().to(self.device)
			del merged_graph.edata['count']
			merged_graphs.append(merged_graph)

		return GCNNorm(self.device)(merged_graphs)

class GCNNorm():
	def __init__(self, device):
		self.device = device

	def __call__(self, dataset):
		normalized = [utils.graph_to_normalized_adjacency(graph) for graph in dataset]
		return [utils.weighted_adjacency_to_graph(adj) for adj in normalized]
