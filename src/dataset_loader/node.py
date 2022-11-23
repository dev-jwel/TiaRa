import numpy as np
import math, torch, dgl
import os, time, dateutil.parser
from .template import DatasetTemplate
from loguru import logger

class NodeDatasetTemplate(DatasetTemplate):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **{**kwargs, **{'time_aggregation': None}})

	def _download(self):
		"""
		This dataset should be downloaded already
		"""
		assert os.path.exists('{}/{}'.format(self.raw_path, self.raw_file_name))

	def process(self, dataset_name):
		"""
		This dataset is processed directly as this is not a raw file.

		Parameters
		----------
		dataset_name
			name of the dataset
		"""
		# load dataset

		data = np.load('{}/{}.npz'.format(self.raw_path, dataset_name))
		adjs = data['adjs']
		feature = data['attmats']
		label = data['labels']

		# check dataset

		assert adjs.shape[1] == adjs.shape[2] == feature.shape[0] == label.shape[0]
		assert adjs.shape[0] == feature.shape[1]
		assert self.input_dim == feature.shape[2], 'input_dim must be {} for this dataset'.format(feature.shape[2])

		# process numpy dataset

		self.num_nodes = feature.shape[0]
		for node in range(self.num_nodes):
			adjs[:, node, node] = 0
		self.num_edges = adjs.sum().item()
		self.num_label = label.shape[1]

		adjs = [adjs[t, :, :] for t in range(adjs.shape[0])]
		feature = [feature[:, t, :] for t in range(feature.shape[1])]
		label = np.argmax(label, axis=1)

		# convert dataset to torch sparse and dgl graph

		adjs = [torch.tensor(adj, dtype=torch.long, device=self.device).to_sparse() for adj in adjs]
		indices = [adj.indices() for adj in adjs]

		self.graphs = [dgl.graph((index[0, :], index[1, :]), num_nodes=self.num_nodes, device=self.device) for index in indices]

		# add some attributes

		feature = [torch.tensor(feat, dtype=torch.float, device=self.device) for feat in feature]
		for graph, feat in zip(self.graphs, feature):
			graph.ndata['X'] = feat

		# split nodes

		train_val_boundary = math.floor(self.num_nodes * self.train_ratio)
		val_test_boundary = math.floor(self.num_nodes * (self.train_ratio + self.val_ratio))
		perm = torch.randperm(self.num_nodes, device=self.device)

		label = torch.tensor(label, dtype=torch.long, device=self.device)
		train = torch.full((self.num_nodes,), False, device=self.device)
		train[perm[:train_val_boundary]] = True
		val = torch.full((self.num_nodes,), False, device=self.device)
		val[perm[train_val_boundary:val_test_boundary]] = True
		test = torch.full((self.num_nodes,), False, device=self.device)
		test[perm[val_test_boundary:]] = True

		self.ndata = {
			'label': label,
			'train': train,
			'val': val,
			'test': test
		}

		# dataset description

		logger.info(self)

class Brain(NodeDatasetTemplate):
	def __init__(self, *args, **kwargs):
		super().__init__(
			'Brain', 'Brain.npz', *args, **kwargs
		)

	def process(self):
		super().process('Brain')

class Reddit(NodeDatasetTemplate):
	def __init__(self, *args, **kwargs):
		super().__init__(
			'Reddit', 'reddit.npz', *args, **kwargs
		)

	def process(self):
		super().process('reddit')

class DBLP3(NodeDatasetTemplate):
	def __init__(self, *args, **kwargs):
		super().__init__(
			'DBLP3', 'DBLP3.npz', *args, **kwargs
		)

	def process(self):
		super().process('DBLP3')

class DBLP5(NodeDatasetTemplate):
	def __init__(self, *args, **kwargs):
		super().__init__(
			'DBLP5', 'DBLP5.npz', *args, **kwargs
		)

	def process(self):
		super().process('DBLP5')
