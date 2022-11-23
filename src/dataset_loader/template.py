import math
import torch, dgl
import os, shutil, wget
from utils import fix_seed
from dataset_loader import utils
from loguru import logger

class DatasetTemplate(dgl.data.DGLDataset):
	"""
	Wrapper class for datasets
	Its length is a number of time steps.
	All dataset uses this class to simplify dataset process.
	"""
	def __init__(
		self,
		name,
		url,
		input_dim,
		train_ratio,
		val_ratio,
		device,
		data_dir,
		seed=None,
		time_aggregation=None,
		**kwargs
	):
		"""
		Parameters
		----------
		name
			dataset name
		url
			url of the dataset
		input_dim
			input feature dimension
		train_ratio
			train split ratio
		val_ratio
			validation split ratio
		device
			device name
		data_dir
			path of the data directory
		seed
			random seed
		time_aggregation
			time step size in seconds
		kwargs
			additional arguments for dgl dataset(ex: verbose or force_reload)
		"""
		fix_seed(seed)

		self.input_dim = input_dim
		self.time_aggregation = time_aggregation
		self.train_ratio = train_ratio
		self.val_ratio = val_ratio
		self.device = device
		self.raw_file_name = os.path.basename(url)

		self.adj_lists = None
		self.non_edges = None
		self.num_label = None
		self.ndata = None

		test_ratio = 1 - train_ratio - val_ratio
		assert 0 < train_ratio and train_ratio < 1
		assert 0 < val_ratio and val_ratio < 1
		assert 0 < test_ratio and test_ratio < 1

		data_dir = '{}/{}'.format(data_dir, name)
		cache_dir = '{}/{}-{}-{}-{}-{}'.format(
			data_dir,
			input_dim,
			train_ratio,
			val_ratio,
			seed,
			time_aggregation
		)
		os.makedirs(cache_dir, exist_ok=True)

		super().__init__(name=name, url=url, raw_dir=data_dir, save_dir=cache_dir, **kwargs)

	@property
	def raw_path(self):
		return self.raw_dir

	@property
	def save_path(self):
		return self.save_dir

	def has_cache(self):
		return os.access(self.save_path+'/graphs.pt', os.R_OK)

	def _download(self):
		if not os.path.exists('{}/{}'.format(self.raw_path, self.raw_file_name)):
			self.download()

	def download(self):
		temp = '{}/.{}'.format(self.raw_path, self.raw_file_name)
		dst = '{}/{}'.format(self.raw_path, self.raw_file_name)
		print('downloading {}...'.format(self.name))
		wget.download(self.url, out=temp)
		print('\ndone')
		shutil.move(temp, dst)

	def process(self):
		"""Should be overwritten for each dataset."""
		raise NotImplementedError

	def save(self):
		torch.save(
			{
				'graphs': self.graphs,
				'adj_lists': self.adj_lists,
				'non_edges': self.non_edges,
				'num_label': self.num_label,
				'ndata': self.ndata
			},
			self.save_path+'/.graphs.pt'
		)
		shutil.move(self.save_path+'/.graphs.pt', self.save_path+'/graphs.pt')

	def load(self):
		cache = torch.load(self.save_path+'/graphs.pt', self.device)

		self.graphs = cache['graphs']
		self.num_nodes = self[0].num_nodes()
		for graph in self.graphs:
			assert graph.num_nodes() == self.num_nodes
		self.num_edges = sum([graph.num_edges() for graph in self.graphs])

		self.adj_lists = cache['adj_lists']
		self.non_edges = cache['non_edges']
		self.num_label = cache['num_label']
		self.ndata = cache['ndata']

		logger.info(self)

	def __getitem__(self, idx):
		return self.graphs[idx]

	def __len__(self):
		return len(self.graphs)

	def __str__(self):
		expr = 'TemporalGraph({}, num_nodes={}, num_edges={}, num_timesteps={}'.format(
			self.name, self.num_nodes, self.num_edges, len(self)
		)
		if self.num_label is not None:
			expr += ', num_label={}'.format(self.num_label)
		return expr + ')'
