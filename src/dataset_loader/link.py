import torch, dgl
import csv, math, os, shutil, wget
import gzip, zipfile
import time, dateutil.parser
from loguru import logger
from .template import DatasetTemplate
from . import utils
from tqdm import tqdm

class LinkDatasetTemplate(DatasetTemplate):
	"""
	Dataset wrapper class for temporal link prediction.
	Each dataset has a different preprocess but same process after preprocess.
	"""
	def preprocess(self):
		"""
		Should be overwritten,
		This method should generate num_nodes, raw_edges, node_labels(optional).
		"""
		raise NotImplementedError

	def process(self):
		self.preprocess()
		self.num_edges = len(self.raw_edges)
		os.makedirs(self.save_path, exist_ok=True)

		# generate temporal graphs

		temporal_edges = utils.aggregate_by_time(self.raw_edges, self.time_aggregation)
		undirected_raw_edges = [
			utils.generate_undirected_edges(edges) for edges in temporal_edges
		]
		undirected_edges = [
			torch.tensor(
				[[e['from'], e['to']] for e in edge], dtype=torch.long, device=self.device
			).reshape(-1, 2)
			for edge in undirected_raw_edges
		]
		self.graphs = [
			dgl.graph(data=(edge[:, 0], edge[:, 1]), num_nodes=self.num_nodes, device=self.device)
			for edge in undirected_edges
		]

		for graph in self.graphs:
			graph.ndata['X'] = torch.randn(self.num_nodes, self.input_dim, device=self.device)

		# generate adjacency lists and sample negative edges for validation

		temporal_non_edges = []
		adj_lists = []

		for edges in undirected_raw_edges:
			adj_list = [[] for _ in range(self.num_nodes)]
			for edge in edges:
				if not edge['original']:
					continue
				adj_list[edge['from']].append(edge['to'])
			adj_lists.append(adj_list)
			non_edges = utils.negative_sampling(adj_list)
			non_edges = torch.tensor(non_edges, dtype=torch.long, device=self.device).reshape(-1, 2)
			temporal_non_edges.append(non_edges)

		self.adj_lists = adj_lists
		self.non_edges = temporal_non_edges

		logger.info(self)

	def get_range_by_split(self, split):
		"""
		Parameters
		----------
		split
			dataset split

		Returns
		-------
		begin, end of the given split
		"""
		time_steps = len(self)

		if split == 'train':
			begin_ratio = 0
			end_ratio = self.train_ratio
		elif split == 'val':
			begin_ratio = self.train_ratio
			end_ratio = self.train_ratio + self.val_ratio
		elif split == 'test':
			begin_ratio = self.train_ratio + self.val_ratio
			end_ratio = 1
		else:
			raise NotImplementedError('no such split {}'.format(split))
		return math.floor(time_steps*begin_ratio), math.floor(time_steps*end_ratio)

class BitcoinAlpha(LinkDatasetTemplate):
	def __init__(self, *args, **kwargs):
		super().__init__(
			'BitcoinAlpha', 'https://cs.stanford.edu/~srijan/rev2/rev2data.zip', *args, **kwargs
		)

	def preprocess(self):
		network_filename = 'rev2-data/alpha/alpha_network.csv'.format()
		with zipfile.ZipFile(self.raw_path+'/rev2data.zip', 'r') as zip_ref:
			zip_ref.extract(network_filename, self.raw_path)
		network_filename = '{}/{}'.format(self.raw_path, network_filename)

		with open(network_filename, 'r') as file:
			reader = csv.reader(file)

			num_unique_node = 0
			node_mapper = {}
			raw_edges = []

			for row in reader:
				if row[0] == row[1]:
					continue
				if not row[0] in node_mapper:
					node_mapper[row[0]] = num_unique_node
					num_unique_node += 1
				if not row[1] in node_mapper:
					node_mapper[row[1]] = num_unique_node
					num_unique_node += 1

				raw_edges.append({
					'from': node_mapper[row[0]],
					'to': node_mapper[row[1]],
					'weight': float(row[2]),
					'time': float(row[3])
				})

		self.num_nodes = num_unique_node
		self.raw_edges = raw_edges

class WikiElec(LinkDatasetTemplate):
	def __init__(self, *args, **kwargs):
		super().__init__(
			'WikiElec', 'https://snap.stanford.edu/data/wikiElec.ElecBs3.txt.gz', *args, **kwargs
		)

	def preprocess(self):
		with gzip.open(self.raw_path+'/wikiElec.ElecBs3.txt.gz', 'rt', encoding='latin-1') as file:
			upper_bound = 2e+9

			num_unique_node = 0
			node_mapper = {}
			raw_edges = []

			current_format = 'V'

			for line in file.readlines():
				if line == '':
					continue
				if line[0] == '#':
					continue

				if line[0] == 'E':
					assert current_format == 'V'
					current_format = 'E'

				if line[0] == 'T':
					assert current_format == 'E'
					current_format = 'T'

				if line[0] == 'U':
					assert current_format == 'T'
					current_format = 'U'

					line = line.split('\t')
					user_for_election = line[1]

					if not user_for_election in node_mapper:
						node_mapper[user_for_election] = num_unique_node
						num_unique_node += 1

				if line[0] == 'N':
					assert current_format == 'U'
					current_format = 'N'

				if line[0] == 'V':
					assert current_format == 'N' or current_format == 'V'
					current_format = 'V'

					line = line.split('\t')

					sign = float(line[1])
					user_id = line[2]
					timestamp = line[3]

					timestamp = time.mktime(dateutil.parser.parse(timestamp).timetuple())

					if sign == 0:
						continue

					if timestamp > upper_bound:
						continue

					if user_id == user_for_election:
						continue

					if not user_id in node_mapper:
						node_mapper[user_id] = num_unique_node
						num_unique_node += 1

					raw_edges.append({
						'from': node_mapper[user_id],
						'to': node_mapper[user_for_election],
						'weight': sign,
						'time': timestamp
					})

			assert current_format == 'V'

		self.num_nodes = num_unique_node
		self.raw_edges = raw_edges


class RedditBody(LinkDatasetTemplate):
	def __init__(self, *args, **kwargs):
		super().__init__(
			'RedditBody', 'https://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv', *args, **kwargs
		)

	def preprocess(self):
		filename = 'soc-redditHyperlinks-body.tsv'
		with open('{}/{}'.format(self.raw_path, filename), 'r') as file:
			reader = csv.DictReader(file, delimiter='\t')

			num_unique_node = 0
			node_mapper = {}
			raw_edges = []

			for row in reader:
				if row['SOURCE_SUBREDDIT'] == row['TARGET_SUBREDDIT']:
					continue
				if not row['SOURCE_SUBREDDIT'] in node_mapper:
					node_mapper[row['SOURCE_SUBREDDIT']] = num_unique_node
					num_unique_node += 1
				if not row['TARGET_SUBREDDIT'] in node_mapper:
					node_mapper[row['TARGET_SUBREDDIT']] = num_unique_node
					num_unique_node += 1

				raw_edges.append({
					'from': node_mapper[row['SOURCE_SUBREDDIT']],
					'to': node_mapper[row['TARGET_SUBREDDIT']],
					'weight': float(row['LINK_SENTIMENT']),
					'time': time.mktime(dateutil.parser.parse(row['TIMESTAMP']).timetuple())
				})

		self.num_nodes = num_unique_node
		self.raw_edges = raw_edges
