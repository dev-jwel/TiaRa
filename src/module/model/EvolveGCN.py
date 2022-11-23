import torch, dgl, torch_geometric_temporal
from utils import normalize_graph
from loguru import logger

class EvolveGCN(torch.nn.Module):
	def __init__(
		self,
		num_nodes,
		input_dim=32,
		hidden_dim=32,
		output_dim=32,
		rnn='LSTM',
		num_layers=3,
		dropout=0,
		dropedge=0,
		renorm_order='sym',
		device='cuda',
		**_kwargs
	):
		"""
		Parameters
		----------
		num_nodes
			number of vertices
		input_dim
			input feature dimension
		output_dim
			output feature dimension
		rnn
			RNN model
		num_layers
			number of layers
		dropout
			dropout ratio
		dropedge
			dropedge ratio
		renorm_order
			normalization order after dropedge
		device
			device name
		"""
		super().__init__()

		assert input_dim == hidden_dim and hidden_dim == output_dim, 'EvolveGCN requires same input and output dimension'
		dim = input_dim
		self.layers = torch.nn.ModuleList()
		self.norms = torch.nn.ModuleList()
		self.dropout =  torch.nn.Dropout(dropout)
		self.dropedge = dgl.DropEdge(dropedge) if dropedge > 0 else None
		self.renorm_order = renorm_order
		self.act = torch.nn.ReLU()

		for _ in range(num_layers):
			if rnn == 'LSTM':
				layer = torch_geometric_temporal.nn.EvolveGCNO(dim, normalize=False, add_self_loops=False)
			elif rnn == 'GRU':
				layer = torch_geometric_temporal.nn.EvolveGCNH(num_nodes, dim, normalize=False, add_self_loops=False)
			else:
				raise NotImplementedError('no such RNN model {}'.format(rnn))

			self.layers.append(layer.to(device))
			self.norms.append(
				torch.nn.BatchNorm1d(dim, device=device)
			)

	def norm(self, X, normfn):
		"""
		Parameters
		----------
		X
			(T, N, F) shape tensor
		normfn
			BatchNorm1d function

		Returns
		-------
		Normalized X
		"""
		return normfn(X.permute(1, 2, 0)).permute(2, 0, 1)

	def forward(self, dataset, start, end):
		"""
		dataset
			temporal dataset
		start
			start time step of temporal dataset
		end
			end time step of temporal dataset

		Returns
		-------
		Embedding tensor
		"""
		input_graphs = dataset.input_graphs[:end]
		if self.training and self.dropedge:
			input_graphs = [
				normalize_graph(
					self.dropedge(graph.remove_self_loop()).add_self_loop(), self.renorm_order
				)
				for graph in input_graphs
			]

		feature = torch.stack([graph.ndata['X'] for graph in dataset[:end]])

		for layer, norm in zip(self.layers, self.norms):
			layer.weight = layer.initial_weight
			feature = torch.stack([
				layer(feature[t, :, :], torch.stack(graph.edges()), edge_weight=graph.edata['w'])
				for t, graph in enumerate(input_graphs)
			])
			feature = self.dropout(self.act(self.norm(feature, norm)))

		return feature[start:, :, :]
