import torch, dgl
from torch.nn import Parameter
from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import glorot, zeros
from utils import normalize_graph
from loguru import logger

class GCRN(torch.nn.Module):
	def __init__(
		self,
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
		input_dim
			input feature dimension
		hidden_dim
			hidden feature dimension
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

		dimensions = [input_dim] + (num_layers-1)*[hidden_dim] + [output_dim]
		self.layers = torch.nn.ModuleList()
		self.norms = torch.nn.ModuleList()
		self.dropout =  torch.nn.Dropout(dropout)
		self.dropedge = dgl.DropEdge(dropedge) if dropedge > 0 else None
		self.renorm_order = renorm_order
		self.rnn = rnn
		self.act = torch.nn.ReLU()

		if rnn == 'LSTM':
			GNN = GConvLSTM
		elif rnn == 'GRU':
			GNN = GConvGRU
		else:
			raise NotImplementedError('no such RNN model {}'.format(rnn))

		for layer in range(len(dimensions)-1):
			self.layers.append(
				GNN(dimensions[layer], dimensions[layer+1], normalize=False).to(device)
			)
			self.norms.append(
				torch.nn.BatchNorm1d(dimensions[layer+1], device=device)
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
		Parameters
		----------
		dataset
			temporal graph dataset
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
			H = None
			C = None
			Hs = []

			for t, graph in enumerate(input_graphs):
				feat = feature[t, :, :]
				indices = torch.stack(graph.edges())

				if self.rnn == 'LSTM':
					H, C = layer(feat, indices, H=H, C=C, edge_weight=graph.edata['w'])
				elif self.rnn == 'GRU':
					H = layer(feat, indices, H=H, edge_weight=graph.edata['w'])
				else:
					raise NotImplementedError('no such RNN model {}'.format(self.rnn))

				Hs.append(H)

			feature = self.dropout(self.act(self.norm(torch.stack(Hs), norm)))

		return feature[start:, :, :]

# Original GCRN models used Chebyshev GCN, but we modified it to use Kipf's GCN due to memory limitation.
# Below models are based on the source code from pytorch geometric temporal
# See https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html

class GConvGRU(torch.nn.Module):
	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		normalize: bool = True,
		bias: bool = True,
	):
		super(GConvGRU, self).__init__()

		self.in_channels = in_channels
		self.out_channels = out_channels
		self.normalize = normalize
		self.bias = bias
		self._create_parameters_and_layers()

	def _create_update_gate_parameters_and_layers(self):

		self.conv_x_z = GCNConv(
			in_channels=self.in_channels,
			out_channels=self.out_channels,
			normalize=self.normalize,
			bias=self.bias,
		)

		self.conv_h_z = GCNConv(
			in_channels=self.out_channels,
			out_channels=self.out_channels,
			normalize=self.normalize,
			bias=self.bias,
		)

	def _create_reset_gate_parameters_and_layers(self):

		self.conv_x_r = GCNConv(
			in_channels=self.in_channels,
			out_channels=self.out_channels,
			normalize=self.normalize,
			bias=self.bias,
		)

		self.conv_h_r = GCNConv(
			in_channels=self.out_channels,
			out_channels=self.out_channels,
			normalize=self.normalize,
			bias=self.bias,
		)

	def _create_candidate_state_parameters_and_layers(self):

		self.conv_x_h = GCNConv(
			in_channels=self.in_channels,
			out_channels=self.out_channels,
			normalize=self.normalize,
			bias=self.bias,
		)

		self.conv_h_h = GCNConv(
			in_channels=self.out_channels,
			out_channels=self.out_channels,
			normalize=self.normalize,
			bias=self.bias,
		)

	def _create_parameters_and_layers(self):
		self._create_update_gate_parameters_and_layers()
		self._create_reset_gate_parameters_and_layers()
		self._create_candidate_state_parameters_and_layers()

	def _set_hidden_state(self, X, H):
		if H is None:
			H = torch.zeros(X.shape[0], self.out_channels).to(X.device)
		return H

	def _calculate_update_gate(self, X, edge_index, edge_weight, H):
		Z = self.conv_x_z(X, edge_index, edge_weight)
		Z = Z + self.conv_h_z(H, edge_index, edge_weight)
		Z = torch.sigmoid(Z)
		return Z

	def _calculate_reset_gate(self, X, edge_index, edge_weight, H):
		R = self.conv_x_r(X, edge_index, edge_weight)
		R = R + self.conv_h_r(H, edge_index, edge_weight)
		R = torch.sigmoid(R)
		return R

	def _calculate_candidate_state(self, X, edge_index, edge_weight, H, R):
		H_tilde = self.conv_x_h(X, edge_index, edge_weight)
		H_tilde = H_tilde + self.conv_h_h(H * R, edge_index, edge_weight)
		H_tilde = torch.tanh(H_tilde)
		return H_tilde

	def _calculate_hidden_state(self, Z, H, H_tilde):
		H = Z * H + (1 - Z) * H_tilde
		return H

	def forward(
		self,
		X: torch.FloatTensor,
		edge_index: torch.LongTensor,
		edge_weight: torch.FloatTensor = None,
		H: torch.FloatTensor = None,
	) -> torch.FloatTensor:
		"""
		Making a forward pass. If edge weights are not present the forward pass
		defaults to an unweighted graph. If the hidden state matrix is not present
		when the forward pass is called it is initialized with zeros.

		Arg types:
			* **X** *(PyTorch Float Tensor)* - Node features.
			* **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
			* **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
			* **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.
			* **lambda_max** *(PyTorch Tensor, optional but mandatory if normalization is not sym)* - Largest eigenvalue of Laplacian.


		Return types:
			* **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
		"""
		H = self._set_hidden_state(X, H)
		Z = self._calculate_update_gate(X, edge_index, edge_weight, H)
		R = self._calculate_reset_gate(X, edge_index, edge_weight, H)
		H_tilde = self._calculate_candidate_state(X, edge_index, edge_weight, H, R)
		H = self._calculate_hidden_state(Z, H, H_tilde)
		return H

class GConvLSTM(torch.nn.Module):
	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		normalize: bool = True,
		bias: bool = True,
	):
		super(GConvLSTM, self).__init__()

		self.in_channels = in_channels
		self.out_channels = out_channels
		self.normalize = normalize
		self.bias = bias
		self._create_parameters_and_layers()
		self._set_parameters()

	def _create_input_gate_parameters_and_layers(self):

		self.conv_x_i = GCNConv(
			in_channels=self.in_channels,
			out_channels=self.out_channels,
			normalize=self.normalize,
			bias=self.bias,
		)

		self.conv_h_i = GCNConv(
			in_channels=self.out_channels,
			out_channels=self.out_channels,
			normalize=self.normalize,
			bias=self.bias,
		)

		self.w_c_i = Parameter(torch.Tensor(1, self.out_channels))
		self.b_i = Parameter(torch.Tensor(1, self.out_channels))

	def _create_forget_gate_parameters_and_layers(self):

		self.conv_x_f = GCNConv(
			in_channels=self.in_channels,
			out_channels=self.out_channels,
			normalize=self.normalize,
			bias=self.bias,
		)

		self.conv_h_f = GCNConv(
			in_channels=self.out_channels,
			out_channels=self.out_channels,
			normalize=self.normalize,
			bias=self.bias,
		)

		self.w_c_f = Parameter(torch.Tensor(1, self.out_channels))
		self.b_f = Parameter(torch.Tensor(1, self.out_channels))

	def _create_cell_state_parameters_and_layers(self):

		self.conv_x_c = GCNConv(
			in_channels=self.in_channels,
			out_channels=self.out_channels,
			normalize=self.normalize,
			bias=self.bias,
		)

		self.conv_h_c = GCNConv(
			in_channels=self.out_channels,
			out_channels=self.out_channels,
			normalize=self.normalize,
			bias=self.bias,
		)

		self.b_c = Parameter(torch.Tensor(1, self.out_channels))

	def _create_output_gate_parameters_and_layers(self):

		self.conv_x_o = GCNConv(
			in_channels=self.in_channels,
			out_channels=self.out_channels,
			normalize=self.normalize,
			bias=self.bias,
		)

		self.conv_h_o = GCNConv(
			in_channels=self.out_channels,
			out_channels=self.out_channels,
			normalize=self.normalize,
			bias=self.bias,
		)

		self.w_c_o = Parameter(torch.Tensor(1, self.out_channels))
		self.b_o = Parameter(torch.Tensor(1, self.out_channels))

	def _create_parameters_and_layers(self):
		self._create_input_gate_parameters_and_layers()
		self._create_forget_gate_parameters_and_layers()
		self._create_cell_state_parameters_and_layers()
		self._create_output_gate_parameters_and_layers()

	def _set_parameters(self):
		glorot(self.w_c_i)
		glorot(self.w_c_f)
		glorot(self.w_c_o)
		zeros(self.b_i)
		zeros(self.b_f)
		zeros(self.b_c)
		zeros(self.b_o)

	def _set_hidden_state(self, X, H):
		if H is None:
			H = torch.zeros(X.shape[0], self.out_channels).to(X.device)
		return H

	def _set_cell_state(self, X, C):
		if C is None:
			C = torch.zeros(X.shape[0], self.out_channels).to(X.device)
		return C

	def _calculate_input_gate(self, X, edge_index, edge_weight, H, C):
		I = self.conv_x_i(X, edge_index, edge_weight)
		I = I + self.conv_h_i(H, edge_index, edge_weight)
		I = I + (self.w_c_i * C)
		I = I + self.b_i
		I = torch.sigmoid(I)
		return I

	def _calculate_forget_gate(self, X, edge_index, edge_weight, H, C,):
		F = self.conv_x_f(X, edge_index, edge_weight)
		F = F + self.conv_h_f(H, edge_index, edge_weight)
		F = F + (self.w_c_f * C)
		F = F + self.b_f
		F = torch.sigmoid(F)
		return F

	def _calculate_cell_state(self, X, edge_index, edge_weight, H, C, I, F):
		T = self.conv_x_c(X, edge_index, edge_weight)
		T = T + self.conv_h_c(H, edge_index, edge_weight)
		T = T + self.b_c
		T = torch.tanh(T)
		C = F * C + I * T
		return C

	def _calculate_output_gate(self, X, edge_index, edge_weight, H, C,):
		O = self.conv_x_o(X, edge_index, edge_weight)
		O = O + self.conv_h_o(H, edge_index, edge_weight)
		O = O + (self.w_c_o * C)
		O = O + self.b_o
		O = torch.sigmoid(O)
		return O

	def _calculate_hidden_state(self, O, C):
		H = O * torch.tanh(C)
		return H

	def forward(
		self,
		X: torch.FloatTensor,
		edge_index: torch.LongTensor,
		edge_weight: torch.FloatTensor = None,
		H: torch.FloatTensor = None,
		C: torch.FloatTensor = None,
	) -> torch.FloatTensor:
		"""
		Making a forward pass. If edge weights are not present the forward pass
		defaults to an unweighted graph. If the hidden state and cell state
		matrices are not present when the forward pass is called these are
		initialized with zeros.

		Arg types:
			* **X** *(PyTorch Float Tensor)* - Node features.
			* **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
			* **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
			* **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.
			* **C** *(PyTorch Float Tensor, optional)* - Cell state matrix for all nodes.
			* **lambda_max** *(PyTorch Tensor, optional but mandatory if normalization is not sym)* - Largest eigenvalue of Laplacian.

		Return types:
			* **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
			* **C** *(PyTorch Float Tensor)* - Cell state matrix for all nodes.
		"""
		H = self._set_hidden_state(X, H)
		C = self._set_cell_state(X, C)
		I = self._calculate_input_gate(X, edge_index, edge_weight, H, C)
		F = self._calculate_forget_gate(X, edge_index, edge_weight, H, C)
		C = self._calculate_cell_state(X, edge_index, edge_weight, H, C, I, F)
		O = self._calculate_output_gate(X, edge_index, edge_weight, H, C)
		H = self._calculate_hidden_state(O, C)
		return H, C
