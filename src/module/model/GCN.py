import torch, dgl
from utils import normalize_graph
from loguru import logger

class GCN(torch.nn.Module):
    def __init__(
        self,
        input_dim=32,
        hidden_dim=32,
        output_dim=32,
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
        self.dropout = torch.nn.Dropout(dropout)
        self.dropedge = dgl.DropEdge(dropedge) if dropedge > 0 else None
        self.renorm_order = renorm_order

        for layer in range(len(dimensions)-1):
            self.layers.append(
                dgl.nn.GraphConv(dimensions[layer], dimensions[layer+1], activation=torch.nn.ReLU()).to(device)
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
            feature = [
                layer(graph, feature[t, :, :], edge_weight=graph.edata['w'])
                for t, graph in enumerate(input_graphs)
            ]
            feature = self.dropout(self.norm(torch.stack(feature), norm))

        return feature[start:, :, :]
