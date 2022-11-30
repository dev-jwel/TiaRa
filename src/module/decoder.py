import torch
from loguru import logger

class NodeDecoder(torch.nn.Module):
    def __init__(self, emb_dim, decoder_dim, num_label, device):
        """
        Parameters
        ----------
        emb_dim
            dimension size of the model's output embedding feature
        decoder_dim
            hidden classifier dimension
        num_label
            number of label
        device
            device name
        """
        super().__init__()

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, decoder_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(decoder_dim, num_label)
        ).to(device)

    def forward(self, embedding):
        """
        Parameters
        ----------
        embedding
            embedding feature

        Returns
        -------
        Logits respect to given embedding and indices
        """
        return self.mlp(embedding)

class PairDecoder(torch.nn.Module):
    def __init__(self, emb_dim, decoder_dim, device):
        """
        Parameters
        ----------
        emb_dim
            dimension size of the model's output embedding feature
        decoder_dim
            hidden classifier dimension
        device
            device name
        """
        super().__init__()

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_dim, decoder_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(decoder_dim, 2)
        ).to(device)

    def forward(self, embedding, indices):
        """
        Parameters
        ----------
        embedding
            embedding feature
        indices
            list of the edge which is index of the node

        Returns
        -------
        Logits respect to given embedding and indices
        """
        ret = []
        assert embedding.shape[0] == len(indices)
        for emb, idx in zip(torch.split(embedding, 1, dim=0), indices):
            emb = emb.squeeze()
            from_emb = emb[idx[:, 0]]
            to_emb = emb[idx[:, 1]]
            ret.append(self.mlp(torch.hstack([from_emb, to_emb])))
        return ret
