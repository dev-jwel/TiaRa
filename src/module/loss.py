import torch
from loguru import logger

class PairLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fn = torch.nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        """
        Parameters
        ----------
        logits
            output of the model
        labels
            real labels in the dataset

        Returns
        -------
        BCE loss
        """
        logits = torch.concat(logits)
        labels = torch.concat(labels)
        return self.fn(logits, labels).mean()

class NodeLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fn = torch.nn.CrossEntropyLoss()

    def forward(self, logit, label):
        """
        Parameters
        ----------
        logit
            output of the model
        label
            real labels in the dataset

        Returns
        -------
        Cross entropy loss
        """
        return self.fn(logit, label)
