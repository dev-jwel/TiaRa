import torch
import torchmetrics
from loguru import logger

class AUCMetric():
	def __init__(self):
		self.softmax = torch.nn.Softmax(dim=1)

	def __call__(self, logits, labels):
		"""
		Parameters
		----------
			logits
				output of the model
			labels
				real labels in the dataset

		Returns
		-------
		AUC value
		"""
		assert len(logits) == len(labels)
		for logit, label in zip(logits, labels):
			assert logit.shape[0] == label.shape[0]

		probabilities = [self.softmax(logit) for logit in logits]
		probabilities = torch.concat(probabilities)
		labels = torch.concat(labels)

		return torchmetrics.functional.auroc(probabilities, labels, num_classes=2)

class F1Metric():
	def __init__(self, num_label):
		self.num_label = num_label

	def __call__(self, logits, labels):
		"""
		Parameters
		----------
			logits
				output of the model
			labels
				real labels in the dataset

		Returns
		-------
		F1 value
		"""
		assert logits.shape[0] == labels.shape[0]

		predicts = torch.argmax(logits, 1)
		return torchmetrics.functional.f1_score(
			predicts, labels, num_classes=self.num_label, average='macro'
		)
