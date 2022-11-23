import dgl, torch
from tqdm.autonotebook import tqdm
from loguru import logger
from copy import deepcopy
from itertools import chain
from dataset_loader.link import LinkDatasetTemplate
from dataset_loader.node import NodeDatasetTemplate
from dataset_loader.utils import negative_sampling
import utils

class Trainer():
	def __init__(self, model, decoder, lossfn, dataset, evaluator, augment_method):
		"""
		Parameters
		----------
		model
			graph neural network model
		decoder
			decoder model
		lossfn
			loss function
		dataset
			dataset
		evaluator
			evaluator
		augment_method
			dataset augmentation method
		"""
		self.model = model
		self.decoder = decoder
		self.lossfn = lossfn
		self.dataset = dataset
		self.evaluator = evaluator
		self.dataset.input_graphs = augment_method(dataset)

	def train(
		self,
		epochs,
		lr,
		weight_decay,
		lr_decay,
		early_stopping
	):
		"""
		Parameters
		----------
		epochs
			max epochs
		lr
			learning rate
		weight_decay
			weight decay
		lr_decay
			learning rate decay
		early_stopping
			early stopping patient

		Returns
		-------
		model
			trained model
		decoder
			trained decoder
		history
			traning history
		"""
		# initialize before main loop

		optimizer = torch.optim.Adam(
			chain(
				self.model.parameters(),
				self.decoder.parameters()
			),
			lr=lr, weight_decay=weight_decay
		)
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, lr_decay)

		best_model_state = deepcopy(self.model.state_dict())
		best_decoder_state = deepcopy(self.decoder.state_dict())
		best_val_epoch = -1
		best_val_metric = -float('Inf')
		history = []

		# main loop

		epoch_pbar = tqdm(range(epochs), desc='epoch', position=0)
		for epoch in epoch_pbar:
			# train step

			self.model.train()
			self.decoder.train()

			train_loss, train_metric = self.calc_loss_and_metrics('train', False)

			optimizer.zero_grad()
			train_loss.backward()
			optimizer.step()

			epoch_message = 'train loss: {:7.4f}, train metric: {:7.4}'.format(
				train_loss.item(), train_metric.item()
			)

			# validation and test step

			self.model.eval()
			self.decoder.eval()

			with torch.no_grad():
				val_loss, val_metric = self.calc_loss_and_metrics('val', True)

			epoch_message += ', val loss: {:7.4f}, val metric: {:7.4f}'.format(
				val_loss.item(), val_metric.item()
			)

			with torch.no_grad():
				test_loss, test_metric = self.calc_loss_and_metrics('test', True)

			epoch_message += ', test loss: {:7.4f}, test metric: {:7.4f}'.format(
				test_loss.item(), test_metric.item()
			)

			# lr scheduling, early stopping, tqdm logging

			scheduler.step()

			history.append({
				'train_metric': train_metric.item(),
				'train_loss': train_loss.item(),
				'val_metric': val_metric.item(),
				'val_loss': val_loss.item(),
				'test_metric': test_metric.item(),
				'test_loss': test_loss.item()
			})

			if val_metric > best_val_metric:
				best_val_metric = val_metric
				best_val_epoch = epoch
				best_model_state = deepcopy(self.model.state_dict())
				best_decoder_state = deepcopy(self.decoder.state_dict())

			if epoch - best_val_epoch > early_stopping:
				break

			epoch_pbar.write(epoch_message)

		# trainner main loop has finished, return best model and decoder with training history

		self.model.load_state_dict(best_model_state)
		self.decoder.load_state_dict(best_decoder_state)
		return self.model, self.decoder, history

	def calc_loss_and_metrics(self, split, evaluate):
		"""
		Calculate losses and metrics for given task

		Parameters
		----------
		split
			dataset split
		grad
			whether need to calcudate gradient

		Returns
		-------
		loss and metrics
		"""
		if isinstance(self.dataset, LinkDatasetTemplate):
			return self.link_prediction(split, evaluate)
		elif isinstance(self.dataset, NodeDatasetTemplate):
			return self.node_prediction(split)
		else:
			raise

	def link_prediction(self, split, evaluate):
		"""
		Calculate losses and metrics for link prediction task

		Be careful for the start and end points of the input and output of this task
		This task aims to predict links at time t + 1 given data at time t

		Parameters
		----------
		split
			dataset split
		evalueate
			use stored negative edge sample

		Returns
		-------
		loss and metric
		"""
		start, end = self.dataset.get_range_by_split(split)

		# graphs and edges generates labels
		# but start and end indicates time steps of input features
		if split == 'train':
			input_start, input_end = start, end-1
			output_start, output_end = start+1, end
		else:
			input_start, input_end = start-1, end-1
			output_start, output_end = start, end

		edges = [
			torch.stack(graph.edges(), dim=1).reshape(-1, 2)
			for graph in self.dataset[output_start:output_end]
		]

		if evaluate:
			non_edges = self.dataset.non_edges[output_start:output_end]
		else:
			non_edges = [
				negative_sampling(adj_list)
				for adj_list in self.dataset.adj_lists[output_start:output_end]
			]
			non_edges = [
				torch.tensor(non_edge, dtype=torch.long, device=self.dataset.device).reshape(-1, 2)
				for non_edge in non_edges
			]

		assert len(edges) == len(non_edges)
		for edge, non_edge in zip(edges, non_edges):
			assert edge.shape == non_edge.shape

		pairs = [torch.concat([edge, non_edge]) for edge, non_edge in zip(edges, non_edges)]

		label_kwargs = {'dtype': torch.long, 'device': self.dataset.device}
		labels = [
			torch.concat([
				torch.ones(edge.shape[0], **label_kwargs),
				torch.zeros(non_edge.shape[0], **label_kwargs)
			])
			for edge, non_edge in zip(edges, non_edges)
		]

		embedding = self.model(self.dataset, input_start, input_end)
		logits = self.decoder(embedding, pairs)
		loss = self.lossfn(logits, labels)
		metric = self.evaluator(logits, labels)

		return loss, metric

	def node_prediction(self, split):
		"""
		Calculate losses and metrics for node prediction task
		Only last time step has used for this task

		Parameters
		----------
		split
			dataset split

		Returns
		-------
		loss and metric
		"""
		embedding = self.model(
			self.dataset, len(self.dataset)-1, len(self.dataset)
		).squeeze()
		logit = self.decoder(embedding)

		label = self.dataset.ndata['label']
		mask = self.dataset.ndata[split]
		logit, label = logit[mask], label[mask]

		loss = self.lossfn(logit, label)
		metric = self.evaluator(logit, label)
		return loss, metric
