import os, json
import torch
from fire import Fire
from loguru import logger
from utils import fix_seed
import dataset_loader, module
from dataset_loader.link import LinkDatasetTemplate
from dataset_loader.node import NodeDatasetTemplate
from trainer import Trainer
from evaluator import AUCMetric, F1Metric
import augmenter

def main(
	# dataset and preprocessing parameters
	dataset='BitcoinAlpha',
	time_aggregation=1200000,

	# train / validataion / test ratio
	train_ratio=0.7,
	val_ratio=0.1,

	# augment method
	augment_method='tiara',
	alpha=0.2,
	beta=0.3,
	eps=1e-3,
	K=100,
	symmetric_trick=True,
	dense=False,

	# epochs and early stopping
	epochs=200,
	early_stopping=50,

	# hyperparameters for training
	lr=0.05,
	weight_decay=0.0001,
	lr_decay=0.999,

	# stuffs
	data_dir='data',
	device='cuda',
	verbose=False,
	seed=None,

	# model arguments
	model='GCRN',
	input_dim=32,
	output_dim=32,
	decoder_dim=32,
	**kwargs
):
	"""
	Main function for experiment

	Parameters
	----------
	dataset
		dataset name
	time_aggregation
		time step size in secjjonds

	train_ratio
		train split ratio
	val_ratio
		validation split ratio

	augment_method
		dataset augmentation method
	alpha
		teleport probability
	beta
		time treval probability
	eps
		link threshold
	K
		number of diffusion iteration
	symmetric_trick
		method to generate normalized symmetric adjacency matrix
	dense
		whether to use dense adjacency matrix

	epochs
		max epochs
	early_stopping
		early stopping patient
	metric_for_early_stopping
		metric for early stopping

	lr
		learning rate
	weight_decay
		weight decay
	lr_decay
		learning rate decay

	data_dir
		path of the data directory
	device
		device name
	verbose
		verbose mode
	seed
		random seed

	model
		model name
	input_dim
		input feature dimension
	output_dim
		output feature dimension
	decoder_dim
		hidden classifier dimension
	kwargs
		model parameters

	Returns
	-------
	test_metrics
		final test metrics
	history
		training history

	Examples
	--------
	Usage for python code:

	>>> import main as experiment

	Usage for command line:

	$ python src/main.py [--dataset DATASET] [...]
	"""
	# fix seed

	fix_seed(seed)

	# use only one device

	if device[:5] == 'cuda:':
		torch.cuda.set_device(int(device[5:]))

	# load dataset

	dataset_name = dataset
	dataset = dataset_loader.load(
		dataset,
		input_dim,
		train_ratio,
		val_ratio,
		device,
		data_dir,
		seed=seed,
		time_aggregation=time_aggregation
	)

	# build model

	model_arguments = {
		'input_dim': input_dim,
		'output_dim': output_dim,
		'device': device,
		'renorm_order': 'sym',
		**kwargs
	}

	if augment_method == 'tiara' and not symmetric_trick:
		model_arguments['renorm_order'] = 'row'

	model_name = model
	if model == 'GCN':
		model = module.GCN(**model_arguments)
	elif model == 'GCRN':
		model = module.GCRN(**model_arguments)
	elif model == 'EvolveGCN':
		model = module.EvolveGCN(num_nodes=dataset.num_nodes, **model_arguments)
	else:
		raise NotImplementedError("no such model {}".format(model))

	# build augmenter

	tiara_argments = (
		alpha, beta, eps, K,
		symmetric_trick,
		device,
		dense,
		verbose
	)

	if augment_method == 'tiara':
		augment_method = augmenter.Tiara(*tiara_argments)
	elif augment_method == 'merge':
		augment_method = augmenter.Merge(device)
	elif augment_method == 'none':
		augment_method = augmenter.GCNNorm(device)
	else:
		raise NotImplementedError('no such augmenter {}'.format(augment_method))

	# build decoder, loss function, evaluator

	if isinstance(dataset, LinkDatasetTemplate):
		decoder = module.PairDecoder(output_dim, decoder_dim, device)
		lossfn = module.PairLoss()
		evaluator = AUCMetric()
	elif isinstance(dataset, NodeDatasetTemplate):
		decoder = module.NodeDecoder(output_dim, decoder_dim, dataset.num_label, device)
		lossfn = module.NodeLoss()
		evaluator = F1Metric(dataset.num_label)
	else:
		raise

	# train the model

	trainer = Trainer(
		model, decoder, lossfn, dataset, evaluator, augment_method
	)
	model, decoder, history = trainer.train(
		epochs,
		lr,
		weight_decay,
		lr_decay,
		early_stopping
	)

	# test the model

	model.eval()
	decoder.eval()

	with torch.no_grad():
		_, test_metric = trainer.calc_loss_and_metrics('test', True)

	logger.info('dataset: {}'.format(dataset_name))
	logger.info('model: {}'.format(model_name))
	logger.info('seed: {}'.format(seed))
	logger.info('final metric score is {:7.4f}'.format(test_metric))
	return test_metric.item(), history

def main_wraper(**kwargs):
	conf_file = kwargs.get('conf_file', None)
	if conf_file is not None:
		config = json.load(open(conf_file, 'r'))
		for k in kwargs:
			if k in config:
				logger.warning('{} will be overwritten!'.format(k))

		setting = {**config, **kwargs}
		report_setting(setting)
		test_metric, history = main(**setting)

		save_file = kwargs.get('save_file', None)
		if save_file is not None:
			os.makedirs('results', exist_ok=True)
			json.dump({'test_metric': test_metric, 'history': history}, open('results/' + save_file, 'w'))

	else:
		report_setting(kwargs)
		main(**kwargs)

def report_setting(setting):
	for k, v in setting.items():
		logger.info('{}: {}'.format(k, v))

if __name__ == "__main__":
	Fire(main_wraper)
