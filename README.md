# TiaRa

This repository contains a PyTorch implementaion of TiaRa of the paper titled "Time-aware Random Walk Diffusion to Improve Dynamic Graph Learning" submitted to AAAI-23.

## Prerequisites

This implementation has been tested in conda virtual environment. Please run `conda env create -f environment.yml [-n ENVNAME]` to create it. Note that the default name of the environment is `tiara`. Please see the full list of packages to be insalled in `environment.yml` where you can change their versions if necessary. The representative packages that we use are as follows:
* pytorch
* torchmetrics
* pytorch-sparse
* dgl
* pytorch-geometric
* pytorch-geometric-temporal

## Datasets and Settings

We provide datasets and preprocessing code used in the paper. The current `./data` contains datasets only for node classification, but datasets for temporal link prediction will be automatically downloaded at runtime.

|**Dataset**|**Nodes**|**Edges**|**Time Step**|**Features**|**Labels**|**Task**|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|[BitcoinAlpha](https://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html)|3,783|31,748|138|32|2|Link|
|[WikiElec](https://snap.stanford.edu/data/wiki-Elec.html)|7,125|212,854|100|32|2|Link|
|[RedditBody](https://snap.stanford.edu/data/soc-RedditHyperlinks.html)|7,125|212,854|100|32|2|Link|
|[Brain](https://tinyurl.com/y67ywq6j)|5,000|1,955,488|12|20|10|Node|
|[DBLP3](https://tinyurl.com/y67ywq6j)|4,257|23,540|10|100|3|Node|
|[DBLP5](https://tinyurl.com/y67ywq6j)|6,606|42,815|10|100|5|Node|
|[Reddit](https://tinyurl.com/y67ywq6j)|8,291|264,050|10|20|4|Node|

We perform experiments with a list of random seeds, {117, 3690, 2534, 1576, 1781}, and the searched hyperparameters of all models with TiaRa and datasets are at `settings`.
We conducted our experiments on RTX 3090 (24GB VRAM) with CUDA 11.3.

## Demo

We included a demo script `link_demo.sh` which reproduces our experiments for the temporal link prediction task.
```
bash link_demo.sh
```
The above script runs an experiment of GCRN+TiaRa on the BitcoinAlpha dataset with a random seed `117` where the searched hyperparameters are found at `settings/GCRN-BitcoinAlpha-tiara.json`.

The follofing script reproduces our experiments for the node classification task.
```
bash node_demo.sh
```
The script runs an experiment of GCRN+TiaRa on the Brain dataset with a random seed `117` where the searched hyperparameters are found at `settings/GCRN-Brain-tiara.json`.

If you want to perform an experiment on another dataset, use `run.sh`:
```bash
bash run.sh ${DATASET} ${MODEL} ${SEED} [arguments]
```
where `${DATASET}` is in the dataset table and `${MODEL}` is `GCRN`, `EvolveGCN`, and `GCN`.
You can change `[arguments]` which are options described below if necessary.

## Usage and Options

The `run.sh` uses `src/main.py` to conduct experiments with TiaRa where the implmentation of TiaRa is found at `src/augmenter.py`.
```bash
python src/main.py [--<argument name> <argument value>] [...]
```

We describe the detailed options of `src/main.py`. The following table summarizes options related to device, seed, and data.

|**Option**|**Description**|**Default**|
|:-|:-|:-|
|`device`|device name|`cuda`|
|`seed`|random seed for reproduction|`None`|
|`dataset`|dataset name|`BitcoinAlpha`|
|`time-aggregation`|length of time range in a single time step |`1200000`|
|`train-ratio`|ratio for train split|`0.7`|
|`val-ratio`|ratio for validation split|`0.1`|
|`data-dir`|dataset path|`data`|
|`verbose`|print additional informations|`False`|

* If your GPU memory is small, then you might encounter `CUDA out of memory` error. In this case, consider `--device cpu` on a workstation with enough memory space.

The following table summarizes options related to augmentation methods.

|**Option**|**Description**|**Default**|
|:-|:-|:-|
|`augment-method`|graph augmentation method|`tiara`|
|`alpha`|restart probability for tiara|`0.2`|
|`beta`|time travel probability for tiara|`0.3`|
|`eps`|filtering threshold for tiara|`1e-3`|
|`K`|number of power iteration for tiara|`100`|
|`symmetric-trick`|symmetric trick strategy for tiara|`True`|
|`dropedge`|dropedge ratio|`0`|

The following table summarizes options related to optimizers where we use Adam optimizer in this work.

|**Option**|**Description**|**Default**|
|:-|:-|:-|
|`lr`|learning rate|`0.05`|
|`weight-decay`|weight decay value|`0.0001`|
|`lr-decay`|learning rate decay value|`0.999`|

The following table summarizes options related to GNN models.

|**Option**|**Description**|**Default**|
|:-|:-|:-|
|`model`|GNN model name|`GCRN`|
|`input-dim`|input dimmension size for GNN model|`32`|
|`hidden-dim`|hidden dimmension size for GNN model|`32`|
|`output-dim`|outoput dimmension size for GNN model|`32`|
|`decoder-dim`|hidden dimmension size for decoder model|`32`|
|`num-blocks`|number of layers for GNN model|`3`|
|`rnn`|RNN model name for GNN model except GCN|`LSTM`|
|`dropout`|dropout ratio|`0`|

* `model`: {GCRN, EvolveGCN, GCN}
* `rnn`: {LSTM, GRU}


## Information of other implementations

We refer to open-source implementation of GNN models and augmentation methods at the following links:
* GCN: https://github.com/dmlc/dgl
* GCRN: https://github.com/benedekrozemberczki/pytorch_geometric_temporal
* EvolveGCN: https://github.com/benedekrozemberczki/pytorch_geometric_temporal
* DropEdge: https://github.com/dmlc/dgl
* GDC: https://github.com/gasteigerjo/gdc

## Citation

Please cite the paper if you use this code in your own work:

```
@inproceedings{LeeJ2023tiara,
  title={Time-aware Random Walk Diffusion to Improve Dynamic Graph Learning},
  author={Jong{-}whi Lee and Jinhong Jung},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2023}
}
```