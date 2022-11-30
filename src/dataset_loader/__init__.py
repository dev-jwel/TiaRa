from .link import BitcoinAlpha, WikiElec, RedditBody
from .node import Brain, Reddit, DBLP3, DBLP5

def load(name, *args, **kwargs):
    dataset_classes = {
        'BitcoinAlpha': BitcoinAlpha,
        'WikiElec': WikiElec,
        'RedditBody': RedditBody,
        'Brain': Brain,
        'Reddit': Reddit,
        'DBLP3': DBLP3,
        'DBLP5': DBLP5
    }

    if name in dataset_classes:
        return dataset_classes[name](*args, **kwargs)
    else:
        raise NotImplementedError('no such dataset {}'.format(name))
