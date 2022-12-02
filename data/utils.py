import os

import torch
import torch.utils.data as D

import numpy as np
from sklearn.preprocessing import StandardScaler


DOMAIN_NAME_TO_INDEX = {
    'in': 0,
    'out': 1.
}


def load_dataset(config):
    dataset_path = os.path.join(config.data_dir, f'{config.dataset_name}.npz')
    data = np.load(dataset_path)
    features, targets = data['X'], data['y']

    dataset = features, targets
    return dataset


def process_dataset(dataset, config=None):
    features, targets = dataset
    scaler = StandardScaler()

    features_in = features[targets == DOMAIN_NAME_TO_INDEX['in']]
    scaler.fit(features_in)
    features = scaler.transform(features)
    
    dataset = features, targets
    return dataset, scaler


def get_datasets(dataset, config=None):
    features, targets = dataset
    datasets = {
        domain_name: D.TensorDataset(torch.from_numpy(features[targets == domain_index]).to(torch.float32))
        for domain_name, domain_index in DOMAIN_NAME_TO_INDEX.items()
    }

    return datasets


def get_dataloaders(datasets, config=None):
    dataloaders = {
        domain_name: D.DataLoader(
            dataset, 
            batch_size=config.batch_size, 
            num_workers=config.num_workers, 
            shuffle=(domain_name == 'in'), 
            drop_last=(domain_name == 'in')
        ) for domain_name, dataset in datasets.items()
    }

    return dataloaders