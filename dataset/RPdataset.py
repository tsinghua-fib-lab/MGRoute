import os
import torch
import numpy as np
import pickle
from torch.utils.data import Dataset

class RPDataset(Dataset):

    def __init__(self, args, flags, mode='train'):

        super().__init__()

        base_path = os.path.join(os.path.dirname(__file__), os.pardir, 'data', args.dataset_name)
        assert mode in ['train', 'val', 'test']
        self.features = pickle.load(open(os.path.join(base_path, f'features_{mode}.pkl'), 'rb'))
        
        if mode == 'train':
            use_idx = len(self.features) * args.train_use_ratio
        elif mode == 'val':
            use_idx = len(self.features) * args.val_use_ratio
        elif mode == 'test':
            use_idx = len(self.features) * args.test_use_ratio

        self.features = self.features[:int(use_idx)]


        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]







        