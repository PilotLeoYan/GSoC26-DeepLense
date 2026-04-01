import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple


class NpyDataset(Dataset):
    """
    Dataset for .npy files organized as follows:
    
    dataset/
    ├── train
    │   ├── no
    │   ├── sphere
    │   └── vort
    └── val
        ├── no
        ├── sphere
        └── vort
    """

    def __init__(self, root: str, split: str = 'train'):
        self.samples: list[tuple[Path, int]] = []  # (path, label)
        self.n_samples_per_class: list[int] = []
        self.labels_name: list[str] = []
        self.labels: list[int] = []
        
        split_dir = Path(root) / split
        class_dirs = sorted(split_dir.iterdir())
        
        print(f'split: {split}')
        for label, class_dir in enumerate(class_dirs): 
            if class_dir.is_dir():
                npy_files = class_dir.glob('*.npy')
                
                self.n_samples_per_class.append(len(list(npy_files)))
                self.labels_name.append(class_dir.name)
                self.labels.append(label)
                
                print(f'\tclass: {self.labels_name[-1]} (label: {self.labels[-1]}), number of samples: {self.n_samples_per_class[-1]}')
                
                for npy_file in class_dir.glob('*.npy'):
                    self.samples.append((npy_file, label))
        
        print(f'\ttotal samples: {len(self.samples)}')

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        
        data = np.load(path)  # NumPy array on CPU RAM
        
        tensor = torch.from_numpy(data).float()
        return tensor, label


def build_dataloaders(
    data_root: str,
    batch_size: int = 64,
    valid_batch_size: int = 128,
    num_workers: int = 4,
    drop_last_train: bool = True,
    drop_last_valid: bool = False,
) -> Tuple[DataLoader, DataLoader]:

    train_dataset = NpyDataset(root=data_root, split='train')
    valid_dataset = NpyDataset(root=data_root, split='val')

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2, 
        persistent_workers=True,
        drop_last=drop_last_train,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=valid_batch_size,  
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        drop_last=drop_last_valid,
    )

    return train_loader, valid_loader
