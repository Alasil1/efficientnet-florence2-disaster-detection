"""dataloaders.py
Helpers to create datasets and dataloaders (train/val/test).
"""
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
import numpy as np
from dataset import EfficientNetV2Dataset


def make_datasets(dataset_root, input_size=384, use_table2=True, seed=42):
    train_dataset = EfficientNetV2Dataset(dataset_root, split="train", use_table2=use_table2, seed=seed, input_size=input_size)
    val_dataset = EfficientNetV2Dataset(dataset_root, split="val", use_table2=use_table2, seed=seed, input_size=input_size)
    test_dataset = EfficientNetV2Dataset(dataset_root, split="test", use_table2=use_table2, seed=seed, input_size=input_size)
    return train_dataset, val_dataset, test_dataset


def make_loaders(train_dataset, val_dataset, test_dataset, batch_size=32):
    # build weighted sampler from train dataset
    labels = [int(sample['labels']) for sample in train_dataset]
    classes = np.unique(labels)
    class_counts = np.array([(labels == c).sum() for c in classes], dtype=np.int64)
    class_weight = {c: 1.0 / cnt for c, cnt in zip(classes, class_counts)}
    sample_weights = np.array([class_weight[int(l)] for l in labels], dtype=np.float64)

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, class_weight


if __name__ == "__main__":
    print("dataloaders.py: use make_datasets and make_loaders")
