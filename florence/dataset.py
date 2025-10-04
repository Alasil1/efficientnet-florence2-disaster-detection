"""dataset.py
Florence-2 dataset for disaster classification.
"""
import os
import glob
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A


class ClassificationAIDERDataset(Dataset):
    """Dataset for Florence-2 disaster classification with Albumentations augmentation."""
    
    def __init__(self, dataset_root, processor, split="train",
                 train_ratio=0.5, val_ratio=0.2, use_table2=False, seed=42):
        self.processor = processor
        self.split = split
        self.use_table2 = use_table2
        self.seed = seed
        self.class_names = ["collapsed_building", "fire", "flooded_areas", "normal", "traffic_incident"]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        # Augmentation (applied only in training & non-normal classes)
        self.augment_transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=30, p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.CLAHE(p=0.2),
            A.RandomShadow(p=0.2),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        ]) if split == "train" else None

        self.images = self._load_images_with_splits(dataset_root, split, train_ratio, val_ratio)
        
        # Print distribution
        class_counts = {}
        for _, class_name in self.images:
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print(f"Florence Dataset split={split} -> {len(self.images)} images")
        for class_name, count in class_counts.items():
            aug_status = "❌ (No Aug)" if class_name == "normal" else "✅ (Augmented)"
            print(f"   {class_name}: {count} images {aug_status}")

    def _load_images_with_splits(self, dataset_root, split, train_ratio=0.5, val_ratio=0.2):
        rng = random.Random(self.seed)

        TARGET = {
            "collapsed_building": {"train": 367, "val": 41,  "test": 103},
            "fire":               {"train": 249, "val": 63,  "test": 209},
            "flooded_areas":      {"train": 252, "val": 63,  "test": 211},
            "traffic_incident":   {"train": 232, "val": 59,  "test": 194},
            "normal":             {"train": 2107,"val": 527, "test": 1756},
        }

        all_images_by_class = {}
        for class_name in self.class_names:
            class_path = os.path.join(dataset_root, class_name)
            if not os.path.exists(class_path):
                print(f"⚠️ Warning: {class_path} not found")
                continue

            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
            image_paths = []
            for ext in image_extensions:
                image_paths.extend(glob.glob(os.path.join(class_path, ext)))
                image_paths.extend(glob.glob(os.path.join(class_path, ext.upper())))
            
            rng.shuffle(image_paths)
            all_images_by_class[class_name] = image_paths

        split_data = []
        for class_name, image_paths in all_images_by_class.items():
            if self.use_table2:
                train_k = TARGET[class_name]["train"]
                val_k   = TARGET[class_name]["val"]
                test_k  = TARGET[class_name]["test"]
            
                if split == "train":
                    class_images = image_paths[:train_k]
                elif split == "val":
                    class_images = image_paths[train_k:train_k + val_k]
                else:  # test
                    class_images = image_paths[train_k + val_k:train_k + val_k + test_k]
            else:
                total_images = len(image_paths)
                train_end = int(total_images * train_ratio)
                val_end = int(total_images * (train_ratio + val_ratio))
                
                if split == "train":
                    class_images = image_paths[:train_end]
                elif split == "val":
                    class_images = image_paths[train_end:val_end]
                else:
                    class_images = image_paths[val_end:]

            for img_path in class_images:
                split_data.append((img_path, class_name))

        rng.shuffle(split_data)
        return split_data

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path, class_name = self.images[idx]
        image = Image.open(image_path).convert('RGB')

        should_augment = (
            self.split == "train" and 
            class_name != "normal" and 
            self.augment_transforms is not None
        )
        
        if should_augment:
            if not isinstance(image, np.ndarray):
                image = np.array(image)
            augmented = self.augment_transforms(image=image)
            image = augmented["image"]
            if not hasattr(self, 'aug_counts'):
                self.aug_counts = {}
            self.aug_counts[class_name] = self.aug_counts.get(class_name, 0) + 1
        
        return {
            'image': image,
            'labels': self.class_to_idx[class_name],
            'class_name': class_name,
            'was_augmented': should_augment
        }
