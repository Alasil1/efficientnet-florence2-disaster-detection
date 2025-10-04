"""dataset.py
Dataset class extracted and cleaned from the notebook.
"""
import os
import glob
import random
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


class EfficientNetV2Dataset(Dataset):
    """Dataset for EfficientNetV2 experiments.

    Notes:
    - If split == 'train' and class_name != 'normal' we apply Albumentations augmentations
    - If use_table2=True, fixed counts per-class are used (from notebook target table)
    """
    def __init__(self, dataset_root, split="train", train_ratio=0.5, val_ratio=0.2,
                 use_table2=False, seed=42, input_size=384):
        self.split = split
        self.use_table2 = use_table2
        self.seed = seed
        self.input_size = input_size
        self.class_names = ["collapsed_building", "fire", "flooded_areas", "normal", "traffic_incident"]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        # torchvision transforms for base (validation/test) pipeline
        self.base_transforms = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Albumentations augmentations for training
        self.augment_transforms = A.Compose([
            A.Resize(int(self.input_size * 1.08), int(self.input_size * 1.08)),
            A.RandomCrop(self.input_size, self.input_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=30, p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.CLAHE(p=0.2),
            A.RandomShadow(p=0.2),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]) if split == "train" else None

        self.images = self._load_images_with_splits(dataset_root, split, train_ratio, val_ratio)

        # minimal distribution print
        class_counts = {}
        for _, class_name in self.images:
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        print(f"Dataset split={split} -> {len(self.images)} images")
        for cname, cnt in class_counts.items():
            print(f"  {cname}: {cnt}")

        self.aug_counts = {}

    def _load_images_with_splits(self, dataset_root, split, train_ratio=0.5, val_ratio=0.2):
        rng = random.Random(self.seed)

        TARGET = {
            "collapsed_building": {"train": 367, "val": 41,  "test": 103},
            "fire":               {"train": 249, "val": 63,  "test": 209},
            "flooded_areas":      {"train": 252, "val": 63,  "test": 211},
            "traffic_incident":   {"train": 232, "val": 59,  "test": 194},
            "normal":             {"train": 2107,"val": 527, "test": 1756},
        }

        split_data = []
        for class_name in self.class_names:
            class_path = os.path.join(dataset_root, class_name)
            if not os.path.exists(class_path):
                print(f"Warning: {class_path} not found, skipping")
                continue

            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
            image_paths = []
            for ext in image_extensions:
                image_paths.extend(glob.glob(os.path.join(class_path, ext)))
                image_paths.extend(glob.glob(os.path.join(class_path, ext.upper())))

            rng.shuffle(image_paths)

            if self.use_table2:
                train_count = TARGET[class_name]["train"]
                val_count = TARGET[class_name]["val"]
                test_count = TARGET[class_name]["test"]
                total_needed = train_count + val_count + test_count
                if len(image_paths) < total_needed:
                    print(f"Warning: {class_name} has {len(image_paths)} images, needs {total_needed}")

                if split == "train":
                    class_images = image_paths[:train_count]
                elif split == "val":
                    class_images = image_paths[train_count:train_count + val_count]
                else:
                    class_images = image_paths[train_count + val_count:train_count + val_count + test_count]
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

            print(f"  {class_name} - {split}: {len(class_images)} images")

        rng.shuffle(split_data)
        return split_data

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path, class_name = self.images[idx]
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = Image.new('RGB', (self.input_size, self.input_size), (0, 0, 0))

        should_augment = (
            self.split == "train" and
            class_name != "normal" and
            self.augment_transforms is not None
        )

        if should_augment:
            image = np.array(image)
            image = self.augment_transforms(image=image)["image"]
            self.aug_counts[class_name] = self.aug_counts.get(class_name, 0) + 1
        else:
            image = self.base_transforms(image)

        return {
            'image': image,
            'labels': self.class_to_idx[class_name],
            'class_name': class_name,
            'was_augmented': should_augment,
            'image_path': image_path
        }

    def get_class_distribution(self):
        class_counts = {}
        for _, class_name in self.images:
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        return class_counts


if __name__ == "__main__":
    print("dataset.py: instantiate with your dataset root to test")
