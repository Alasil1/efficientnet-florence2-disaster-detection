# EfficientNetV2 Disaster Classification

CNN-based approach using EfficientNetV2-S for 5-class disaster scene classification.

## Files

- `dataset.py` - Dataset with Albumentations augmentation
- `model.py` - DisasterClassifier (EfficientNetV2-S + custom head)
- `dataloaders.py` - DataLoader creation with weighted sampling
- `utils.py` - Training, validation, plotting utilities
- `train.py` - Training script
- `infer.py` - Inference/evaluation script

## Usage

### Training

```powershell
python train.py --data_root PATH_TO_DATA --batch_size 32 --epochs 50 --lr 2e-4
```

### Inference

```powershell
python infer.py --data_root PATH_TO_DATA --model_path ..\models\best_model.pth
```

## Model Architecture

```
EfficientNetV2-S (ImageNet pretrained)
  ↓
Features extraction (frozen/fine-tuned)
  ↓
Dropout(0.3) → Linear(1280→512) → ReLU → Dropout(0.3) → Linear(512→5)
  ↓
Class logits
```

## Key Hyperparameters

- **Input size:** 384×384
- **Batch size:** 32
- **Learning rate:** 2e-4 (backbone: 2e-5, head: 2e-4)
- **Optimizer:** AdamW (weight decay 0.01)
- **Scheduler:** ReduceLROnPlateau (patience=5, factor=0.5)
- **Loss:** CrossEntropyLoss with label smoothing (0.1)
- **Early stopping:** Patience=10

## Augmentations (Training only, excluding 'normal' class)

- Resize(1.08×) + RandomCrop(384)
- HorizontalFlip (p=0.5)
- VerticalFlip (p=0.2)
- Rotate (±30°, p=0.7)
- RandomBrightnessContrast (p=0.5)
- CLAHE (p=0.2)
- RandomShadow (p=0.2)
- GaussianBlur (p=0.5)

## Expected Results

- Training time: ~2-3 hours (50 epochs, GPU)
- Inference: ~10ms per image
- Memory: ~3GB VRAM
- Parameters: ~21M (all trainable)

## Tips

- Use `--patience 15` for longer training without early stop
- Adjust `--lr` if loss plateaus early
- For smaller datasets, increase dropout to 0.4-0.5
