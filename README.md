# Disaster Classification: EfficientNetV2 & Florence-2

This repository contains implementations of two approaches for disaster scene classification (collapsed_building, fire, flooded_areas, normal, traffic_incident):

1. **EfficientNetV2-S** - CNN-based classifier with custom head
2. **Florence-2** - Vision-language model with LoRA fine-tuning

**✨ Pre-trained weights included!** Both models are ready to use without additional training.

---

## 📁 Project Structure

```
├── efficientnet/          # EfficientNetV2 approach
│   ├── dataset.py         # Dataset with augmentations
│   ├── model.py           # DisasterClassifier model
│   ├── dataloaders.py     # DataLoader utilities
│   ├── utils.py           # Training/validation functions
│   ├── train.py           # Training script
│   └── infer.py           # Inference script
│
├── Effiecinet_Net_weight/ # Pre-trained EfficientNet weights
│   └── best_model.pth     # Ready-to-use checkpoint
│
├── florence/              # Florence-2 approach
│   ├── config.py          # Configuration
│   ├── dataset.py         # Florence dataset
│   ├── model.py           # Florence LoRA model
│   ├── utils.py           # Training/eval utilities
│   ├── train.py           # Training script
│   └── infer.py           # Inference script
│
├── Florence_inf/          # Pre-trained Florence weights
│   └── ...                # LoRA adapters and configs
│
├── external_eval/         # External dataset evaluation
│   ├── eval_efficientnet_on_external.py
│   └── eval_florence_on_external.py
│
├── notebooks/             # Original Jupyter notebooks (reference)
│   ├── efficency-net.ipynb
│   ├── train-florence.ipynb
│   └── test-External_dataset.ipynb
│
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt
```

### 2. Use Pre-trained Models (Recommended)

**EfficientNetV2 Inference:**
```powershell
# Uses Effiecinet_Net_weight/best_model.pth by default
python efficientnet\infer.py --data_root PATH_TO_TEST_DATA
```

**Florence-2 Inference:**
```powershell
# Uses Florence_inf folder by default
python florence\infer.py --image_path test_image.jpg
```

### 3. Prepare Dataset (for training or evaluation)

Your dataset should have this structure:

```
DATA_ROOT/
  collapsed_building/
  fire/
  flooded_areas/
  normal/
  traffic_incident/
```

Each class directory contains the images for that class.

---

## 🔧 EfficientNetV2 Approach

### Quick Start with Pre-trained Weights

**Use the included weights for immediate inference:**

```powershell
# Uses Effiecinet_Net_weight/best_model.pth by default
python efficientnet\infer.py --data_root PATH_TO_DATA
```

### Training from Scratch

```powershell
python efficientnet\train.py --data_root PATH_TO_DATA --batch_size 32 --epochs 50
```

### Key Features
- EfficientNetV2-S backbone with ImageNet pretrained weights
- Custom classifier head (512 hidden units)
- Albumentations augmentation (excluding 'normal' class)
- Weighted sampler for class imbalance
- Early stopping with patience
- Label smoothing (0.1)

### Hyperparameters
- Input size: 384×384
- Batch size: 32
- Learning rate: 2e-4 (0.1× for backbone, 1× for head)
- Optimizer: AdamW with weight decay 0.01
- Scheduler: ReduceLROnPlateau

---

## 🌐 Florence-2 Approach

### Quick Start with Pre-trained Weights

**Use the included LoRA adapters for immediate inference:**

```powershell
# Uses Florence_inf folder by default (no HF token needed)
python florence\infer.py --image_path PATH_TO_IMAGE
```

### Training from Scratch

```powershell
python florence\train.py --data_root PATH_TO_DATA --hf_token YOUR_HF_TOKEN --batch_size 16 --epochs 10
```

### Key Features
- Florence-2-base-ft with LoRA adapters
- Text completion-based classification
- Class-specific prompts and completions
- Robust extraction from generated text
- Gradient checkpointing for memory efficiency

### LoRA Configuration
- r=64, alpha=128, dropout=0.07
- Target modules: q_proj, k_proj, v_proj, o_proj
- Trainable params: ~20M (vs 230M total)

### Hyperparameters
- Batch size: 16
- Learning rate: 5e-5
- Optimizer: AdamW
- Scheduler: Cosine with warmup (100 steps)
- Max new tokens: 50
- Num beams: 10

---

## 🧪 External Dataset Evaluation

Both models can be evaluated on external datasets (e.g., fire/nofire binary classification):

### Florence-2

```powershell
python external_eval\eval_florence_on_external.py --repo_id YOUR_HF_REPO --dataset_root PATH_TO_EXTERNAL_DATA --hf_token YOUR_HF_TOKEN
```

### EfficientNet

```powershell
python external_eval\eval_efficientnet_on_external.py --model_path PATH_TO_MODEL --dataset_root PATH_TO_EXTERNAL_DATA
```

---

## 📊 Model Comparison

| Feature | EfficientNetV2 | Florence-2 |
|---------|---------------|-----------|
| Parameters | ~21M | ~230M (20M trainable) |
| Input | 384×384 RGB | Variable size |
| Training time | ~2-3h (50 epochs) | ~4-5h (10 epochs) |
| Inference speed | ~10ms/image | ~100ms/image |
| Classification method | Direct logits | Text completion |
| Augmentation | Albumentations | Albumentations |
| Best use case | Fast inference | Zero-shot capable |

---


## 🔑 Key Notes

### EfficientNetV2
- Uses fixed per-class splits when `use_table2=True` in dataset
- Augmentations applied only to non-'normal' classes during training
- Can be used for transfer learning on other disaster datasets

### Florence-2
- Requires Hugging Face token for model access
- Generates natural language descriptions then extracts class
- Adapters can be pushed to HF Hub for sharing
- Better for few-shot and zero-shot scenarios

### External Evaluation
- Both models map 5-class predictions to binary (fire/normal)
- EfficientNet uses probability comparison between fire (idx=1) and normal (idx=3)
- Florence uses text pattern matching with fallback to 'normal'

---

## 🛠️ Development Tips

1. **For faster iteration:** Start with EfficientNetV2 (smaller, faster)
2. **For better generalization:** Use Florence-2 (handles diverse descriptions)
3. **For production:** EfficientNetV2 (lower latency)
4. **For research:** Florence-2 (interpretable outputs)

---



## ❓ Troubleshooting

**Q: CUDA out of memory during Florence training?**
- Reduce batch size to 8 or 4
- Use gradient accumulation
- Enable mixed precision training

**Q: EfficientNet overfitting?**
- Increase dropout rate
- Add more augmentations
- Use earlier checkpoint

**Q: Florence generating wrong completions?**
- Check your prompt format
- Tune num_beams parameter
- Review extraction patterns in `utils.py`

