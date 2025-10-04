# Quick Commands Cheat Sheet

## 🚀 Setup

```powershell
# 1. Clone repo and setup environment
git clone https://github.com/Alasil1/efficientnet-florence2-disaster-detection.git
cd efficientnet-florence2-disaster-detection
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt

# 2. Download AIDER dataset from Kaggle
# Visit: https://www.kaggle.com/datasets/chaitanya99/aider
# Extract to: C:\path\to\AIDER
```

---

## 🔧 EfficientNetV2

### 🎯 Quick Inference (Pre-trained Weights)
```powershell
# Uses Effiecinet_Net_weight/best_model.pth by default
python efficientnet\infer.py --data_root "C:\path\to\AIDER\test"
```

### Train
```powershell
python efficientnet\train.py --data_root "C:\path\to\AIDER" --batch_size 32 --epochs 50
```

### Test
```powershell
python efficientnet\infer.py --data_root "C:\path\to\AIDER" --model_path models\best_model.pth
```

### External Eval
```powershell
python external_eval\eval_efficientnet_on_external.py --model_path models\best_model.pth --dataset_root "C:\path\to\fire_dataset"
```

---

## 🌐 Florence-2

### 🎯 Quick Inference (Pre-trained Weights)
```powershell
# Uses Florence_inf folder by default
python florence\infer.py --image_path "test.jpg"
```

### Train
```powershell
python florence\train.py --data_root "C:\path\to\AIDER" --hf_token YOUR_HF_TOKEN --batch_size 16 --epochs 10
```

### Test (Custom Trained Model)
```powershell
python florence\infer.py --repo_id "username/disaster-florence" --subfolder "florence2-best-f1-0.9690" --image_path "test.jpg" --hf_token YOUR_HF_TOKEN
```

### External Eval
```powershell
python external_eval\eval_florence_on_external.py --repo_id "username/disaster-florence" --subfolder "florence2-best-f1-0.9690" --dataset_root "C:\path\to\fire_dataset" --hf_token hf_YOUR_TOKEN
```

---

## 📊 Common Options

### EfficientNet train.py
```
--data_root PATH         # Dataset root directory
--batch_size INT         # Batch size (default: 32)
--lr FLOAT               # Learning rate (default: 2e-4)
--epochs INT             # Number of epochs (default: 50)
--patience INT           # Early stopping patience (default: 10)
--save_dir PATH          # Model save directory (default: ./models)
--device cuda|cpu        # Device to use
```

### Florence train.py
```
--data_root PATH         # Dataset root directory
--batch_size INT         # Batch size (default: 16)
--lr FLOAT               # Learning rate (default: 5e-5)
--epochs INT             # Number of epochs (default: 10)
--save_dir PATH          # Model save directory (default: ./florence_models)
--hf_token TOKEN         # Hugging Face token (required)
```

---

## 🐛 Troubleshooting

### CUDA Out of Memory

**EfficientNet:**
```powershell
python efficientnet\train.py --data_root PATH --batch_size 16  # Reduce from 32
```

**Florence:**
```powershell
python florence\train.py --data_root PATH --batch_size 8  # Reduce from 16
```

### Slow Training

**Check GPU usage:**
```powershell
nvidia-smi
```

**Use fewer workers if CPU-bound:**
Edit `config.py` (Florence) or pass custom num_workers

### Model Not Loading

**Check file paths (Windows):**
```powershell
# Use forward slashes or escape backslashes
--model_path "C:/path/to/model.pth"
# OR
--model_path "C:\\path\\to\\model.pth"
```

---

## 📦 Hugging Face Hub

### Login
```powershell
huggingface-cli login
# Enter your token when prompted
```

### Push Model (Florence)
After training, push to Hub:
```python
from huggingface_hub import HfApi
api = HfApi(token="hf_YOUR_TOKEN")
api.upload_folder(
    folder_path="./florence_models/florence2-best-f1-0.9690",
    repo_id="username/disaster-florence",
    repo_type="model"
)
```

### Download Model
```powershell
# Automatically handled by infer.py, or manually:
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(repo_id="username/disaster-florence", filename="best_model.pth")
```

---

## 📁 File Organization

```
Final Code/
├── efficientnet/              # All EfficientNet files
├── florence/                  # All Florence files  
├── external_eval/             # External evaluation scripts
├── models/                    # EfficientNet checkpoints (created during training)
├── florence_models/           # Florence checkpoints (created during training)
├── efficency-net.ipynb        # Original notebook (reference)
├── train-florence.ipynb       # Original notebook (reference)
├── test-External_dataset.ipynb # Original notebook (reference)
├── requirements.txt           # Dependencies
├── README.md                  # Main documentation
└── NOTEBOOK_MAPPING.md        # Notebook→script mapping
```

---

## ⚡ Speed Tips

1. **Use SSD for dataset:** Significantly faster I/O
2. **Increase num_workers:** 2-4 for DataLoaders (edit config)
3. **Mixed precision (Florence):** Add `torch.cuda.amp` for FP16
4. **Precomputed augmentations:** Cache augmented images offline
5. **Smaller validation set:** Use subset for faster validation during dev

---

## 🎓 Learning Resources

- **EfficientNetV2 paper:** https://arxiv.org/abs/2104.00298
- **Florence-2 paper:** https://arxiv.org/abs/2311.06242
- **LoRA paper:** https://arxiv.org/abs/2106.09685
- **Albumentations docs:** https://albumentations.ai/docs/

---

## 📞 Getting Help

1. Check `README.md` in each folder
2. Review `NOTEBOOK_MAPPING.md` for notebook→script correspondence
3. Read inline code comments
4. Open an issue on GitHub
