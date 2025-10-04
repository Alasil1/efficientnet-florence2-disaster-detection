# Notebook → Python Files Mapping

This document maps the original notebooks to the organized Python files.

## 📓 `efficency-net.ipynb` → `efficientnet/`

**Location:** `notebooks/efficency-net.ipynb`

| Notebook Section | Python File | Description |
|-----------------|-------------|-------------|
| DATA cells | `dataset.py` | EfficientNetV2Dataset class |
| Model definition | `model.py` | DisasterClassifier class |
| DataLoader setup | `dataloaders.py` | make_datasets, make_loaders |
| Training functions | `utils.py` | train_epoch, validate, plotting |
| Training loop | `train.py` | Main training script with early stopping |
| Evaluation cells | `infer.py` | Load checkpoint and evaluate |
| FLOPs calculation | `utils.py` | calculate_flops_thop |

**Original notebook:** Mixed dataset, model, training, and evaluation in cells  
**New structure:** Modular files with clear separation of concerns

---

## 📓 `train-florence.ipynb` → `florence/`

**Location:** `notebooks/train-florence.ipynb`

| Notebook Section | Python File | Description |
|-----------------|-------------|-------------|
| Config class | `config.py` | Hyperparameters and settings |
| Dataset class | `dataset.py` | ClassificationAIDERDataset |
| Model + LoRA | `model.py` | FlorenceLoRAClassifier |
| Collate function | `utils.py` | make_collate_fn |
| Class completions | `utils.py` | get_class_completion |
| Text extraction | `utils.py` | extract_class_from_completion_robust |
| Evaluation function | `utils.py` | evaluate_completion_model_with_metrics |
| Training loop | `train.py` | Main training with scheduler |
| Inference cells | `infer.py` | Load from HF Hub and predict |

**Original notebook:** End-to-end Florence training with HF Hub integration  
**New structure:** Config-driven with reusable utilities

---

## 📓 `test-External_dataset.ipynb` → `external_eval/`

**Location:** `notebooks/test-External_dataset.ipynb`

| Notebook Section | Python File | Description |
|-----------------|-------------|-------------|
| Florence external test | `eval_florence_on_external.py` | Evaluate Florence on fire/nofire |
| EfficientNet external test | `eval_efficientnet_on_external.py` | Evaluate EfficientNet on fire/nofire |
| Binary mapping logic | Both files | 5-class → 2-class conversion |
| Metrics & plots | Both files | Classification reports, confusion matrix |

**Original notebook:** Interactive testing on external dataset  
**New structure:** Standalone evaluation scripts for each model

---

## 🔄 Quick Migration Guide

### If you were using the notebooks and want to switch to scripts:

1. **EfficientNet users:**
   ```powershell
   # Old: Run cells in efficency-net.ipynb
   # New: 
   cd efficientnet
   python train.py --data_root YOUR_DATA --batch_size 32 --epochs 50
   python infer.py --data_root YOUR_DATA --model_path ../models/best_model.pth
   ```

2. **Florence users:**
   ```powershell
   # Old: Run cells in train-florence.ipynb
   # New:
   cd florence
   python train.py --data_root YOUR_DATA --hf_token YOUR_TOKEN --batch_size 16
   python infer.py --repo_id YOUR_REPO --image_path test.jpg --hf_token YOUR_TOKEN
   ```

3. **External evaluation:**
   ```powershell
   # Old: Run cells in test-External_dataset.ipynb
   # New:
   cd external_eval
   python eval_florence_on_external.py --repo_id YOUR_REPO --dataset_root EXTERNAL_DATA --hf_token YOUR_TOKEN
   # OR
   python eval_efficientnet_on_external.py --model_path MODEL.pth --dataset_root EXTERNAL_DATA
   ```

---

## 🎯 Key Differences

### Notebooks
✅ Great for exploration and visualization  
✅ Interactive debugging  
❌ Hard to version control  
❌ Difficult to integrate into pipelines  
❌ No command-line arguments  

### Python Scripts
✅ Clean version control  
✅ Easy CI/CD integration  
✅ Command-line configurable  
✅ Modular and reusable  
❌ Less interactive  

---

## 💡 Best Practices

1. **Keep notebooks for:**
   - Data exploration
   - Visualization
   - Quick experiments

2. **Use scripts for:**
   - Production training
   - Automated evaluation
   - Reproducible experiments
   - Model deployment

3. **Hybrid approach:**
   - Develop in notebooks
   - Convert to scripts for final runs
   - Keep both in repo (notebooks in `notebooks/` folder)

---

## 🔧 Extending the Code

### Adding a new model approach:

1. Create folder: `new_approach/`
2. Add files: `model.py`, `dataset.py`, `train.py`, `infer.py`
3. Update main `README.md`
4. Create `new_approach/README.md` with usage

### Adding new evaluation metrics:

1. Edit `efficientnet/utils.py` or `florence/utils.py`
2. Add to `validate()` or `evaluate_completion_model_with_metrics()`
3. Update README with new metrics

### Supporting new datasets:

1. Modify `dataset.py` classes
2. Adjust class names and mappings
3. Update augmentation strategies if needed
