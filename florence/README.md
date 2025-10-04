# Florence-2 Disaster Classification

Vision-language model approach using Florence-2 with LoRA fine-tuning for disaster classification via text completion.

## Files

- `config.py` - Configuration (LoRA params, training settings)
- `dataset.py` - Dataset for Florence-2 with augmentations
- `model.py` - FlorenceLoRAClassifier wrapper
- `utils.py` - Collate function, evaluation, text extraction
- `train.py` - Training script
- `infer.py` - Inference script

## Usage

### Training

```powershell
python train.py --data_root PATH_TO_DATA --hf_token YOUR_HF_TOKEN --batch_size 16 --epochs 10
```

### Inference (from HF Hub)

```powershell
python infer.py --repo_id YOUR_HF_REPO --subfolder florence2-best-f1-0.XXXX --image_path test_image.jpg --hf_token YOUR_HF_TOKEN
```

### Inference (local checkpoint)

```powershell
python infer.py --repo_id PATH_TO_LOCAL_DIR --image_path test_image.jpg
```

## How It Works

1. **Input:** Image + prompt `<MORE_DETAILED_CAPTION>`
2. **Model generates:** Text completion describing the scene
3. **Extraction:** Parse text to identify disaster class
4. **Output:** Predicted class index

### Class-Specific Completions

```python
collapsed_building → "This is a collapsed building scene"
fire → "This shows an active fire with visible flames and smoke"
flooded_areas → "This is a flooded area scene"
normal → "This is a normal and undamaged scene"
traffic_incident → "This depicts a traffic accident with damaged vehicles"
```

## Model Architecture

```
Florence-2-base-ft (230M params)
  ↓
LoRA adapters on q_proj, k_proj, v_proj, o_proj
  ↓
Generate text completion
  ↓
Extract class from text (pattern matching)
```

## LoRA Configuration

```python
r = 64                    # Rank
alpha = 128               # Scaling factor
dropout = 0.07            # Dropout rate
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
```

**Trainable:** ~20M params (LoRA only)  
**Total:** ~230M params

## Key Hyperparameters

- **Batch size:** 16
- **Learning rate:** 5e-5
- **Optimizer:** AdamW
- **Scheduler:** Cosine with warmup (100 steps, 0.5 cycles)
- **Max new tokens:** 50
- **Num beams:** 10
- **Training epochs:** 10

## Augmentations (Training, excluding 'normal')

- HorizontalFlip (p=0.5)
- VerticalFlip (p=0.2)
- Rotate (±30°, p=0.7)
- RandomBrightnessContrast (p=0.5)
- CLAHE (p=0.2)
- RandomShadow (p=0.2)
- GaussianBlur (p=0.5)

## Expected Results

- Training time: ~4-5 hours (10 epochs, GPU)
- Inference: ~100ms per image (beam search)
- Memory: ~12GB VRAM (fp32) or ~6GB (fp16)
- Validation F1 (weighted): ~0.97

## Tips

- **OOM?** Reduce batch size to 8 or use gradient accumulation
- **Wrong predictions?** Tune beam search (`num_beams=3-15`)
- **Slow inference?** Use `do_sample=False` and reduce `num_beams`
- **Custom classes?** Modify `get_class_completion()` and extraction patterns

## Pushing to Hugging Face Hub

After training, push your best checkpoint:

```python
from huggingface_hub import HfApi
api = HfApi(token="YOUR_HF_TOKEN")
api.upload_folder(
    folder_path="./florence_models/florence2-best-f1-0.XXXX",
    repo_id="YOUR_USERNAME/disaster-florence",
    repo_type="model"
)
```

## Advanced: Custom Extraction Patterns

Edit `utils.py` → `extract_class_from_completion_robust()` to add new keywords or adjust scoring:

```python
class_patterns = {
    1: {  # fire
        'primary': ['fire scene', 'fire'],           # +3 points
        'secondary': ['burning', 'flames'],          # +2 points
        'context': ['this is', 'shows']              # +1 point
    },
    # ... add more classes
}
```
