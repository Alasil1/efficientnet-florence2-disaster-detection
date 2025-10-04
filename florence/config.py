"""config.py
Configuration for Florence-2 training.
"""

class Config:
    # Model settings
    model_name = "microsoft/Florence-2-base-ft"
    device = "cuda"  # Will be auto-detected in train.py
    torch_dtype = "float16"  # or "float32"
    
    # LoRA Configuration
    lora_r = 64
    lora_alpha = 128
    lora_dropout = 0.07
    
    # Training settings
    batch_size = 16
    learning_rate = 5e-5
    num_epochs = 10
    
    # Memory optimization
    dataloader_num_workers = 2
