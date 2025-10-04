"""model.py
Florence-2 LoRA classifier model.
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from config import Config


class FlorenceLoRAClassifier(nn.Module):
    """Florence-2 with LoRA fine-tuning for disaster classification."""
    
    def __init__(self, model_name, device, torch_dtype=torch.float16):
        super().__init__()

        # Load base Florence-2 (encoder-decoder VLM)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True
        ).to(device)

        # LoRA configuration
        TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
        lora_cfg = LoraConfig(
            r=Config.lora_r,
            lora_alpha=Config.lora_alpha,
            target_modules=TARGET_MODULES,
            task_type="CAUSAL_LM",
            lora_dropout=Config.lora_dropout,
            bias="none",
            inference_mode=False
        )
        self.model = get_peft_model(self.model, lora_cfg)
        
    def forward(self, pixel_values, input_ids=None, attention_mask=None, labels=None):
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        return outputs

    def generate(self, pixel_values, input_ids, **kwargs):
        return self.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            **kwargs
        )


def count_parameters(model):
    """Count total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params
