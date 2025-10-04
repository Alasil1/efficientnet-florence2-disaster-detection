"""infer.py
Inference script for Florence-2 model.
"""
import argparse
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import PeftModel
from PIL import Image

from utils import extract_class_from_completion_robust


def load_florence_model(base_model_id, repo_id, subfolder=None, device=None):
    """Load Florence-2 model with LoRA adapter."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        trust_remote_code=True
    )
    
    if subfolder:
        model = PeftModel.from_pretrained(
            base_model,
            repo_id,
            subfolder=subfolder,
            is_trainable=False
        )
    else:
        model = PeftModel.from_pretrained(
            base_model,
            repo_id,
            is_trainable=False
        )
    
    processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)
    final_model = model.merge_and_unload()
    final_model = final_model.to(device)
    final_model.eval()
    
    return final_model, processor


def predict_single_image(model, processor, image_path, device):
    """Predict single image."""
    image = Image.open(image_path).convert("RGB")
    
    inputs = processor(
        text="<MORE_DETAILED_CAPTION>",
        images=[image],
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values=inputs['pixel_values'],
            input_ids=inputs['input_ids'],
            max_new_tokens=50,
            do_sample=False,
            num_beams=10,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id
        )
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    predicted_class = extract_class_from_completion_robust(generated_text)
    
    class_names = ["collapsed_building", "fire", "flooded_areas", "normal", "traffic_incident"]
    
    print(f"\n🖼️ Image: {image_path}")
    print(f"Generated: '{generated_text}'")
    print(f"Predicted class: {class_names[predicted_class]} ({predicted_class})")
    
    return predicted_class, generated_text


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base_model', type=str, default="microsoft/Florence-2-base-ft")
    p.add_argument('--repo_id', type=str, default='Florence_inf', 
                    help='Path to LoRA weights folder or HF repo ID (default: Florence_inf)')
    p.add_argument('--subfolder', type=str, default=None)
    p.add_argument('--image_path', type=str, required=True)
    p.add_argument('--hf_token', type=str, default=None)
    args = p.parse_args()
    
    if args.hf_token:
        from huggingface_hub import login
        login(args.hf_token)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, processor = load_florence_model(args.base_model, args.repo_id, args.subfolder, device)
    
    predict_single_image(model, processor, args.image_path, device)


if __name__ == '__main__':
    main()
