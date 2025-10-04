"""eval_florence_on_external.py
Evaluate Florence-2 model on external fire/nofire dataset.
"""
import argparse
import time
import torch
import numpy as np
import pandas as pd
from torchvision import datasets
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append('../florence')
from utils import extract_class_from_completion_robust


def evaluate_on_external_dataset(model, processor, dataset_root, device):
    """Evaluate Florence model on external ImageFolder dataset (fire/nofire)."""
    
    # Load dataset
    dataset = datasets.ImageFolder(root=dataset_root)
    class_names_binary = ["fire" if c == "fire" else "normal" for c in dataset.classes]
    
    print(f"Original classes: {dataset.classes}")
    print(f"Mapped classes: {class_names_binary}")
    print(f"Total images: {len(dataset.samples)}")
    
    all_predictions = []
    all_true_labels = []
    all_processing_times = []
    all_generated_texts = []
    all_image_paths = []
    
    print("Processing entire dataset...")
    for idx, (img_path, true_label) in enumerate(tqdm(dataset.samples, desc="Processing images")):
        try:
            image = Image.open(img_path).convert('RGB')
            prompt = "<MORE_DETAILED_CAPTION>"
            
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, dtype=torch.float32)
            
            start_time = time.time()
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=False,
                    num_beams=3,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id
                )
            end_time = time.time()
            processing_time = end_time - start_time
            
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            predicted_class_5 = extract_class_from_completion_robust(generated_text)
            
            # Map 5-class to binary (fire=1, normal=3 in original -> fire=0, normal=1 in binary)
            if predicted_class_5 == 1:  # fire
                predicted_class = 0
            elif predicted_class_5 == 3:  # normal
                predicted_class = 1
            else:
                predicted_class = 1  # default to normal
            
            all_predictions.append(predicted_class)
            all_true_labels.append(true_label)
            all_processing_times.append(processing_time)
            all_generated_texts.append(generated_text)
            all_image_paths.append(img_path)
            
        except Exception as e:
            print(f"Error processing image {idx} ({img_path}): {str(e)}")
            all_predictions.append(None)
            all_true_labels.append(true_label)
            all_processing_times.append(0)
            all_generated_texts.append("")
            all_image_paths.append(img_path)
    
    # Calculate metrics
    valid_predictions = [(pred, true) for pred, true in zip(all_predictions, all_true_labels) if pred is not None]
    if valid_predictions:
        valid_preds, valid_trues = zip(*valid_predictions)
        final_accuracy = np.mean(np.array(valid_preds) == np.array(valid_trues))
        
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        print(f"Total images processed: {len(dataset.samples)}")
        print(f"Successfully processed: {len(valid_predictions)}")
        print(f"Overall accuracy: {final_accuracy:.3f}")
        print(f"Average processing time: {np.mean([t for t in all_processing_times if t > 0]):.3f}s")
        
        # Classification report
        print("\nCLASSIFICATION REPORT:")
        print(classification_report(valid_trues, valid_preds, target_names=["fire", "nofire"]))
        
        # Confusion matrix
        cm = confusion_matrix(valid_trues, valid_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["fire", "nofire"], yticklabels=["fire", "nofire"])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix - Florence on External Dataset")
        plt.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base_model', type=str, default="microsoft/Florence-2-base-ft")
    p.add_argument('--repo_id', type=str, required=True)
    p.add_argument('--subfolder', type=str, default=None)
    p.add_argument('--dataset_root', type=str, required=True)
    p.add_argument('--hf_token', type=str, default=None)
    args = p.parse_args()
    
    if args.hf_token:
        from huggingface_hub import login
        login(args.hf_token)
    
    # Import after setting path
    sys.path.append('../florence')
    from infer import load_florence_model
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, processor = load_florence_model(args.base_model, args.repo_id, args.subfolder, device)
    
    evaluate_on_external_dataset(model, processor, args.dataset_root, device)


if __name__ == '__main__':
    main()
