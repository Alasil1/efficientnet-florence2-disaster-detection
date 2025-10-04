"""train.py
Main training script for Florence-2 LoRA disaster classifier.
"""
import argparse
import os
import time
import torch
from transformers import AutoProcessor, get_cosine_schedule_with_warmup
from torch.optim import AdamW

from config import Config
from dataset import ClassificationAIDERDataset
from model import FlorenceLoRAClassifier, count_parameters
from utils import make_loaders, evaluate_completion_model_with_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', type=str, required=True)
    p.add_argument('--batch_size', type=int, default=Config.batch_size)
    p.add_argument('--lr', type=float, default=Config.learning_rate)
    p.add_argument('--epochs', type=int, default=Config.num_epochs)
    p.add_argument('--save_dir', type=str, default='./florence_models')
    p.add_argument('--hf_token', type=str, default=None, help='Hugging Face token for login')
    return p.parse_args()


def main():
    args = parse_args()
    
    if args.hf_token:
        from huggingface_hub import login
        login(args.hf_token)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load processor and model
    processor = AutoProcessor.from_pretrained(Config.model_name, trust_remote_code=True)
    model = FlorenceLoRAClassifier(
        model_name=Config.model_name,
        device=device,
        torch_dtype=torch.float32
    )
    
    total, trainable = count_parameters(model)
    print(f"Total parameters: {total/1e6:.2f}M")
    print(f"Trainable (LoRA) parameters: {trainable/1e6:.2f}M")
    
    # Create datasets
    train_dataset = ClassificationAIDERDataset(args.data_root, processor, split="train", use_table2=True)
    val_dataset = ClassificationAIDERDataset(args.data_root, processor, split="val", use_table2=True)
    test_dataset = ClassificationAIDERDataset(args.data_root, processor, split="test", use_table2=True)
    
    train_loader, val_loader, test_loader = make_loaders(
        train_dataset, val_dataset, test_dataset, processor, 
        batch_size=args.batch_size, num_workers=Config.dataloader_num_workers
    )
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr)
    num_training_steps = args.epochs * len(train_loader)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=num_training_steps,
        num_cycles=0.5
    )
    
    best_val_f1 = 0.0
    
    print("🚀 Starting training...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0
        
        from tqdm import tqdm
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            outputs = model(
                pixel_values=batch['pixel_values'].to(device, dtype=torch.float32),
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device) if batch['attention_mask'] is not None else None,
                labels=batch['labels'].to(device)
            )
            
            loss = outputs.loss
            loss.backward()
            running_loss += loss.item()
            
            optimizer.step()
            lr_scheduler.step()
            
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Validation
        print("🔍 Evaluating on validation set...")
        results = evaluate_completion_model_with_metrics(model, val_loader, processor)
        
        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Training Loss: {running_loss/len(train_loader):.4f}")
        print(f"  Validation Accuracy: {results['accuracy']:.2f}%")
        print(f"  Weighted F1 Score: {results['f1_weighted']:.4f}")
        
        # Save best model
        if results['f1_weighted'] > best_val_f1:
            best_val_f1 = results['f1_weighted']
            
            checkpoint_dir = os.path.join(args.save_dir, f"florence2-best-f1-{results['f1_weighted']:.4f}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            model.save_pretrained(checkpoint_dir)
            processor.save_pretrained(checkpoint_dir)
            
            torch.save({
                'epoch': epoch + 1,
                'best_f1': best_val_f1,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'loss': running_loss/len(train_loader),
                'metrics': results
            }, os.path.join(checkpoint_dir, 'training_state.pt'))
            
            print(f"🔥 New best model saved! F1: {results['f1_weighted']:.4f}")
    
    print(f"\n🎊 Training completed! Best F1: {best_val_f1:.4f}")


if __name__ == '__main__':
    main()
