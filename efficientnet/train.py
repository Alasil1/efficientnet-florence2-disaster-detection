"""train.py
Main training script that uses the dataset, model and utils.
Run: python train.py --data_root PATH_TO_DATASET
"""
import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataloaders import make_datasets, make_loaders
from model import DisasterClassifier
from utils import train_epoch, validate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', type=str, required=True)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--save_dir', type=str, default='./models')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    train_dataset, val_dataset, test_dataset = make_datasets(args.data_root)
    train_loader, val_loader, test_loader, class_weight = make_loaders(train_dataset, val_dataset, test_dataset, batch_size=args.batch_size)

    class_names = train_dataset.class_names
    model = DisasterClassifier(num_classes=len(class_names))
    device = torch.device(args.device)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW([
        {'params': model.backbone.features.parameters(), 'lr': args.lr * 0.1},
        {'params': model.backbone.classifier.parameters(), 'lr': args.lr}
    ], weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5, verbose=True)

    best_val_acc = 0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    patience_counter = 0

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc, val_report = validate(model, val_loader, criterion, device, class_names)

        scheduler.step(val_acc)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch}: Train loss={train_loss:.4f}, acc={train_acc:.2f}% | Val loss={val_loss:.4f}, acc={val_acc:.2f}%")
        print(f"Val F1 (macro): {val_report['macro avg']['f1-score']:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_report': val_report,
                'class_names': class_names
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"Saved new best model with val_acc={val_acc:.2f}%")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping triggered after {epoch} epochs")
            break

    training_time = time.time() - start_time
    print(f"Training completed in {training_time/60:.1f} minutes. Best val acc: {best_val_acc:.2f}%")


if __name__ == '__main__':
    main()
