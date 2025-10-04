"""infer.py
Small script to load a saved checkpoint and run evaluation / visualizations.
"""
import os
import torch
from model import DisasterClassifier
from utils import validate
from dataloaders import make_datasets, make_loaders


def load_checkpoint(model_path, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint['class_names']
    model = DisasterClassifier(num_classes=len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model, class_names, checkpoint


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', required=True)
    p.add_argument('--model_path', default='Effiecinet_Net_weight/best_model.pth', 
                    help='Path to model checkpoint (default: Effiecinet_Net_weight/best_model.pth)')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset, val_dataset, test_dataset = make_datasets(args.data_root)
    _, _, test_loader = make_loaders(train_dataset, val_dataset, test_dataset)

    model, class_names, chk = load_checkpoint(args.model_path, device)

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    test_loss, test_acc, test_report = validate(model, test_loader, criterion, device, class_names)
    print(f"Test acc: {test_acc:.2f}% | Test F1 (macro): {test_report['macro avg']['f1-score']:.4f}")


if __name__ == '__main__':
    main()
