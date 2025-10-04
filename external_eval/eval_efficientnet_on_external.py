"""eval_efficientnet_on_external.py
Evaluate EfficientNet model on external fire/nofire dataset.
"""
import argparse
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

import sys
sys.path.append('../efficientnet')
from model import DisasterClassifier


def evaluate_efficientnet_on_external(model, dataset_root, device):
    """Evaluate EfficientNet model on external fire/nofire dataset."""
    
    # Define transforms (match training)
    test_transform = transforms.Compose([
        transforms.Resize((384, 384)),  # EfficientNetV2-S input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    fire_dataset = datasets.ImageFolder(
        root=dataset_root,
        transform=test_transform
    )
    
    print(f"Dataset classes: {fire_dataset.classes}")
    print(f"Total images: {len(fire_dataset)}")
    
    # Create DataLoader
    val_loader = DataLoader(
        fire_dataset, 
        batch_size=32,
        shuffle=False,  
        num_workers=2,  
        pin_memory=True
    )
    
    # Evaluate
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    class_names = ["fire", "nofire"]  # Binary mapping
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Evaluating EfficientNet')
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            # For 5-class model on 2-class data:
            # Extract fire (idx=1) and normal (idx=3) probabilities
            probabilities = torch.softmax(outputs, dim=1)
            fire_probs = probabilities[:, 1]
            normal_probs = probabilities[:, 3]
            
            # Binary prediction: 0=fire, 1=nofire
            binary_preds = (normal_probs > fire_probs).long()
            
            total += labels.size(0)
            correct += binary_preds.eq(labels).sum().item()
            
            all_preds.extend(binary_preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'Acc': f'{100.*correct/total:.2f}%'})
    
    accuracy = 100. * correct / total
    
    print(f"\n📊 Results:")
    print(f"Accuracy: {accuracy:.2f}%")
    
    # Classification report
    print("\nCLASSIFICATION REPORT:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - EfficientNet on External Dataset")
    plt.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', type=str, required=True, help='Path to .pth checkpoint')
    p.add_argument('--dataset_root', type=str, required=True)
    args = p.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    checkpoint = torch.load(args.model_path, map_location=device)
    class_names = checkpoint['class_names']
    
    model = DisasterClassifier(num_classes=len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded from {args.model_path}")
    print(f"Model classes: {class_names}")
    
    evaluate_efficientnet_on_external(model, args.dataset_root, device)


if __name__ == '__main__':
    main()
