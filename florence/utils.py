"""utils.py
Utilities for Florence-2 training and evaluation.
"""
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def get_class_completion(class_name):
    """Generate class-specific completion targets."""
    completions = {
        "collapsed_building": "This is a collapsed building scene",
        "fire": "This shows an active fire with visible flames and smoke", 
        "flooded_areas": "This is a flooded area scene",
        "normal": "This is a normal and undamaged scene.",
        "traffic_incident": "This depicts a traffic accident with damaged vehicles"
    }
    return completions[class_name]


def extract_class_from_completion_robust(generated_text):
    """Robust extraction that handles variations in generated text."""
    text = generated_text.lower().replace("<MORE_DETAILED_CAPTION>", "").replace("<more_detailed_caption>", "").strip()
    
    class_patterns = {
        1: {  # fire
            'primary': ['fire scene', 'fire'],
            'secondary': ['fire', 'burning', 'flames', 'smoke', 'blaze'],
            'context': ['this is', 'shows', 'depicts']
        },
        0: {  # collapsed_building  
            'primary': ['collapsed building scene', 'building collapse'],
            'secondary': ['collapsed', 'destroyed building', 'building damage', 'structural damage'],
            'context': ['this is', 'shows', 'depicts']
        },
        2: {  # flooded_areas
            'primary': ['flooded area scene', 'flooding scene'],
            'secondary': ['flood', 'flooded', 'water damage', 'inundated'],
            'context': ['this is', 'shows', 'depicts']
        },
        4: {  # traffic_incident
            'primary': ['traffic incident scene', 'traffic accident'],
            'secondary': ['accident', 'crash', 'collision', 'vehicle damage', 'car damaged'],
            'context': ['this is', 'shows', 'depicts']
        },
        3: {  # normal
            'primary': ['normal', 'normal undamaged scene', 'undamaged scene'],
            'secondary': ['normal', 'regular', 'typical', 'ordinary', 'undamaged', 'no signs of', 'no disasters'],
            'context': ['this is', 'shows', 'depicts', 'a normal']
        }
    }
    
    class_scores = {}
    for class_idx, patterns in class_patterns.items():
        score = 0
        for pattern in patterns['primary']:
            if pattern in text:
                score += 3
        for keyword in patterns['secondary']:
            if keyword in text:
                score += 2
        for context in patterns['context']:
            if context in text:
                score += 1
                break
        class_scores[class_idx] = score
    
    if max(class_scores.values()) > 0:
        return max(class_scores, key=class_scores.get)
    else:
        return 3  # default to normal


def make_collate_fn(processor):
    """Create collate function for Florence-2 batching."""
    def collate(batch):
        images = [item['image'] for item in batch]
        input_prompts = ['<MORE_DETAILED_CAPTION>'] * len(images)
        
        target_completions = []
        for item in batch:
            class_name = item['class_name']
            completion = get_class_completion(class_name)
            target_completions.append(f"<MORE_DETAILED_CAPTION>{completion}")
        
        inputs = processor(images=images, text=input_prompts, return_tensors="pt", padding=True)
        targets = processor.tokenizer(
            target_completions, 
            return_tensors="pt", 
            padding=True,
            truncation=True
        )
        
        return {
            'pixel_values': inputs['pixel_values'],
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs.get('attention_mask'),
            'labels': targets['input_ids'],
            'class_labels': torch.tensor([item['labels'] for item in batch], dtype=torch.long),
            'images': images
        }
    return collate


def make_loaders(train_dataset, val_dataset, test_dataset, processor, batch_size=16, num_workers=2):
    """Create train/val/test loaders with weighted sampling for train."""
    labels = [int(sample["labels"]) for sample in train_dataset]
    classes = np.unique(labels)
    class_counts = np.array([(labels == c).sum() for c in classes], dtype=np.int64)
    class_weight = {c: 1.0 / cnt for c, cnt in zip(classes, class_counts)}
    sample_weights = np.array([class_weight[int(l)] for l in labels], dtype=np.float64)

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True
    )
    
    collate_fn = make_collate_fn(processor)
    
    train_loader = DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader


def evaluate_completion_model_with_metrics(model, val_loader, processor):
    """Evaluate model with robust extraction, F1 score, and confusion matrix."""
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device) 
    y_true = []
    y_pred = []
    
    print("\n🔍 Evaluating with robust extraction...")
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            batch_size = batch['class_labels'].size(0)
            
            for i in range(batch_size):
                try:
                    inputs = processor(
                        text="<MORE_DETAILED_CAPTION>",
                        images=[batch['images'][i]],
                        return_tensors="pt"
                    ).to('cuda')
                    
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
                    true_class = int(batch['class_labels'][i])
                    
                    y_true.append(true_class)
                    y_pred.append(predicted_class)
                    
                except Exception:
                    y_true.append(int(batch['class_labels'][i]))
                    y_pred.append(3)  # default to 'normal'
                    continue
    
    # Calculate metrics
    accuracy = 100 * sum([int(t==p) for t,p in zip(y_true, y_pred)]) / max(len(y_true), 1)
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    cm = confusion_matrix(y_true, y_pred)
    
    class_names = ["collapsed_building", "fire", "flooded_areas", "normal", "traffic_incident"]
    
    print(f"\n📊 EVALUATION RESULTS:")
    print(f"🎯 Accuracy: {accuracy:.2f}%")
    print(f"📈 Weighted F1 Score: {f1_weighted:.4f}")
    print(f"📈 Macro F1 Score: {f1_macro:.4f}")
    
    print(f"\n📋 CLASSIFICATION REPORT:")
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()
    
    return {
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'confusion_matrix': cm,
        'y_true': y_true,
        'y_pred': y_pred
    }
