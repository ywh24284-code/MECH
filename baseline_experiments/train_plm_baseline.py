#!/usr/bin/env python3
"""
PLM Baseline Model Training Script (Group A)

Supported models:
- BERT-base
- RoBERTa-large

Uses the same Prompt-tuning style input as the hybrid model
Full Fine-tuning, single-task opinion evolution classification
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm
import warnings
import sys

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class OpinionEvolutionDataset(Dataset):
    
    def __init__(
        self,
        csv_file: str,
        tokenizer,
        max_length: int = 256,
        use_context: bool = True,
        context_window: int = 5,
        use_turn_indicators: bool = True
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_context = use_context
        self.context_window = context_window
        self.use_turn_indicators = use_turn_indicators
        
        self.teacher_names = {'T', 'Teacher', 'Ms. G', 'Mrs. G'}
    
        print(f"Loading data: {csv_file} ...")
        df = pd.read_csv(csv_file)
        
        required_cols = ['Sentence', 'Speaker', 'label']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"CSV missing required columns: {missing_cols}. Current columns: {df.columns.tolist()}")
        
        # Group by classroom
        grouped = df.groupby('class', sort=False)
        
        self.samples = []
        self.all_opinion_labels = []
        
        for class_id, group in grouped:
            sentences = group['Sentence'].tolist()
            speakers = group['Speaker'].tolist()
            classifications = group['label'].tolist()
            
            for i in range(len(sentences)):
                # Extract context
                start_idx = max(0, i - context_window)
                context_sentences = sentences[start_idx:i]
                context_speakers = speakers[start_idx:i]
                
                sample = {
                    'sentence': sentences[i],
                    'speaker': speakers[i],
                    'opinion_label': classifications[i],
                    'context_sentences': context_sentences,
                    'context_speakers': context_speakers,
                    'class_id': class_id
                }
                
                self.samples.append(sample)
                self.all_opinion_labels.append(classifications[i])
        
        print(f"Loaded. Valid samples: {len(self.samples)}")
        
        print(f"\nOpinion evolution label distribution:")
        unique_labels, counts = np.unique(self.all_opinion_labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            label_name = ['Irrelevant', 'New', 'Strengthened', 'Weakened', 'Adopted', 'Refuted'][label]
            print(f"  {label} ({label_name}): {count} samples")
    
    def _get_role_name(self, speaker_name, current_speaker_name):
        """Get relative role marker"""
        s_name = str(speaker_name).strip()
        if s_name.startswith('T') or s_name in self.teacher_names:
            return "[TEACHER]"
        if s_name == str(current_speaker_name).strip():
            return "[CURRENT]"
        return "[OTHER]"
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Target text
        target_text = f"[CURRENT] {sample['sentence']}"
        
        # Build context
        context_text = ""
        if self.use_context and sample['context_sentences']:
            context_parts = []
            for turn_idx, (ctx_sentence, ctx_speaker) in enumerate(
                zip(sample['context_sentences'], sample['context_speakers'])
            ):
                role = self._get_role_name(ctx_speaker, sample['speaker'])
                if self.use_turn_indicators:
                    turn_marker = f"[TURN_{turn_idx}]"
                    context_parts.append(f"{turn_marker} {role} {ctx_sentence}")
                else:
                    context_parts.append(f"{role} {ctx_sentence}")
            context_text = " ".join(context_parts)
        
        # Tokenize
        if context_text:
            encoding = self.tokenizer(
                text=context_text,
                text_pair=target_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        else:
            encoding = self.tokenizer(
                text=target_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(sample['opinion_label'], dtype=torch.long),
        }

class PLMClassifier(nn.Module):
    """
    PLM Baseline Classifier
    Supports BERT, RoBERTa and other pretrained models
    """
    
    def __init__(self, model_path, num_classes=6, dropout=0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_path)
        self.hidden_size = self.encoder.config.hidden_size
        self.num_classes = num_classes
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            pooled_output = outputs.last_hidden_state[:, 0, :]
        
        logits = self.classifier(pooled_output)
        return logits

def train_epoch(model, dataloader, optimizer, scheduler, device, criterion):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # Statistics
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return avg_loss, accuracy, macro_f1


def evaluate(model, dataloader, device, criterion):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'predictions': all_preds,
        'labels': all_labels
    }

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from utils import EarlyStopping  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description='PLM Baseline Model Training')
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['bert', 'roberta', 'deberta'],
                        help='Model type')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Pretrained model path')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Data directory')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=12)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--context_window', type=int, default=5)
    parser.add_argument('--random_seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = f'../baseline_{args.model_type}'
    
    config = {
        'model_type': args.model_type,
        'model_path': args.model_path,
        'data_dir': args.data_dir,
        'output_dir': args.output_dir,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'dropout': args.dropout,
        'max_length': args.max_length,
        'context_window': args.context_window,
        'random_seed': args.random_seed,
        'num_classes': 6
    }
    
    print("=" * 80)
    print(f"PLM Baseline Training - {args.model_type.upper()}")
    print("=" * 80)
    print(json.dumps(config, indent=2, ensure_ascii=False))
    
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Save config
    with open(os.path.join(config['output_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    # Set random seed
    import random
    random.seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    torch.manual_seed(config['random_seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['random_seed'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load tokenizer
    print(f"\nLoading Tokenizer: {config['model_path']} ...")
    tokenizer = AutoTokenizer.from_pretrained(config['model_path'])
    
    # Add special tokens
    special_tokens = ['[TEACHER]', '[CURRENT]', '[OTHER]']
    special_tokens += [f'[TURN_{i}]' for i in range(config['context_window'] + 2)]
    num_added = tokenizer.add_tokens(special_tokens)
    print(f"Registered {num_added} new tokens")
    
    # Load datasets
    train_dataset = OpinionEvolutionDataset(
        os.path.join(config['data_dir'], 'train.csv'),
        tokenizer,
        max_length=config['max_length'],
        context_window=config['context_window']
    )
    
    val_dataset = OpinionEvolutionDataset(
        os.path.join(config['data_dir'], 'val.csv'),
        tokenizer,
        max_length=config['max_length'],
        context_window=config['context_window']
    )
    
    test_dataset = OpinionEvolutionDataset(
        os.path.join(config['data_dir'], 'test.csv'),
        tokenizer,
        max_length=config['max_length'],
        context_window=config['context_window']
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Initialize model
    print(f"\nInitializing {config['model_type'].upper()} model...")
    model = PLMClassifier(
        model_path=config['model_path'],
        num_classes=config['num_classes'],
        dropout=config['dropout']
    )
    
    # Resize embeddings for new tokens
    model.encoder.resize_token_embeddings(len(tokenizer))
    model.to(device)
    
    # Class weights (for imbalanced data)
    opinion_class_weights = torch.tensor([0.8, 1.2, 1.0, 3.0, 1.5, 2.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=opinion_class_weights)
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.01)
    
    total_steps = len(train_loader) * config['num_epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=5, min_delta=0.005)
    
    # Training
    print("\nStarting training...")
    best_val_f1 = 0
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, scheduler, device, criterion
        )
        
        val_results = evaluate(model, val_loader, device, criterion)
        
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val   - Loss: {val_results['loss']:.4f}, Acc: {val_results['accuracy']:.4f}, F1: {val_results['macro_f1']:.4f}")
        
        # Save best model
        if val_results['macro_f1'] > best_val_f1:
            best_val_f1 = val_results['macro_f1']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_macro_f1': val_results['macro_f1'],
                'config': config
            }
            torch.save(checkpoint, os.path.join(config['output_dir'], 'best_model.pth'))
            print(f"✓ Saved best model (Val F1: {val_results['macro_f1']:.4f})")
        
        # Early stopping
        if early_stopping(val_results['macro_f1']):
            print(f"Early stopping triggered! Best F1: {early_stopping.best_score:.4f}")
            break
    
    # Test set evaluation
    print("\n" + "=" * 80)
    print("Test Set Evaluation")
    print("=" * 80)
    
    checkpoint = torch.load(os.path.join(config['output_dir'], 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_results = evaluate(model, test_loader, device, criterion)
    
    print(f"\nTest results:")
    print(f"  Accuracy: {test_results['accuracy']:.4f}")
    print(f"  Macro-F1: {test_results['macro_f1']:.4f}")
    
    # Generate classification report
    label_names = ['Irrelevant', 'New', 'Strengthened', 'Weakened', 'Adopted', 'Refuted']
    report = classification_report(
        test_results['labels'],
        test_results['predictions'],
        target_names=label_names,
        digits=4
    )
    
    print("\nDetailed classification report:")
    print(report)
    
    # Save report
    with open(os.path.join(config['output_dir'], 'test_report.txt'), 'w') as f:
        f.write(f"Model type: {config['model_type'].upper()}\n")
        f.write(f"Accuracy: {test_results['accuracy']:.4f}\n")
        f.write(f"Macro-F1: {test_results['macro_f1']:.4f}\n\n")
        f.write(report)
    
    # Confusion matrix
    cm = confusion_matrix(test_results['labels'], test_results['predictions'])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names)
    plt.title(f'{config["model_type"].upper()} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(config['output_dir'], 'confusion_matrix.png'), dpi=300)
    print(f"\n✓ Confusion matrix saved")
    
    print("\nTraining complete!")
    print(f"Model saved to: {config['output_dir']}")


if __name__ == '__main__':
    main()
