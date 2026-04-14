#!/usr/bin/env python3
"""
Multi-task model training script
Jointly train DA recognition and OE classification
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
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import warnings
import transformers

warnings.filterwarnings('ignore')
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

torch.backends.cudnn.enabled = False  # Disabled for deterministic reproducibility

from multi_task_model import MultiTaskDialogueModel, MultiTaskLoss


from error_analysis import analyze_errors, print_error_stats

class MultiTaskDialogueDataset(Dataset):
    """
    Multi-task dataset: provides both dialogue act labels and opinion evolution labels
    """
    
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
        print(f"Config: Window={context_window}, TurnIndicators={use_turn_indicators}")
        
        df = pd.read_csv(csv_file)

        required_cols = ['Sentence', 'Speaker', 'act_label', 'label']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}. Current columns: {df.columns.tolist()}")
        
        # Group by classroom
        grouped = df.groupby('class', sort=False)
        
        self.samples = []
        self.all_dialogue_act_labels = []
        self.all_opinion_labels = []
        
        for class_id, group in grouped:
            sentences = group['Sentence'].tolist()
            speakers = group['Speaker'].tolist()
            dialogue_act_tags = group['act_label'].tolist()
            classifications = group['label'].tolist() 
            
            for i in range(len(sentences)):
                start_idx = max(0, i - context_window)
                context_sentences = sentences[start_idx:i]
                context_speakers = speakers[start_idx:i]
                
                sample = {
                    'sentence': sentences[i],
                    'speaker': speakers[i],
                    'dialogue_act_label': dialogue_act_tags[i],
                    'opinion_label': classifications[i],
                    'context_sentences': context_sentences,
                    'context_speakers': context_speakers,
                    'class_id': class_id
                }
                
                self.samples.append(sample)
                self.all_dialogue_act_labels.append(dialogue_act_tags[i])
                self.all_opinion_labels.append(classifications[i])
        
        print(f"Loaded. Valid samples: {len(self.samples)}")

        print(f"\nDialogue act label distribution:")
        unique_acts, counts_acts = np.unique(self.all_dialogue_act_labels, return_counts=True)
        for act, count in zip(unique_acts, counts_acts):
            print(f"  Label {act}: {count} samples")

        print(f"\nOpinion evolution label distribution:")
        unique_opinions, counts_opinions = np.unique(self.all_opinion_labels, return_counts=True)
        for opinion, count in zip(unique_opinions, counts_opinions):
            print(f"  Label {opinion}: {count} samples")
    
    def _get_role_name(self, speaker_name, current_speaker_name):
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
   
        target_text = f"[CURRENT] {sample['sentence']}"

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
            'dialogue_act_label': torch.tensor(sample['dialogue_act_label'], dtype=torch.long),
            'opinion_label': torch.tensor(sample['opinion_label'], dtype=torch.long),
        }

def train_epoch(model, dataloader, optimizer, scheduler, device, criterion, accum_steps=1, is_multi_task=True):
    model.train()
    total_loss = 0
    total_dialogue_act_loss = 0
    total_opinion_loss = 0
    
    dialogue_act_preds = []
    dialogue_act_labels = []
    opinion_preds = []
    opinion_labels = []
    
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc="Training")
    for step, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        dialogue_act_label = batch['dialogue_act_label'].to(device)
        opinion_label = batch['opinion_label'].to(device)
        
        if is_multi_task:
            outputs = model(input_ids, attention_mask)
            
            loss_dict = criterion(
                outputs['dialogue_act_logits'],
                outputs['opinion_logits'],
                dialogue_act_label,
                opinion_label
            )
            
            loss = loss_dict['total_loss'] / accum_steps
    
            dialogue_act_pred = outputs['dialogue_act_logits'].argmax(dim=-1)
            opinion_pred = outputs['opinion_logits'].argmax(dim=-1)
            
            dialogue_act_preds.extend(dialogue_act_pred.cpu().tolist())
            dialogue_act_labels.extend(dialogue_act_label.cpu().tolist())
            total_dialogue_act_loss += loss_dict['dialogue_act_loss'].item()
            total_opinion_loss += loss_dict['opinion_loss'].item()
        else:
            logits = model(input_ids, attention_mask)
            
            loss_fn = nn.CrossEntropyLoss(weight=criterion.opinion_class_weights if hasattr(criterion, 'opinion_class_weights') else None)
            loss = loss_fn(logits, opinion_label) / accum_steps
            opinion_pred = logits.argmax(dim=-1)
            
            loss_dict = {
                'total_loss': loss * accum_steps,
                'dialogue_act_loss': torch.tensor(0.0),
                'opinion_loss': loss * accum_steps
            }
        
        loss.backward()

        if (step + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss_dict['total_loss'].item()
        opinion_preds.extend(opinion_pred.cpu().tolist())
        opinion_labels.extend(opinion_label.cpu().tolist())
        
        if is_multi_task:
            pbar.set_postfix({
                'loss': f"{loss_dict['total_loss'].item():.4f}",
                'act_loss': f"{loss_dict['dialogue_act_loss'].item():.4f}",
                'op_loss': f"{loss_dict['opinion_loss'].item():.4f}"
            })
        else:
            pbar.set_postfix({
                'loss': f"{loss_dict['total_loss'].item():.4f}",
                'op_loss': f"{loss_dict['opinion_loss'].item():.4f}"
            })

    avg_loss = total_loss / len(dataloader)
    
    if is_multi_task:
        act_acc = accuracy_score(dialogue_act_labels, dialogue_act_preds)
        act_f1 = f1_score(dialogue_act_labels, dialogue_act_preds, average='macro')
    else:
        act_acc = 0.0
        act_f1 = 0.0
    
    op_acc = accuracy_score(opinion_labels, opinion_preds)
    op_f1 = f1_score(opinion_labels, opinion_preds, average='macro')
    
    return avg_loss, act_acc, act_f1, op_acc, op_f1


def evaluate(model, dataloader, device, criterion, is_multi_task=True):
    model.eval()
    total_loss = 0
    
    dialogue_act_preds = []
    dialogue_act_labels = []
    opinion_preds = []
    opinion_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            dialogue_act_label = batch['dialogue_act_label'].to(device)
            opinion_label = batch['opinion_label'].to(device)
            
            if is_multi_task:
                outputs = model(input_ids, attention_mask)
                
                loss_dict = criterion(
                    outputs['dialogue_act_logits'],
                    outputs['opinion_logits'],
                    dialogue_act_label,
                    opinion_label
                )
                
                dialogue_act_pred = outputs['dialogue_act_logits'].argmax(dim=-1)
                opinion_pred = outputs['opinion_logits'].argmax(dim=-1)
                
                dialogue_act_preds.extend(dialogue_act_pred.cpu().tolist())
                dialogue_act_labels.extend(dialogue_act_label.cpu().tolist())
            else:
                logits = model(input_ids, attention_mask)
                
                loss_fn = nn.CrossEntropyLoss(weight=criterion.opinion_class_weights if hasattr(criterion, 'opinion_class_weights') else None)
                loss = loss_fn(logits, opinion_label)
                
                loss_dict = {
                    'total_loss': loss,
                    'dialogue_act_loss': torch.tensor(0.0),
                    'opinion_loss': loss
                }
                
                opinion_pred = logits.argmax(dim=-1)
            
            total_loss += loss_dict['total_loss'].item()
            opinion_preds.extend(opinion_pred.cpu().tolist())
            opinion_labels.extend(opinion_label.cpu().tolist())
    
    avg_loss = total_loss / len(dataloader)
    
    if is_multi_task:
        act_acc = accuracy_score(dialogue_act_labels, dialogue_act_preds)
        act_f1 = f1_score(dialogue_act_labels, dialogue_act_preds, average='macro', zero_division=0)
    else:
        act_acc = 0.0
        act_f1 = 0.0
    
    op_acc = accuracy_score(opinion_labels, opinion_preds)
    op_f1 = f1_score(opinion_labels, opinion_preds, average='macro', zero_division=0)
    
    return {
        'loss': avg_loss,
        'dialogue_act_accuracy': act_acc,
        'dialogue_act_macro_f1': act_f1,
        'opinion_accuracy': op_acc,
        'opinion_macro_f1': op_f1,
        'dialogue_act_predictions': dialogue_act_preds,
        'dialogue_act_labels': dialogue_act_labels,
        'opinion_predictions': opinion_preds,
        'opinion_labels': opinion_labels
    }

from utils import EarlyStopping  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description='Multi-task model training')
    parser.add_argument('--task_type', type=str, default='multi',
                        choices=['single', 'multi'],
                        help='Task type: single(single-task OE only) / multi(multi-task DA+OE)')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Data directory')
    parser.add_argument('--model_type', type=str, default='deberta',
                        choices=['deberta', 'roberta', 'bert', 'llama', 'qwen'],
                        help='Pretrained model type')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Pretrained model path (uses default if not specified)')
    parser.add_argument('--num_dialogue_acts', type=int, default=12,
                        help='Number of dialogue act classes')
    parser.add_argument('--num_opinion_classes', type=int, default=6,
                        help='Number of opinion classes')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=15)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--dropout', type=float, default=None,
                        help='Dropout ratio (auto-adaptive based on sample_ratio by default)')
    parser.add_argument('--use_class_balance', action='store_true',
                        help='Enable class-balanced sampling (oversample minority classes)')
    parser.add_argument('--balance_target', type=str, default='opinion',
                        choices=['opinion', 'dialogue_act', 'both'],
                        help='Balance target: opinion/dialogue_act/both')
    parser.add_argument('--sample_ratio', type=float, default=1.0,
                        help='Training data sample ratio (0.0-1.0)')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (auto-generated by default)')
    
    args = parser.parse_args()
    
    model_configs = {
        'deberta': {
            'path': 'microsoft/deberta-v3-base',
            'learning_rate': 2e-5,
            'batch_size': 16,
            'params': '184M'
        },
        'roberta': {
            'path': 'roberta-large',
            'learning_rate': 1e-5,
            'batch_size': 8,
            'params': '355M'
        },
        'bert': {
            'path': 'bert-base-uncased',
            'learning_rate': 2e-5,
            'batch_size': 16,
            'params': '110M'
        }
    }
    
    if args.model_path is None:
        args.model_path = model_configs[args.model_type]['path']
        print(f"Using default model path: {args.model_path}")

    if args.learning_rate == 2e-5 and args.model_type == 'roberta':
        args.learning_rate = model_configs['roberta']['learning_rate']
        print(f"RoBERTa auto-adjusting learning rate: {args.learning_rate}")

    if args.batch_size == 16 and args.model_type == 'roberta':
        args.batch_size = model_configs['roberta']['batch_size']
        print(f"RoBERTa auto-adjusting batch_size: {args.batch_size}")
    
    model_name = args.model_path
    
    is_multi_task = (args.task_type == 'multi')
    if args.output_dir is None:
        task_suffix = 'multitask' if is_multi_task else 'singletask'
        args.output_dir = f"../discriminative_model_{task_suffix}"
        print(f"Output directory: {args.output_dir}")
    

    if args.sample_ratio >= 0.8:  
        recommended_lr = 1e-5
        recommended_dropout = 0.3
        recommended_epochs = 12
    else:
        recommended_lr = 2e-5
        recommended_dropout = 0.6
        recommended_epochs = 15
    if args.learning_rate == 2e-5 and args.sample_ratio >= 0.8:
        args.learning_rate = recommended_lr
        print(f"Full data training detected, auto-adjusting learning rate: {args.learning_rate}")
    
    if args.dropout is None:
        args.dropout = recommended_dropout
        print(f"Auto-setting dropout: {args.dropout} (sample_ratio={args.sample_ratio})")
    
    if args.num_epochs == 15 and args.sample_ratio >= 0.8:
        args.num_epochs = recommended_epochs
        print(f"Full data training detected, auto-adjusting epoch count: {args.num_epochs}")
    
    config = {
        'data_dir': args.data_dir,
        'model_type': args.model_type,
        'model_name': model_name,
        'is_multi_task': is_multi_task,
        'num_dialogue_acts': args.num_dialogue_acts,
        'num_opinion_classes': args.num_opinion_classes,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate, 
        'dropout': args.dropout, 
        'sample_ratio': args.sample_ratio,
        'random_seed': args.random_seed,
        'output_dir': args.output_dir,
        'max_length': 256,
        'use_context': True,
        'context_window': 5,
        'dialogue_act_weight': 0.4,
        'opinion_weight': 0.6,
        'use_class_balance': args.use_class_balance,
        'balance_target': args.balance_target,
        'use_act_logits_in_opinion': False
    }
    
    task_name = "Multi-task" if is_multi_task else "Single-task"
    print("=" * 80)
    print(f"{task_name} model training - {int(args.sample_ratio*100)}% training data")
    print("=" * 80)
    print(json.dumps(config, indent=2, ensure_ascii=False))
    
    os.makedirs(config['output_dir'], exist_ok=True)
    with open(os.path.join(config['output_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    import random
    random.seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    torch.manual_seed(config['random_seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['random_seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load tokenizer
    print(f"\nLoading Tokenizer: {config['model_type'].upper()} ({config['model_name']}) ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    
    # Add special tokens
    special_tokens = ['[TEACHER]', '[CURRENT]', '[OTHER]', 'Unknown']
    special_tokens += [f'[TURN_{i}]' for i in range(config['context_window'] + 2)]
    num_added = tokenizer.add_tokens(special_tokens)
    print(f"Registered {num_added} new tokens")
    
    # Load datasets
    train_dataset = MultiTaskDialogueDataset(
        os.path.join(config['data_dir'], 'train.csv'),
        tokenizer,
        max_length=config['max_length'],
        use_context=config['use_context'],
        context_window=config['context_window']
    )
    
    val_dataset = MultiTaskDialogueDataset(
        os.path.join(config['data_dir'], 'val.csv'),
        tokenizer,
        max_length=config['max_length'],
        use_context=config['use_context'],
        context_window=config['context_window']
    )
    
    test_dataset = MultiTaskDialogueDataset(
        os.path.join(config['data_dir'], 'test.csv'),
        tokenizer,
        max_length=config['max_length'],
        use_context=config['use_context'],
        context_window=config['context_window']
    )
    
    if config['sample_ratio'] < 1.0:
        print(f"Few-shot sampling: using {int(config['sample_ratio']*100)}% training data")
        from sklearn.model_selection import train_test_split
        
        original_size = len(train_dataset.samples)
        all_indices = list(range(len(train_dataset.samples)))
        all_opinion_labels = train_dataset.all_opinion_labels
        selected_indices, _ = train_test_split(
            all_indices,
            train_size=config['sample_ratio'],
            stratify=all_opinion_labels,
            random_state=config['random_seed']
        )
        
        train_dataset.samples = [train_dataset.samples[i] for i in selected_indices]
        train_dataset.all_dialogue_act_labels = [train_dataset.all_dialogue_act_labels[i] for i in selected_indices]
        train_dataset.all_opinion_labels = [train_dataset.all_opinion_labels[i] for i in selected_indices]
        
        print(f"  Original samples: {original_size}")
        print(f"  Sampled samples: {len(train_dataset.samples)}")
        print(f"  Post-sampling label distribution:")
        unique_labels, counts = np.unique(train_dataset.all_opinion_labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            print(f"    Label {label}: {count} samples ({count/len(train_dataset.samples)*100:.1f}%)")
    
    train_sampler = None
    if config['use_class_balance']:
        from torch.utils.data import WeightedRandomSampler
        
        print(f"\nEnable class-balanced sampling (target: {config['balance_target']})")
        
 
        if config['balance_target'] == 'opinion':
            labels = [s['opinion_label'] for s in train_dataset.samples]
        elif config['balance_target'] == 'dialogue_act':
            labels = [s['dialogue_act_label'] for s in train_dataset.samples if s['dialogue_act_label'] != -1]
        else:
            labels = [s['opinion_label'] for s in train_dataset.samples]
        

        from collections import Counter
        label_counts = Counter(labels)
        total_count = len(labels)
        
        max_count = max(label_counts.values())
        class_weights = {label: max_count / count for label, count in label_counts.items()}

        if config['balance_target'] == 'opinion':
            sample_weights = [class_weights[s['opinion_label']] for s in train_dataset.samples]
        elif config['balance_target'] == 'dialogue_act':
            sample_weights = [
                class_weights[s['dialogue_act_label']] if s['dialogue_act_label'] != -1 else 1.0
                for s in train_dataset.samples
            ]
        else:
            sample_weights = [class_weights[s['opinion_label']] for s in train_dataset.samples]
        
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        print(f"  Class weights: {dict(sorted(class_weights.items()))}")
        print(f"  Pre-balance distribution: {dict(sorted(label_counts.items()))}")
        print(f"  Sampler created: {len(sample_weights)} samples")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        sampler=train_sampler,
        shuffle=(train_sampler is None) 
    )
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    

    is_multi_task = config.get('is_multi_task', True)
    
    if is_multi_task:
        print(f"\nInitializing multi-task model (DA + OE)...")
        model = MultiTaskDialogueModel(
            model_path=config['model_name'],
            num_dialogue_acts=config['num_dialogue_acts'],
            num_opinion_classes=config['num_opinion_classes'],
            dropout=config['dropout'],
            use_act_logits_in_opinion=config.get('use_act_logits_in_opinion', False)
        )
    else:
        print(f"\nInitializing single-task model (OE only)...")
        from hybrid_opinion_classifier import DialogueAwareModel
        model = DialogueAwareModel(
            model_path=config['model_name'],
            num_classes=config['num_opinion_classes'],
            dropout=config['dropout']
        )
    
    model.encoder.resize_token_embeddings(len(tokenizer))
    model.to(device)
    criterion = MultiTaskLoss(
        dialogue_act_weight=config['dialogue_act_weight'],
        opinion_weight=config['opinion_weight'],
        opinion_class_weights=opinion_class_weights,
        use_opinion_reweighting=True
    )
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.05)
    
    total_steps = len(train_loader) * config['num_epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    early_stopping = EarlyStopping(patience=5, min_delta=0.005, mode='max')
    print("\nStarting training...")
    best_val_opinion_f1 = 0
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
        train_loss, train_act_acc, train_act_f1, train_op_acc, train_op_f1 = train_epoch(
            model, train_loader, optimizer, scheduler, device, criterion, is_multi_task=config['is_multi_task']
        )
        
        val_results = evaluate(model, val_loader, device, criterion, is_multi_task=config['is_multi_task'])
        
        print(f"Train - Loss: {train_loss:.4f}, Act-Acc: {train_act_acc:.4f}, Act-F1: {train_act_f1:.4f}, "
              f"Op-Acc: {train_op_acc:.4f}, Op-F1: {train_op_f1:.4f}")
        print(f"Val   - Loss: {val_results['loss']:.4f}, Act-Acc: {val_results['dialogue_act_accuracy']:.4f}, "
              f"Act-F1: {val_results['dialogue_act_macro_f1']:.4f}, Op-Acc: {val_results['opinion_accuracy']:.4f}, "
              f"Op-F1: {val_results['opinion_macro_f1']:.4f}")
        
       
        if val_results['opinion_macro_f1'] > best_val_opinion_f1:
            best_val_opinion_f1 = val_results['opinion_macro_f1']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_opinion_macro_f1': val_results['opinion_macro_f1'],
                'val_dialogue_act_macro_f1': val_results['dialogue_act_macro_f1'],
                'config': config
            }
            torch.save(checkpoint, os.path.join(config['output_dir'], 'best_model.pth'))
            print(f"✓ Saved best model (Val Opinion-F1: {val_results['opinion_macro_f1']:.4f})")
        
        if early_stopping(val_results['opinion_macro_f1']):
            print(f"Early stopping triggered! Best Opinion F1: {early_stopping.best_score:.4f}")
            break
    

    print("\n" + "=" * 80)
    print("Test Set Evaluation")
    print("=" * 80)
    
    checkpoint = torch.load(os.path.join(config['output_dir'], 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_results = evaluate(model, test_loader, device, criterion, is_multi_task=config['is_multi_task'])
    
    print(f"\nTest results:")
    print(f"  Dialogue act - Acc: {test_results['dialogue_act_accuracy']:.4f}, Macro-F1: {test_results['dialogue_act_macro_f1']:.4f}")
    print(f"  Opinion evolution - Acc: {test_results['opinion_accuracy']:.4f}, Macro-F1: {test_results['opinion_macro_f1']:.4f}")
    
   
    print("\n" + "=" * 80)
    print("Generating Confusion Matrix Visualization")
    print("=" * 80)
    

    opinion_labels = ['Irrelevant', 'New', 'Strengthened', 'Weakened', 'Adopted', 'Refuted']
    opinion_cm = confusion_matrix(test_results['opinion_labels'], test_results['opinion_predictions'])
    

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        opinion_cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=opinion_labels,
        yticklabels=opinion_labels,
        cbar_kws={'label': 'Count'},
        linewidths=0.5,
        linecolor='gray'
    )
    plt.title('Opinion Evolution Confusion Matrix (Absolute Counts)', fontsize=16, pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    cm_count_path = os.path.join(config['output_dir'], 'confusion_matrix_opinion_count.png')
    plt.savefig(cm_count_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Opinion evolution confusion matrix (counts) saved: {cm_count_path}")
    plt.close()
    
    opinion_cm_normalized = opinion_cm.astype('float') / opinion_cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        opinion_cm_normalized, 
        annot=True, 
        fmt='.2%', 
        cmap='YlOrRd', 
        xticklabels=opinion_labels,
        yticklabels=opinion_labels,
        cbar_kws={'label': 'Proportion'},
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor='gray'
    )
    plt.title('Opinion Evolution Confusion Matrix (Normalized)', fontsize=16, pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    cm_norm_path = os.path.join(config['output_dir'], 'confusion_matrix_opinion_normalized.png')
    plt.savefig(cm_norm_path, dpi=300, bbox_inches='tight')
    print(f"✓ Opinion evolution confusion matrix (normalized) saved: {cm_norm_path}")
    plt.close()
    
    dialogue_act_labels = [
        'None (Teacher)',
        'Keeping Together',
        'Relate',
        'Restating',
        'Revoicing',
        'Press Accuracy',
        'Press Reasoning',
        'None (Student)',
        'Relating to Another Student',
        'Asking for Info',
        'Making a Claim',
        'Providing Evidence/Reasoning'
    ]
    dialogue_act_cm = confusion_matrix(test_results['dialogue_act_labels'], test_results['dialogue_act_predictions'])
    
    plt.figure(figsize=(16, 14))
    sns.heatmap(
        dialogue_act_cm, 
        annot=True, 
        fmt='d', 
        cmap='Greens', 
        xticklabels=dialogue_act_labels,
        yticklabels=dialogue_act_labels,
        cbar_kws={'label': 'Count'},
        linewidths=0.5,
        linecolor='gray'
    )
    plt.title('Dialogue Act Confusion Matrix (Absolute Counts)', fontsize=16, pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    cm_act_count_path = os.path.join(config['output_dir'], 'confusion_matrix_dialogue_act_count.png')
    plt.savefig(cm_act_count_path, dpi=300, bbox_inches='tight')
    print(f"✓ Dialogue act confusion matrix (counts) saved: {cm_act_count_path}")
    plt.close()
    
    dialogue_act_cm_normalized = dialogue_act_cm.astype('float') / dialogue_act_cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(16, 14))
    sns.heatmap(
        dialogue_act_cm_normalized, 
        annot=True, 
        fmt='.2%', 
        cmap='YlGn', 
        xticklabels=dialogue_act_labels,
        yticklabels=dialogue_act_labels,
        cbar_kws={'label': 'Proportion'},
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor='gray'
    )
    plt.title('Dialogue Act Confusion Matrix (Normalized)', fontsize=16, pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    cm_act_norm_path = os.path.join(config['output_dir'], 'confusion_matrix_dialogue_act_normalized.png')
    plt.savefig(cm_act_norm_path, dpi=300, bbox_inches='tight')
    print(f"✓ Dialogue act confusion matrix (normalized) saved: {cm_act_norm_path}")
    plt.close()
    
    print("\n" + "=" * 80)
    print("Generating Error Analysis Report")
    print("=" * 80)
    

    test_csv_path = os.path.join(config['data_dir'], 'test.csv')
    test_df = pd.read_csv(test_csv_path)
    

    error_csv_path = os.path.join(config['output_dir'], 'error_analysis_test.csv')
    error_stats = analyze_errors(
        predictions=test_results['opinion_predictions'],
        labels=test_results['opinion_labels'],
        dialogue_act_preds=test_results['dialogue_act_predictions'],
        dialogue_act_labels=test_results['dialogue_act_labels'],
        data_df=test_df,
        output_path=error_csv_path,
        split_name='test'
    )
    
    print_error_stats(error_stats)
    print(f"\n✓ Error analysis report saved: {error_csv_path}")
    
    print("\nTraining complete!")
    print(f"Model saved at: {config['output_dir']}")


if __name__ == '__main__':
    main()
