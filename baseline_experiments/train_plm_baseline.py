#!/usr/bin/env python3
"""
PLM基线模型训练脚本 (Group A)

支持的模型:
- BERT-base
- RoBERTa-large

使用与混合模型相同的Prompt-tuning式输入 ([TEACHER], [CURRENT] 等)
全量Fine-tuning，单任务观点演化分类
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

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 添加父目录到路径以导入数据集类
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# 1. 数据集（复用多任务数据集，但只使用OE标签）
# =============================================================================

class OpinionEvolutionDataset(Dataset):
    """
    观点演化数据集 (单任务版本)
    使用与混合模型相同的Prompt-tuning输入格式
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
        
        # 教师名字集合
        self.teacher_names = {'T', 'Teacher', 'Ms. G', 'Mrs. G'}
        
        # 读取数据
        print(f"正在加载数据: {csv_file} ...")
        df = pd.read_csv(csv_file)
        
        # 检查必要的列
        required_cols = ['Sentence', 'Speaker', 'label']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"CSV缺少必要的列: {missing_cols}。当前列: {df.columns.tolist()}")
        
        # 按课堂分组
        grouped = df.groupby('class', sort=False)
        
        self.samples = []
        self.all_opinion_labels = []
        
        for class_id, group in grouped:
            sentences = group['Sentence'].tolist()
            speakers = group['Speaker'].tolist()
            classifications = group['label'].tolist()
            
            for i in range(len(sentences)):
                # 提取上下文
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
        
        print(f"加载完成。有效样本数: {len(self.samples)}")
        
        # 统计标签分布
        print(f"\n观点演化标签分布:")
        unique_labels, counts = np.unique(self.all_opinion_labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            label_name = ['Irrelevant', 'New', 'Strengthened', 'Weakened', 'Adopted', 'Refuted'][label]
            print(f"  {label} ({label_name}): {count} 样本")
    
    def _get_role_name(self, speaker_name, current_speaker_name):
        """获取相对角色标记"""
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
        
        # 目标文本
        target_text = f"[CURRENT] {sample['sentence']}"
        
        # 构造上下文
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
        
        # 分词
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


# =============================================================================
# 2. PLM基线模型
# =============================================================================

class PLMClassifier(nn.Module):
    """
    PLM基线分类器
    支持 BERT, RoBERTa 等预训练模型
    """
    
    def __init__(self, model_path, num_classes=6, dropout=0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_path)
        self.hidden_size = self.encoder.config.hidden_size
        self.num_classes = num_classes
        
        # 分类头
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
        
        # 兼容不同模型的输出格式
        # BERT/RoBERTa有pooler_output，DeBERTa没有
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            # DeBERTa等模型：使用[CLS] token (第一个token)
            pooled_output = outputs.last_hidden_state[:, 0, :]
        
        logits = self.classifier(pooled_output)
        return logits


# =============================================================================
# 3. 训练和评估函数
# =============================================================================

def train_epoch(model, dataloader, optimizer, scheduler, device, criterion):
    """训练一个epoch"""
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
        
        # 前向传播
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # 统计
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
    """评估模型"""
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


# =============================================================================
# 4. 早停
# =============================================================================

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if score > (self.best_score + self.min_delta):
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False


# =============================================================================
# 5. 主训练流程
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='PLM基线模型训练')
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['bert', 'roberta', 'deberta'],
                        help='模型类型')
    parser.add_argument('--model_path', type=str, required=True,
                        help='预训练模型路径')
    parser.add_argument('--data_dir', type=str, default='dataset_split_result_v4',
                        help='数据目录')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=12)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--context_window', type=int, default=5)
    parser.add_argument('--random_seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # 设置输出目录
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
    print(f"PLM基线训练 - {args.model_type.upper()}")
    print("=" * 80)
    print(json.dumps(config, indent=2, ensure_ascii=False))
    
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # 保存配置
    with open(os.path.join(config['output_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    # 设置随机种子
    import random
    random.seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    torch.manual_seed(config['random_seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['random_seed'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 加载tokenizer
    print(f"\n加载 Tokenizer: {config['model_path']} ...")
    tokenizer = AutoTokenizer.from_pretrained(config['model_path'])
    
    # 添加special tokens
    special_tokens = ['[TEACHER]', '[CURRENT]', '[OTHER]']
    special_tokens += [f'[TURN_{i}]' for i in range(config['context_window'] + 2)]
    num_added = tokenizer.add_tokens(special_tokens)
    print(f"已注册 {num_added} 个新 Token")
    
    # 加载数据集
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
    
    # 初始化模型
    print(f"\n初始化 {config['model_type'].upper()} 模型...")
    model = PLMClassifier(
        model_path=config['model_path'],
        num_classes=config['num_classes'],
        dropout=config['dropout']
    )
    
    # 调整embedding大小以适应新token
    model.encoder.resize_token_embeddings(len(tokenizer))
    model.to(device)
    
    # 类别权重（针对不平衡数据）
    opinion_class_weights = torch.tensor([0.8, 1.2, 1.0, 3.0, 1.5, 2.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=opinion_class_weights)
    
    # 优化器
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.01)
    
    total_steps = len(train_loader) * config['num_epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # 早停
    early_stopping = EarlyStopping(patience=5, min_delta=0.005)
    
    # 训练
    print("\n开始训练...")
    best_val_f1 = 0
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, scheduler, device, criterion
        )
        
        val_results = evaluate(model, val_loader, device, criterion)
        
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val   - Loss: {val_results['loss']:.4f}, Acc: {val_results['accuracy']:.4f}, F1: {val_results['macro_f1']:.4f}")
        
        # 保存最佳模型
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
            print(f"✓ 保存最佳模型 (Val F1: {val_results['macro_f1']:.4f})")
        
        # 早停
        if early_stopping(val_results['macro_f1']):
            print(f"早停触发！最佳 F1: {early_stopping.best_score:.4f}")
            break
    
    # 测试集评估
    print("\n" + "=" * 80)
    print("测试集评估")
    print("=" * 80)
    
    checkpoint = torch.load(os.path.join(config['output_dir'], 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_results = evaluate(model, test_loader, device, criterion)
    
    print(f"\n测试集结果:")
    print(f"  准确率: {test_results['accuracy']:.4f}")
    print(f"  Macro-F1: {test_results['macro_f1']:.4f}")
    
    # 生成分类报告
    label_names = ['Irrelevant', 'New', 'Strengthened', 'Weakened', 'Adopted', 'Refuted']
    report = classification_report(
        test_results['labels'],
        test_results['predictions'],
        target_names=label_names,
        digits=4
    )
    
    print("\n详细分类报告:")
    print(report)
    
    # 保存报告
    with open(os.path.join(config['output_dir'], 'test_report.txt'), 'w') as f:
        f.write(f"模型类型: {config['model_type'].upper()}\n")
        f.write(f"准确率: {test_results['accuracy']:.4f}\n")
        f.write(f"Macro-F1: {test_results['macro_f1']:.4f}\n\n")
        f.write(report)
    
    # 混淆矩阵
    cm = confusion_matrix(test_results['labels'], test_results['predictions'])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names)
    plt.title(f'{config["model_type"].upper()} - 混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.savefig(os.path.join(config['output_dir'], 'confusion_matrix.png'), dpi=300)
    print(f"\n✓ 混淆矩阵已保存")
    
    print("\n训练完成！")
    print(f"模型保存位置: {config['output_dir']}")


if __name__ == '__main__':
    main()
