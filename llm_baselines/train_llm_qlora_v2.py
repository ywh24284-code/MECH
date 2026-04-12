#!/usr/bin/env python3
"""
LLM QLoRA微调脚本 v2（改进版）

主要改进：
1. 数据重采样：过采样少数类 + 欠采样多数类
2. Few-shot示例：在Prompt中添加分层采样的示例
3. 调整超参数：降低学习率，增加warmup
4. 改进评估：更鲁棒的标签解析

解决v1的类别崩溃问题（只预测Irrelevant）
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
from collections import Counter
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# 1. 数据集（改进版：支持重采样和Few-shot）
# =============================================================================

class OpinionEvolutionLLMDatasetV2(Dataset):
    """改进的LLM微调数据集"""
    
    def __init__(
        self,
        csv_file: str,
        tokenizer,
        max_length: int = 1024,
        context_window: int = 5,
        model_type: str = 'llama',
        use_resampling: bool = True,
        use_few_shot: bool = True,
        few_shot_examples: list = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.context_window = context_window
        self.model_type = model_type
        self.use_few_shot = use_few_shot
        self.few_shot_examples = few_shot_examples or []
        
        self.teacher_names = {'T', 'Teacher', 'Ms. G', 'Mrs. G'}
        self.label_names = ['Irrelevant', 'New', 'Strengthened', 'Weakened', 'Adopted', 'Refuted']
        
        print(f"正在加载数据: {csv_file} ...")
        df = pd.read_csv(csv_file)
        
        # 按课堂分组处理
        grouped = df.groupby('class', sort=False)
        samples_by_label = {i: [] for i in range(6)}
        
        for class_id, group in grouped:
            sentences = group['Sentence'].tolist()
            speakers = group['Speaker'].tolist()
            classifications = group['label'].tolist()
            
            for i in range(len(sentences)):
                start_idx = max(0, i - context_window)
                context_sentences = sentences[start_idx:i]
                context_speakers = speakers[start_idx:i]
                
                prompt = self._build_prompt(
                    current_sentence=sentences[i],
                    current_speaker=speakers[i],
                    context_sentences=context_sentences,
                    context_speakers=context_speakers
                )
                
                label = self.label_names[classifications[i]]
                
                sample = {
                    'prompt': prompt,
                    'label': label,
                    'label_id': classifications[i]
                }
                
                samples_by_label[classifications[i]].append(sample)
        
        # === 数据重采样策略 ===
        if use_resampling and 'train' in csv_file:
            print("\n🔄 应用数据重采样策略...")
            self.samples = self._resample_data(samples_by_label)
        else:
            # 验证集/测试集不重采样
            self.samples = []
            for label_samples in samples_by_label.values():
                self.samples.extend(label_samples)
        
        print(f"加载完成。有效样本数: {len(self.samples)}")
        
        # 统计标签分布
        label_ids = [s['label_id'] for s in self.samples]
        print(f"\n观点演化标签分布:")
        unique_labels, counts = np.unique(label_ids, return_counts=True)
        for label, count in zip(unique_labels, counts):
            print(f"  {label} ({self.label_names[label]}): {count} 样本 ({100*count/len(self.samples):.1f}%)")
    
    def _resample_data(self, samples_by_label):
        """重采样策略：平衡类别分布"""
        
        # 原始分布
        original_counts = {i: len(samples) for i, samples in samples_by_label.items()}
        print(f"  原始分布: {dict(sorted(original_counts.items()))}")
        
        # 目标分布（降低不平衡比例从25:1到约3:1）
        target_distribution = {
            0: 3000,   # Irrelevant: 7649 → 3000（欠采样）
            1: 1307,   # New: 保持
            2: 1618,   # Strengthened: 保持
            3: 1000,   # Weakened: 295 → 1000（过采样3.4倍）
            4: 1000,   # Adopted: 739 → 1000（过采样1.4倍）
            5: 800     # Refuted: 370 → 800（过采样2.2倍）
        }
        
        resampled_samples = []
        
        for label_id in range(6):
            original = samples_by_label[label_id]
            target = target_distribution[label_id]
            
            if len(original) >= target:
                # 欠采样：随机选择
                selected = np.random.choice(len(original), target, replace=False)
                resampled_samples.extend([original[i] for i in selected])
            else:
                # 过采样：重复采样
                selected = np.random.choice(len(original), target, replace=True)
                resampled_samples.extend([original[i] for i in selected])
        
        # 打乱顺序
        np.random.shuffle(resampled_samples)
        
        # 新分布
        new_counts = Counter([s['label_id'] for s in resampled_samples])
        print(f"  重采样后: {dict(sorted(new_counts.items()))}")
        print(f"  样本总数: {len(original_counts)} → {len(resampled_samples)}")
        
        return resampled_samples
    
    def _get_switch_marker(self, prev_speaker, current_speaker):
        """Switch/Same标记"""
        if not prev_speaker:
            return ""
        
        prev = str(prev_speaker).strip()
        curr = str(current_speaker).strip()
        
        prev_is_teacher = prev.startswith('T') or prev in self.teacher_names
        curr_is_teacher = curr.startswith('T') or curr in self.teacher_names
        
        if prev_is_teacher != curr_is_teacher:
            return "[Switch]"
        elif prev == curr:
            return "[Same]"
        else:
            return "[Switch]"
    
    def _build_prompt(self, current_sentence, current_speaker, 
                      context_sentences, context_speakers):
        """构造Prompt（支持Few-shot）"""
        
        # Few-shot示例部分
        few_shot_text = ""
        if self.use_few_shot and self.few_shot_examples:
            few_shot_text = "Here are some examples:\n\n"
            for i, ex in enumerate(self.few_shot_examples[:6], 1):  # 最多6个示例
                few_shot_text += f"Example {i}:\n"
                few_shot_text += f"Context: {ex['context']}\n"
                few_shot_text += f"Current: {ex['current']}\n"
                few_shot_text += f"Label: {ex['label']}\n\n"
            few_shot_text += "---\n\n"
        
        # 对话历史
        dialogue = ""
        if context_sentences:
            for ctx_speaker, ctx_sentence in zip(context_speakers, context_sentences):
                dialogue += f"{ctx_speaker}: {ctx_sentence}\n"
        
        # Switch/Same标记
        switch_marker = ""
        if context_speakers:
            switch_marker = self._get_switch_marker(context_speakers[-1], current_speaker)
        
        # 完整prompt
        prompt = f"""You are analyzing student opinion evolution in classroom dialogue.

{few_shot_text}Dialogue History:
{dialogue.strip() if dialogue else "(No previous dialogue)"}

Current Turn:
{switch_marker} {current_speaker}: {current_sentence}

Task: Classify the opinion evolution type. Choose ONE from:
- Irrelevant: No opinion content or off-topic
- New: New opinion not mentioned before
- Strengthened: Supporting/reinforcing previous opinion
- Weakened: Questioning/doubting previous opinion
- Adopted: Accepting teacher's/another's opinion (usually with speaker switch)
- Refuted: Rejecting/negating an opinion (usually with speaker switch)

IMPORTANT: Output ONLY the label name, no explanation.

Opinion Evolution Type:"""
        
        return prompt
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 完整文本
        full_text = sample['prompt'] + f" {sample['label']}"
        
        # 分词
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        labels = input_ids.clone()
        
        # Mask prompt部分
        prompt_encoding = self.tokenizer(
            sample['prompt'],
            truncation=True,
            return_tensors='pt'
        )
        prompt_length = prompt_encoding['input_ids'].shape[1]
        labels[:prompt_length] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels
        }


# =============================================================================
# 2. Few-shot示例采样
# =============================================================================

def sample_few_shot_examples(train_csv_path, num_per_class=2):
    """从训练集采样Few-shot示例（每类2个）"""
    
    print(f"\n📝 采样Few-shot示例（每类{num_per_class}个）...")
    
    df = pd.read_csv(train_csv_path)
    grouped = df.groupby('class', sort=False)
    
    label_names = ['Irrelevant', 'New', 'Strengthened', 'Weakened', 'Adopted', 'Refuted']
    teacher_names = {'T', 'Teacher', 'Ms. G', 'Mrs. G'}
    
    samples_by_label = {i: [] for i in range(6)}
    
    for class_id, group in grouped:
        sentences = group['Sentence'].tolist()
        speakers = group['Speaker'].tolist()
        labels = group['label'].tolist()
        
        for i in range(len(sentences)):
            # 简化的上下文
            context = ""
            if i > 0:
                context = f"{speakers[i-1]}: {sentences[i-1]}"
            
            # Switch标记
            switch = ""
            if i > 0:
                prev_is_t = speakers[i-1] in teacher_names or str(speakers[i-1]).startswith('T')
                curr_is_t = speakers[i] in teacher_names or str(speakers[i]).startswith('T')
                switch = "[Switch]" if prev_is_t != curr_is_t else "[Same]"
            
            sample = {
                'context': context if context else "(First utterance)",
                'current': f"{switch} {speakers[i]}: {sentences[i]}",
                'label': label_names[labels[i]]
            }
            
            samples_by_label[labels[i]].append(sample)
    
    # 每类采样
    few_shot_examples = []
    for label_id in range(6):
        available = samples_by_label[label_id]
        if len(available) > 0:
            selected_indices = np.random.choice(
                len(available),
                min(num_per_class, len(available)),
                replace=False
            )
            for idx in selected_indices:
                few_shot_examples.append(available[idx])
    
    print(f"  已采样 {len(few_shot_examples)} 个示例")
    return few_shot_examples


# =============================================================================
# 3. 评估函数（改进版）
# =============================================================================

def evaluate_model_v2(model, tokenizer, test_dataset, device, label_names):
    """改进的评估函数"""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("\n正在评估模型...")
    with torch.no_grad():
        for i in tqdm(range(len(test_dataset)), desc="Evaluating"):
            sample = test_dataset.samples[i]
            
            inputs = tokenizer(
                sample['prompt'],
                return_tensors='pt',
                truncation=True,
                max_length=1024
            ).to(device)
            
            # 温度采样（增加多样性）
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=True,
                temperature=0.3,  # 轻微随机性
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id
            )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 改进的标签解析
            pred_label_id = parse_label_robust(generated_text, sample['prompt'], label_names)
            
            all_preds.append(pred_label_id)
            all_labels.append(sample['label_id'])
    
    return all_preds, all_labels


def parse_label_robust(generated_text, prompt, label_names):
    """鲁棒的标签解析"""
    
    # 提取prompt之后的部分
    if prompt in generated_text:
        response = generated_text[len(prompt):].strip()
    else:
        response = generated_text.strip()
    
    # 提取第一行或第一个词
    response = response.split('\n')[0].strip()
    response = response.split('.')[0].strip()
    response_lower = response.lower()
    
    # 精确匹配
    for i, label in enumerate(label_names):
        if label.lower() == response_lower:
            return i
    
    # 包含匹配
    for i, label in enumerate(label_names):
        if label.lower() in response_lower:
            return i
    
    # 部分匹配
    label_keywords = {
        'irrelevant': 0,
        'new': 1,
        'strengthen': 2,
        'weaken': 3,
        'adopt': 4,
        'refut': 5
    }
    
    for keyword, label_id in label_keywords.items():
        if keyword in response_lower:
            return label_id
    
    # 默认返回Irrelevant（避免崩溃）
    return 0


# =============================================================================
# 4. 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='LLM QLoRA微调（改进版）')
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['llama', 'qwen'],
                        help='模型类型')
    parser.add_argument('--model_path', type=str, required=True,
                        help='预训练模型路径')
    parser.add_argument('--data_dir', type=str, default='dataset_split_result_v4')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--lora_rank', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--use_resampling', action='store_true',
                        help='启用数据重采样')
    parser.add_argument('--use_few_shot', action='store_true',
                        help='启用Few-shot示例')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        suffix = "_v2" if (args.use_resampling or args.use_few_shot) else ""
        args.output_dir = f'../llm_baseline_{args.model_type}{suffix}'
    
    config = vars(args)
    
    print("=" * 80)
    print(f"LLM QLoRA微调 v2 - {args.model_type.upper()}")
    print("=" * 80)
    print(json.dumps(config, indent=2, ensure_ascii=False))
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # QLoRA配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # 加载模型
    print(f"\n加载模型: {args.model_path} (4-bit量化) ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    print("加载 Tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 准备模型
    model = prepare_model_for_kbit_training(model)
    

    # LoRA配置
    # 修改说明：Qwen2 和 Llama-3 都遵循相同的命名规范，且微调所有线性层(Linear Layers)效果通常更好
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        target_modules=target_modules,
        bias="none"
    )
    
    model = get_peft_model(model, lora_config)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"\n可训练参数: {trainable_params:,} / {all_params:,} ({100 * trainable_params / all_params:.2f}%)")
    
    # Few-shot示例采样
    few_shot_examples = []
    if args.use_few_shot:
        train_csv = os.path.join(args.data_dir, 'train.csv')
        few_shot_examples = sample_few_shot_examples(train_csv, num_per_class=2)
    
    # 加载数据集
    print("\n加载数据集...")
    train_dataset = OpinionEvolutionLLMDatasetV2(
        os.path.join(args.data_dir, 'train.csv'),
        tokenizer,
        max_length=args.max_length,
        model_type=args.model_type,
        use_resampling=args.use_resampling,
        use_few_shot=args.use_few_shot,
        few_shot_examples=few_shot_examples
    )
    
    val_dataset = OpinionEvolutionLLMDatasetV2(
        os.path.join(args.data_dir, 'val.csv'),
        tokenizer,
        max_length=args.max_length,
        model_type=args.model_type,
        use_resampling=False,
        use_few_shot=args.use_few_shot,
        few_shot_examples=few_shot_examples
    )
    
    test_dataset = OpinionEvolutionLLMDatasetV2(
        os.path.join(args.data_dir, 'test.csv'),
        tokenizer,
        max_length=args.max_length,
        model_type=args.model_type,
        use_resampling=False,
        use_few_shot=args.use_few_shot,
        few_shot_examples=few_shot_examples
    )
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,  # 添加warmup
        weight_decay=0.01,  # 正则化
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none"
    )
    
    # Trainer
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )
    
    # 训练
    print("\n开始训练...")
    trainer.train()
    
    # 保存模型
    print("\n保存模型...")
    trainer.save_model(args.output_dir)
    
    # 测试集评估
    print("\n" + "=" * 80)
    print("测试集评估")
    print("=" * 80)
    
    all_preds, all_labels = evaluate_model_v2(
        model,
        tokenizer,
        test_dataset,
        device,
        train_dataset.label_names
    )
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    print(f"\n测试集结果:")
    print(f"  准确率: {accuracy:.4f}")
    print(f"  Macro-F1: {macro_f1:.4f}")
    
    print("\n详细分类报告:")
    print(classification_report(
        all_labels,
        all_preds,
        target_names=train_dataset.label_names,
        digits=4
    ))
    
    # 保存结果
    metrics = {
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'use_resampling': args.use_resampling,
        'use_few_shot': args.use_few_shot
    }
    
    with open(os.path.join(args.output_dir, 'test_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n训练完成！模型保存位置: {args.output_dir}")


if __name__ == '__main__':
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    
    main()
