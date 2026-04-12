#!/usr/bin/env python3
"""
LLM QLoRA微调脚本 (Group B)

支持的模型:
- Llama-3.1-8B-Instruct
- Qwen2-7B

使用与混合模型相同的Prompt格式（包含Switch/Same标记）
使用QLoRA (4-bit量化 + LoRA) 进行高效微调
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
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# 1. 数据集
# =============================================================================

class OpinionEvolutionLLMDataset(Dataset):
    """
    观点演化LLM微调数据集
    生成指令微调格式的对话
    """
    
    def __init__(
        self,
        csv_file: str,
        tokenizer,
        max_length: int = 1024,
        context_window: int = 5,
        model_type: str = 'llama'
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.context_window = context_window
        self.model_type = model_type
        
        # 教师名字集合
        self.teacher_names = {'T', 'Teacher', 'Ms. G', 'Mrs. G'}
        
        # 标签映射
        self.label_names = [
            'Irrelevant',
            'New',
            'Strengthened',
            'Weakened',
            'Adopted',
            'Refuted'
        ]
        
        print(f"正在加载数据: {csv_file} ...")
        df = pd.read_csv(csv_file)
        
        # 按课堂分组
        grouped = df.groupby('class', sort=False)
        
        self.samples = []
        
        for class_id, group in grouped:
            sentences = group['Sentence'].tolist()
            speakers = group['Speaker'].tolist()
            classifications = group['label'].tolist()
            
            for i in range(len(sentences)):
                # 提取上下文
                start_idx = max(0, i - context_window)
                context_sentences = sentences[start_idx:i]
                context_speakers = speakers[start_idx:i]
                
                # 构造prompt
                prompt = self._build_prompt(
                    current_sentence=sentences[i],
                    current_speaker=speakers[i],
                    context_sentences=context_sentences,
                    context_speakers=context_speakers
                )
                
                label = self.label_names[classifications[i]]
                
                self.samples.append({
                    'prompt': prompt,
                    'label': label,
                    'label_id': classifications[i]
                })
        
        print(f"加载完成。有效样本数: {len(self.samples)}")
        
        # 统计标签分布
        label_ids = [s['label_id'] for s in self.samples]
        print(f"\n观点演化标签分布:")
        unique_labels, counts = np.unique(label_ids, return_counts=True)
        for label, count in zip(unique_labels, counts):
            print(f"  {label} ({self.label_names[label]}): {count} 样本")
    
    def _get_switch_marker(self, prev_speaker, current_speaker):
        """获取Switch/Same标记"""
        if not prev_speaker:
            return ""
        
        prev = str(prev_speaker).strip()
        curr = str(current_speaker).strip()
        
        # 判断是否是教师
        prev_is_teacher = prev.startswith('T') or prev in self.teacher_names
        curr_is_teacher = curr.startswith('T') or curr in self.teacher_names
        
        # 判断是否切换
        if prev_is_teacher != curr_is_teacher:
            return "[Switch]"
        elif prev == curr:
            return "[Same]"
        else:
            return "[Switch]"
    
    def _build_prompt(self, current_sentence, current_speaker, 
                      context_sentences, context_speakers):
        """构造Prompt（与混合模型一致）"""
        
        # 对话历史
        dialogue = ""
        if context_sentences:
            for ctx_speaker, ctx_sentence in zip(context_speakers, context_sentences):
                dialogue += f"{ctx_speaker}: {ctx_sentence}\n"
        
        # 当前轮次的Switch/Same标记
        switch_marker = ""
        if context_speakers:
            switch_marker = self._get_switch_marker(context_speakers[-1], current_speaker)
        
        # 完整prompt
        prompt = f"""You are analyzing student opinion evolution in classroom dialogue.

Dialogue History:
{dialogue.strip() if dialogue else "No previous dialogue."}

Current Turn:
{switch_marker} {current_speaker}: {current_sentence}

Classify the opinion evolution type of the current student's utterance:
- Irrelevant: No opinion content
- New: Expressing a new opinion
- Strengthened: Strengthening previous opinion
- Weakened: Weakening previous opinion
- Adopted: Adopting teacher's opinion
- Refuted: Refuting teacher's opinion

Opinion Evolution Type:"""
        
        return prompt
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 构造完整文本：prompt + label
        full_text = sample['prompt'] + f" {sample['label']}"
        
        # 分词
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 创建labels（只计算label部分的loss）
        input_ids = encoding['input_ids'].squeeze(0)
        labels = input_ids.clone()
        
        # 找到prompt的长度，mask掉prompt部分
        prompt_encoding = self.tokenizer(
            sample['prompt'],
            truncation=True,
            return_tensors='pt'
        )
        prompt_length = prompt_encoding['input_ids'].shape[1]
        labels[:prompt_length] = -100  # -100会被CrossEntropyLoss忽略
        
        return {
            'input_ids': input_ids,
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels
        }


# =============================================================================
# 2. 评估函数
# =============================================================================

def evaluate_model(model, tokenizer, test_dataset, device, label_names):
    """评估模型性能"""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("\n正在评估模型...")
    with torch.no_grad():
        for i in tqdm(range(len(test_dataset)), desc="Evaluating"):
            sample = test_dataset.samples[i]
            
            # 生成预测
            inputs = tokenizer(
                sample['prompt'],
                return_tensors='pt',
                truncation=True,
                max_length=1024
            ).to(device)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.pad_token_id
            )
            
            # 解码输出
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取预测标签
            pred_label = None
            generated_lower = generated_text.lower()
            
            for label in label_names:
                if label.lower() in generated_lower:
                    pred_label = label
                    break
            
            # 如果没找到，默认为Irrelevant
            if pred_label is None:
                pred_label = 'Irrelevant'
            
            # 转换为ID
            pred_id = label_names.index(pred_label)
            true_id = sample['label_id']
            
            all_preds.append(pred_id)
            all_labels.append(true_id)
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # 详细报告
    report = classification_report(
        all_labels,
        all_preds,
        target_names=label_names,
        digits=4
    )
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'report': report,
        'predictions': all_preds,
        'labels': all_labels
    }


# =============================================================================
# 3. 主训练流程
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='LLM QLoRA微调')
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['llama', 'qwen'],
                        help='模型类型')
    parser.add_argument('--model_path', type=str, required=True,
                        help='预训练模型路径')
    parser.add_argument('--data_dir', type=str, default='dataset_split_result_v4',
                        help='数据目录')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='批次大小（建议2-4，大模型显存需求高）')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                        help='梯度累积步数（实际batch=batch_size*accumulation）')
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='训练轮数（微调建议3-5轮）')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                        help='学习率')
    parser.add_argument('--max_length', type=int, default=1024,
                        help='最大序列长度')
    parser.add_argument('--lora_rank', type=int, default=16,
                        help='LoRA秩')
    parser.add_argument('--lora_alpha', type=int, default=32,
                        help='LoRA alpha')
    
    args = parser.parse_args()
    
    # 设置输出目录
    if args.output_dir is None:
        args.output_dir = f'../llm_baseline_{args.model_type}'
    
    config = vars(args)
    
    print("=" * 80)
    print(f"LLM QLoRA微调 - {args.model_type.upper()}")
    print("=" * 80)
    print(json.dumps(config, indent=2, ensure_ascii=False))
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存配置
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # QLoRA配置：4-bit量化
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
    
    # 加载tokenizer
    print("加载 Tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 准备模型进行k-bit训练
    model = prepare_model_for_kbit_training(model)
    
    # LoRA配置
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"] if args.model_type == 'llama' 
                      else ["c_attn", "c_proj"],  # Qwen使用不同的module名
        bias="none"
    )
    
    # 应用LoRA
    model = get_peft_model(model, lora_config)
    
    # 打印可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"\n可训练参数: {trainable_params:,} / {all_params:,} ({100 * trainable_params / all_params:.2f}%)")
    
    # 加载数据集
    print("\n加载数据集...")
    train_dataset = OpinionEvolutionLLMDataset(
        os.path.join(args.data_dir, 'train.csv'),
        tokenizer,
        max_length=args.max_length,
        model_type=args.model_type
    )
    
    val_dataset = OpinionEvolutionLLMDataset(
        os.path.join(args.data_dir, 'val.csv'),
        tokenizer,
        max_length=args.max_length,
        model_type=args.model_type
    )
    
    test_dataset = OpinionEvolutionLLMDataset(
        os.path.join(args.data_dir, 'test.csv'),
        tokenizer,
        max_length=args.max_length,
        model_type=args.model_type
    )
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=100,
        logging_steps=50,
        save_steps=500,
        eval_steps=500,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="none"
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    # 开始训练
    print("\n开始训练...")
    trainer.train()
    
    # 保存最终模型
    print("\n保存模型...")
    trainer.save_model(os.path.join(args.output_dir, 'final_model'))
    tokenizer.save_pretrained(os.path.join(args.output_dir, 'final_model'))
    
    # 测试集评估
    print("\n" + "=" * 80)
    print("测试集评估")
    print("=" * 80)
    
    test_results = evaluate_model(
        model,
        tokenizer,
        test_dataset,
        device,
        train_dataset.label_names
    )
    
    print(f"\n测试集结果:")
    print(f"  准确率: {test_results['accuracy']:.4f}")
    print(f"  Macro-F1: {test_results['macro_f1']:.4f}")
    print(f"\n详细分类报告:")
    print(test_results['report'])
    
    # 保存报告
    with open(os.path.join(args.output_dir, 'test_report.txt'), 'w') as f:
        f.write(f"模型类型: {args.model_type.upper()}\n")
        f.write(f"准确率: {test_results['accuracy']:.4f}\n")
        f.write(f"Macro-F1: {test_results['macro_f1']:.4f}\n\n")
        f.write(test_results['report'])
    
    print(f"\n训练完成！模型保存位置: {args.output_dir}")


if __name__ == '__main__':
    main()
