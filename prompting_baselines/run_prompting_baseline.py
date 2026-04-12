#!/usr/bin/env python3
"""
LLM Prompting基线实验 (Group C)

纯Prompt推理，不使用判别模型筛选，全量调用LLM

支持的模型:
- DeepSeek-v3 (via API)
- GPT-4o (via API)

支持的模式:
- Zero-shot: 无示例
- Few-shot: 带示例（从训练集选择）
"""

import argparse
import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
from collections import Counter
from openai import OpenAI
from dotenv import load_dotenv


# =============================================================================
# 1. Prompt模板
# =============================================================================

class PromptTemplate:
    """Prompt模板管理"""
    
    def __init__(self, mode='zero-shot', num_examples=3):
        self.mode = mode
        self.num_examples = num_examples
        self.label_names = ['Irrelevant', 'New', 'Strengthened', 'Weakened', 'Adopted', 'Refuted']
        self.examples = []
    
    def load_examples(self, train_csv_path):
        """从训练集加载Few-shot示例（分层采样）"""
        if self.mode != 'few-shot':
            return
        
        print(f"\n加载Few-shot示例（每类{self.num_examples}个）...")
        df = pd.read_csv(train_csv_path)
        
        # 按课堂分组获取上下文
        grouped = df.groupby('class', sort=False)
        
        samples_by_label = {i: [] for i in range(6)}
        
        for class_id, group in grouped:
            sentences = group['Sentence'].tolist()
            speakers = group['Speaker'].tolist()
            labels = group['label'].tolist()
            
            for i in range(len(sentences)):
                # 获取上下文
                context = []
                start_idx = max(0, i - 5)
                for j in range(start_idx, i):
                    context.append(f"{speakers[j]}: {sentences[j]}")
                
                context_text = "\n".join(context) if context else "No previous dialogue."
                
                sample = {
                    'context': context_text,
                    'current': f"{speakers[i]}: {sentences[i]}",
                    'label': self.label_names[labels[i]]
                }
                
                samples_by_label[labels[i]].append(sample)
        
        # 每类随机选择N个示例
        for label_id in range(6):
            if len(samples_by_label[label_id]) > 0:
                selected = np.random.choice(
                    len(samples_by_label[label_id]),
                    min(self.num_examples, len(samples_by_label[label_id])),
                    replace=False
                )
                for idx in selected:
                    self.examples.append(samples_by_label[label_id][idx])
        
        print(f"已加载 {len(self.examples)} 个Few-shot示例")

    def build_zero_shot_prompt(self, context, current_turn):
        """构建Zero-shot Prompt（借鉴classify_opinions0.py的设计）"""
        prompt = f"""You are a classroom dialogue analysis expert. Your task is to classify the current utterance based on the relationship between the context (previous utterances) and the current utterance.

You must choose ONE label from the following six categories:

1. **Irrelevant**: The utterance is unrelated to the discussion topic, or is a meta-comment about the conversation state (e.g., "I don't understand"), or is procedural (e.g., teacher managing classroom).
2. **New**: Introduces a new claim, argument, or evidence that has not appeared in the context.
3. **Strengthened**: Provides support, agreement, evidence, or more detailed elaboration for an existing opinion in the context.
4. **Weakened**: Raises questions, counterexamples, disagreements, or points out limitations of an opinion in the context, but does not completely negate it.
5. **Adopted**: The speaker explicitly agrees with or accepts another speaker's opinion from the context (usually occurs when speaker switches).
6. **Refuted**: The speaker explicitly and directly negates an opinion from the context (usually occurs when speaker switches, e.g., starting with "No..." or "Yeah, but...").

---
**Context (Previous Dialogue):**
{context}

**Current Utterance to Classify:**
{current_turn}
---

Your response must contain ONLY one of the six labels above. Do not provide any explanation.

Opinion Evolution Type:"""

        return prompt
    
    def build_few_shot_prompt(self, context, current_turn):
        """构建Few-shot Prompt（改进版）"""
        # 系统说明
        system_desc = """You are a classroom dialogue analysis expert. Classify opinion evolution based on the relationship between context and current utterance.

**Six Categories:**
1. Irrelevant: Unrelated to topic or procedural
2. New: Introduces new claim/argument/evidence
3. Strengthened: Supports/agrees with existing opinion
4. Weakened: Questions/doubts existing opinion (not complete negation)
5. Adopted: Accepts another speaker's opinion (speaker switch)
6. Refuted: Explicitly negates an opinion (speaker switch)

---
"""
        
        # 示例部分(最多显示6个,每类1个)
        examples_text = "**Key Examples:**\n\n"
        
        for i, example in enumerate(self.examples[:min(6, len(self.examples))], 1):
            examples_text += f"Example {i}:\n"
            examples_text += f"Context: {example['context']}\n"
            examples_text += f"Current: {example['current']}\n"
            examples_text += f"Label: {example['label']}\n"
            examples_text += "-" * 60 + "\n\n"
        
        # 当前任务
        prompt = f"""{system_desc}{examples_text}
**Now classify this case:**

Context:
{context}

Current Utterance:
{current_turn}

Your response must be ONLY one label name (Irrelevant/New/Strengthened/Weakened/Adopted/Refuted).

Opinion Evolution Type:"""
        
        return prompt
    
    def build_prompt(self, context, current_turn):
        """构建Prompt"""
        if self.mode == 'zero-shot':
            return self.build_zero_shot_prompt(context, current_turn)
        else:
            return self.build_few_shot_prompt(context, current_turn)


# =============================================================================
# 2. LLM API调用
# =============================================================================

class LLMClient:
    """LLM API客户端（使用OpenAI SDK）"""
    
    def __init__(self, model_type='deepseek', api_key=None, base_url=None):
        self.model_type = model_type
        
        # 加载环境变量
        load_dotenv()
        
        # 设置模型名称
        if model_type == 'deepseek':
            self.model_name = "deepseek-chat"
        elif model_type == 'gpt4o':
            self.model_name = "gpt-4o"
        else:
            self.model_name = model_type
        
        # 从.env读取配置（与混合模型一致）
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        if base_url is None:
            base_url = os.getenv("OPENAI_BASE_URL")
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url if base_url else None
        )
        
        self.label_names = ['Irrelevant', 'New', 'Strengthened', 'Weakened', 'Adopted', 'Refuted']
    
    def call_api(self, prompt, temperature=0.0, max_retries=3):
        """调用LLM API（使用OpenAI SDK）"""
        messages = [{"role": "user", "content": prompt}]
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=50
                )
                
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"⚠️  API调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                    time.sleep(2 ** attempt)
                else:
                    print(f"❌ API调用最终失败: {e}")
                    return None
        
        return None

    def parse_response(self, response_text):
        """解析LLM响应为标签ID"""
        if not response_text:
            return 0  # 默认返回Irrelevant

        # 1. 预处理：转小写，移除前后空白和标点
        import re
        text = response_text.lower().strip()
        # 移除末尾常见的标点符号 (处理 "3." 或 "3," 这种情况)
        text = re.sub(r'[.,!?;:]+$', '', text)

        # === 新增：处理数字输出 (1-6) ===
        # Prompt中的列表是 1-6，而我们的标签索引是 0-5
        if text.isdigit():
            idx = int(text) - 1
            if 0 <= idx < len(self.label_names):
                return idx
        # ==============================

        # 2. 优先：精确匹配
        for i, label in enumerate(self.label_names):
            if text == label.lower():
                return i

        # 3. 次优：在文本中寻找关键词
        words = re.findall(r'\b\w+\b', text)
        for word in reversed(words):
            # 先检查是不是数字单词 (防止模型偶尔输出 "one", "two" 等，虽然DeepSeek很少这么做)
            if word.isdigit():
                idx = int(word) - 1
                if 0 <= idx < len(self.label_names):
                    return idx

            # 再检查文本标签
            for i, label in enumerate(self.label_names):
                if word == label.lower():
                    return i

        # 4. 模糊匹配
        if 'irrelevant' in text or 'off-topic' in text or 'no opinion' in text:
            return 0
        elif 'new' in text:
            return 1
        elif 'strengthen' in text or 'reinforc' in text:
            return 2
        elif 'weaken' in text or 'reduc' in text:
            return 3
        elif 'adopt' in text or 'agree' in text:
            return 4
        elif 'refute' in text or 'disagree' in text:
            return 5

        # 5. 无法解析
        print(f"⚠️  无法解析响应: {response_text[:50]}...")
        return 0


# =============================================================================
# 3. 推理流程
# =============================================================================

def run_prompting_inference(
    test_csv_path,
    train_csv_path,
    llm_client,
    prompt_template,
    output_dir,
    context_window=5
):
    """运行Prompting推理"""
    
    print("\n" + "=" * 80)
    print("开始Prompting推理")
    print("=" * 80)
    
    # 读取测试集
    test_df = pd.read_csv(test_csv_path)
    
    # 按课堂分组
    grouped = test_df.groupby('class', sort=False)
    
    results = []
    all_preds = []
    all_labels = []
    
    total_samples = len(test_df)
    api_calls = 0
    
    print(f"\n测试样本数: {total_samples}")
    print(f"预计API调用: {total_samples} 次 (100%)")
    
    pbar = tqdm(total=total_samples, desc="Prompting推理")
    
    for class_id, group in grouped:
        sentences = group['Sentence'].tolist()
        speakers = group['Speaker'].tolist()
        labels = group['label'].tolist()
        
        previous_speaker = None
        
        for i in range(len(sentences)):
            current_speaker = speakers[i]
            
            # 确定发言人一致性（借鉴classify_opinions0.py）
            if (previous_speaker is None) or \
               (current_speaker == 'T' and previous_speaker == 'T') or \
               (current_speaker == previous_speaker):
                consistency = "same speaker"
            else:
                consistency = "speaker switch"
            
            # 构建上下文（带标签格式，类似classify_opinions0.py）
            context_parts = []
            start_idx = max(0, i - context_window)
            
            for j in range(start_idx, i):
                # 简化版本：上下文不包含预测标签（避免错误传播）
                context_parts.append(f"{speakers[j]}: {sentences[j]}")
            
            context_text = "\n".join(context_parts) if context_parts else "(This is the first utterance)"
            
            # 当前发言包含Speaker和consistency信息
            current_turn = f"Speaker: {current_speaker} ({consistency})\nSentence: {sentences[i]}"
            
            previous_speaker = current_speaker
            
            # 构建Prompt
            prompt = prompt_template.build_prompt(context_text, current_turn)
            
            # 调用LLM
            response = llm_client.call_api(prompt)
            api_calls += 1
            
            # 解析响应
            pred_label_id = llm_client.parse_response(response)
            true_label_id = labels[i]
            
            all_preds.append(pred_label_id)
            all_labels.append(true_label_id)
            
            results.append({
                'class_id': class_id,
                'sentence': sentences[i],
                'speaker': speakers[i],
                'true_label': true_label_id,
                'true_label_name': llm_client.label_names[true_label_id],
                'pred_label': pred_label_id,
                'pred_label_name': llm_client.label_names[pred_label_id],
                'llm_response': response,
                'correct': pred_label_id == true_label_id
            })
            
            pbar.update(1)
            
            # 避免API限流
            time.sleep(0.1)
    
    pbar.close()
    
    # 保存预测结果
    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(output_dir, 'predictions.csv')
    results_df.to_csv(results_csv_path, index=False, encoding='utf-8-sig')
    print(f"\n✓ 预测结果已保存: {results_csv_path}")
    
    # 保存JSON格式
    predictions_json_path = os.path.join(output_dir, 'predictions.json')
    with open(predictions_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✓ JSON格式已保存: {predictions_json_path}")
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    print(f"\n" + "=" * 80)
    print("评估结果")
    print("=" * 80)
    print(f"API调用次数: {api_calls} / {total_samples} ({api_calls/total_samples*100:.1f}%)")
    print(f"准确率: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    
    # 详细分类报告
    report = classification_report(
        all_labels,
        all_preds,
        target_names=llm_client.label_names,
        digits=4
    )
    print(f"\n详细分类报告:")
    print(report)
    
    # 保存指标
    metrics = {
        'model_type': llm_client.model_type,
        'mode': prompt_template.mode,
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'api_calls': api_calls,
        'total_samples': total_samples,
        'api_call_rate': api_calls / total_samples
    }
    
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ 指标已保存: {metrics_path}")
    
    # 保存文本报告
    report_path = os.path.join(output_dir, 'test_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"模型: {llm_client.model_type.upper()}\n")
        f.write(f"模式: {prompt_template.mode}\n")
        f.write(f"准确率: {accuracy:.4f}\n")
        f.write(f"Macro-F1: {macro_f1:.4f}\n")
        f.write(f"API调用率: {api_calls/total_samples*100:.1f}%\n\n")
        f.write(report)
    print(f"✓ 文本报告已保存: {report_path}")
    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=llm_client.label_names,
                yticklabels=llm_client.label_names)
    plt.title(f'{llm_client.model_type.upper()} ({prompt_template.mode}) - 混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300)
    print(f"✓ 混淆矩阵已保存: {cm_path}")
    plt.close()
    
    return metrics


# =============================================================================
# 4. 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='LLM Prompting基线实验')
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['deepseek', 'gpt4o'],
                        help='LLM类型')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['zero-shot', 'few-shot'],
                        help='Prompting模式')
    parser.add_argument('--num_examples', type=int, default=3,
                        help='Few-shot示例数（每类）')
    parser.add_argument('--data_dir', type=str, default='dataset_split_result_v4',
                        help='数据目录')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录')
    parser.add_argument('--api_key', type=str, default=None,
                        help='API密钥（可选，默认从环境变量读取）')
    parser.add_argument('--base_url', type=str, default=None,
                        help='API地址（可选）')
    parser.add_argument('--context_window', type=int, default=5,
                        help='上下文窗口大小')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(args.random_seed)
    
    # 设置输出目录
    if args.output_dir is None:
        args.output_dir = f'../prompting_{args.model_type}_{args.mode.replace("-", "_")}'
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print(f"LLM Prompting基线 - {args.model_type.upper()} ({args.mode})")
    print("=" * 80)
    
    config = vars(args)
    print(json.dumps({k: v for k, v in config.items() if k != 'api_key'}, indent=2, ensure_ascii=False))
    
    # 保存配置
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump({k: v for k, v in config.items() if k != 'api_key'}, f, indent=2, ensure_ascii=False)
    
    # 初始化Prompt模板
    prompt_template = PromptTemplate(mode=args.mode, num_examples=args.num_examples)
    
    if args.mode == 'few-shot':
        train_csv_path = os.path.join(args.data_dir, 'train.csv')
        prompt_template.load_examples(train_csv_path)
    
    # 初始化LLM客户端
    llm_client = LLMClient(
        model_type=args.model_type,
        api_key=args.api_key,
        base_url=args.base_url
    )
    
    # 运行推理
    test_csv_path = os.path.join(args.data_dir, 'test.csv')
    train_csv_path = os.path.join(args.data_dir, 'train.csv')
    
    metrics = run_prompting_inference(
        test_csv_path=test_csv_path,
        train_csv_path=train_csv_path,
        llm_client=llm_client,
        prompt_template=prompt_template,
        output_dir=args.output_dir,
        context_window=args.context_window
    )
    
    print("\n" + "=" * 80)
    print("推理完成！")
    print("=" * 80)
    print(f"结果保存位置: {args.output_dir}")
    print(f"准确率: {metrics['accuracy']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"API调用率: {metrics['api_call_rate']*100:.1f}%")


if __name__ == '__main__':
    main()
