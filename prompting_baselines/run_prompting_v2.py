#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm
import time
from collections import Counter
from openai import OpenAI
from dotenv import load_dotenv

class ImprovedPromptBuilder:
    
    def __init__(self, mode='zero-shot'):
        self.mode = mode
        self.label_names = ['Irrelevant', 'New', 'Strengthened', 'Weakened', 'Adopted', 'Refuted']
    
    def build_zero_shot_prompt(self, context, current_speaker, current_sentence, consistency):
        
        system_prompt = """You are a classroom dialogue analysis expert. Your task is to classify the current utterance based on the relationship between the context (previous utterances) and the current utterance.

You must choose ONE label from the following six categories:

1. **Irrelevant**: The utterance is unrelated to the discussion topic, or is a meta-comment about the conversation state (e.g., "I don't understand"), or is procedural (e.g., teacher managing classroom).
2. **New**: Introduces a new claim, argument, or evidence that has not appeared in the context.
3. **Strengthened**: Provides support, agreement, evidence, or more detailed elaboration for an existing opinion in the context.
4. **Weakened**: Raises questions, counterexamples, disagreements, or points out limitations of an opinion in the context, but does not completely negate it.
5. **Adopted**: The speaker explicitly agrees with or accepts another speaker's opinion from the context (usually occurs when speaker switches).
6. **Refuted**: The speaker explicitly and directly negates an opinion from the context (usually occurs when speaker switches, e.g., starting with "No..." or "Yeah, but...").

---
**Key Examples:**

Example 1: Student adopts teacher's correction
Context:
(New) Toby: An object that you can get stuff inside of it, I guess.
(Refuted) T: Three dimensional is not flat.
Current Utterance:
- Speaker: Toby (speaker switch)
- Sentence: Not flat.
Label: Adopted

Example 2: Teacher strengthens student's idea
Context:
(New) Guy: It's the area around like the outside of it.
Current Utterance:
- Speaker: T (speaker switch)
- Sentence: The area of each of the surfaces, surface area.
Label: Strengthened

Example 3: Student refutes another student
Context:
(Strengthened) Erik: Yeah, it would.
Current Utterance:
- Speaker: Alan (speaker switch)
- Sentence: No, it doesn't.
Label: Refuted
---"""

        user_prompt = f"""
**Context (Previous Dialogue):**
{context if context else "(This is the first utterance)"}

**Current Utterance to Classify:**
- Speaker: {current_speaker} (Note: 'T' means teacher)
- Consistency: {consistency}
- Sentence: {current_sentence}

Your response must contain ONLY one of the six label names. Do not provide any explanation.

Classify:"""

        return system_prompt + user_prompt
    
    def build_few_shot_prompt(self, context, current_speaker, current_sentence, consistency, examples):
        """Few-shot Prompt with real examples"""
        
        system_desc = """You are a classroom dialogue analysis expert. Classify opinion evolution based on the relationship between context and current utterance.

**Six Categories:**
1. Irrelevant: Unrelated to topic or procedural
2. New: Introduces new claim/argument/evidence
3. Strengthened: Supports/agrees with existing opinion
4. Weakened: Questions/doubts existing opinion (not complete negation)
5. Adopted: Accepts another speaker's opinion (speaker switch)
6. Refuted: Explicitly negates an opinion (speaker switch)

---
**Examples from training data:**

"""
        
        for i, ex in enumerate(examples[:6], 1):
            system_desc += f"Example {i}:\n"
            system_desc += f"Context: {ex['context']}\n"
            system_desc += f"Current: Speaker={ex['speaker']} ({ex['consistency']}), Sentence={ex['sentence']}\n"
            system_desc += f"Label: {ex['label']}\n\n"
        
        user_prompt = f"""---
**Now classify:**

Context:
{context if context else "(First utterance)"}

Current Utterance:
- Speaker: {current_speaker}
- Consistency: {consistency}
- Sentence: {current_sentence}

Your response must be ONLY one label name.

Label:"""

        return system_desc + user_prompt
    
    def build_prompt(self, context, current_speaker, current_sentence, consistency, examples=None):
        """统一接口"""
        if self.mode == 'zero-shot':
            return self.build_zero_shot_prompt(context, current_speaker, current_sentence, consistency)
        else:
            return self.build_few_shot_prompt(context, current_speaker, current_sentence, consistency, examples or [])


class LLMClient:
    
    def __init__(self, model_type='deepseek'):
        load_dotenv()
        
        self.model_type = model_type
        self.model_name = "deepseek-chat" if model_type == 'deepseek' else "gpt-4o"
        
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        
        self.client = OpenAI(api_key=api_key, base_url=base_url if base_url else None)
        self.label_names = ['Irrelevant', 'New', 'Strengthened', 'Weakened', 'Adopted', 'Refuted']
    
    def call_api(self, prompt, temperature=0.0, max_retries=3):
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
                    time.sleep(2 ** attempt)
                else:
                    print(f"API调用失败: {e}")
                    return None
        return None
    
    def parse_response(self, response_text):
        if not response_text:
            return 0
        
        text = response_text.lower().strip()
        
        # 精确匹配
        for i, label in enumerate(self.label_names):
            if label.lower() == text:
                return i
        
        # 包含匹配
        for i, label in enumerate(self.label_names):
            if label.lower() in text:
                return i
        
        print(f"无法解析响应: {response_text}")
        return 0

def load_few_shot_examples(train_csv_path, num_per_class=2):
    print(f"\n加载Few-shot示例（每类{num_per_class}个）...")
    df = pd.read_csv(train_csv_path)
    
    grouped = df.groupby('class', sort=False)
    samples_by_label = {i: [] for i in range(6)}
    
    for class_id, group in grouped:
        sentences = group['Sentence'].tolist()
        speakers = group['Speaker'].tolist()
        labels = group['label'].tolist()
        
        previous_speaker = None
        
        for i in range(len(sentences)):
            current_speaker = speakers[i]
            
            # 判断consistency
            if (previous_speaker is None) or \
               (current_speaker == 'T' and previous_speaker == 'T') or \
               (current_speaker == previous_speaker):
                consistency = "same speaker"
            else:
                consistency = "speaker switch"
            
            # 上下文
            context = []
            start_idx = max(0, i - 5)
            for j in range(start_idx, i):
                context.append(f"{speakers[j]}: {sentences[j]}")
            
            context_text = "\n".join(context) if context else "(First utterance)"
            
            sample = {
                'context': context_text,
                'speaker': current_speaker,
                'sentence': sentences[i],
                'consistency': consistency,
                'label': ['Irrelevant', 'New', 'Strengthened', 'Weakened', 'Adopted', 'Refuted'][labels[i]]
            }
            
            samples_by_label[labels[i]].append(sample)
            previous_speaker = current_speaker

    examples = []
    for label_id in range(6):
        if len(samples_by_label[label_id]) > 0:
            selected_indices = np.random.choice(
                len(samples_by_label[label_id]),
                min(num_per_class, len(samples_by_label[label_id])),
                replace=False
            )
            for idx in selected_indices:
                examples.append(samples_by_label[label_id][idx])
    
    print(f"已加载 {len(examples)} 个Few-shot示例")
    return examples


def run_improved_prompting(test_csv_path, train_csv_path, model_type, mode, output_dir):
    
    print("\n" + "=" * 80)
    print(f"改进版Prompting实验: {model_type.upper()} ({mode})")
    print("=" * 80)
    
    # 初始化
    prompt_builder = ImprovedPromptBuilder(mode=mode)
    llm_client = LLMClient(model_type=model_type)
    
    # 加载Few-shot示例
    few_shot_examples = []
    if mode == 'few-shot':
        few_shot_examples = load_few_shot_examples(train_csv_path, num_per_class=2)
    
    # 读取测试集
    test_df = pd.read_csv(test_csv_path)
    grouped = test_df.groupby('class', sort=False)
    
    predictions = []
    true_labels = []
    results = []
    
    total_samples = len(test_df)
    print(f"\n测试样本数: {total_samples}")
    print(f"预计API调用: {total_samples} 次")
    
    pbar = tqdm(total=total_samples, desc="Prompting推理")
    
    for class_id, group in grouped:
        sentences = group['Sentence'].tolist()
        speakers = group['Speaker'].tolist()
        labels = group['label'].tolist()
        
        previous_speaker = None
        context_history = []  # 保存带标签的上下文
        
        for i in range(len(sentences)):
            current_speaker = speakers[i]
            current_sentence = sentences[i]
            
            # 判断consistency
            if (previous_speaker is None) or \
               (current_speaker == 'T' and previous_speaker == 'T') or \
               (current_speaker == previous_speaker):
                consistency = "same speaker"
            else:
                consistency = "speaker switch"
            
            # 构建上下文（带预测标签）
            context_text = "\n".join(context_history[-5:]) if context_history else "(This is the first utterance)"
            
            # 构建Prompt
            prompt = prompt_builder.build_prompt(
                context_text,
                current_speaker,
                current_sentence,
                consistency,
                few_shot_examples
            )
            
            # 调用LLM
            response = llm_client.call_api(prompt)
            pred_label = llm_client.parse_response(response)
            
            predictions.append(pred_label)
            true_labels.append(labels[i])
            
            results.append({
                'class_id': class_id,
                'speaker': current_speaker,
                'sentence': current_sentence,
                'consistency': consistency,
                'true_label': labels[i],
                'pred_label': pred_label,
                'llm_response': response
            })
            
            # 更新上下文历史（使用预测标签）
            pred_label_name = llm_client.label_names[pred_label]
            context_history.append(f"({pred_label_name}) {current_speaker}: {current_sentence}")
            
            previous_speaker = current_speaker
            pbar.update(1)
            time.sleep(0.1)  # 速率限制
    
    pbar.close()
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'predictions_v2.csv'), index=False)
    
    # 评估
    accuracy = accuracy_score(true_labels, predictions)
    macro_f1 = f1_score(true_labels, predictions, average='macro')
    
    print("\n" + "=" * 80)
    print("评估结果 (改进版)")
    print("=" * 80)
    print(f"准确率: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    
    print("\n详细分类报告:")
    print(classification_report(
        true_labels,
        predictions,
        target_names=llm_client.label_names,
        digits=4
    ))
    
    # 混淆矩阵
    cm = confusion_matrix(true_labels, predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=llm_client.label_names,
        yticklabels=llm_client.label_names
    )
    plt.title(f'Confusion Matrix - {model_type.upper()} ({mode})', fontsize=14)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_v2.png'), dpi=300)
    plt.close()
    
    # 保存指标
    metrics = {
        'model_type': model_type,
        'mode': mode,
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'num_samples': total_samples
    }
    
    with open(os.path.join(output_dir, 'metrics_v2.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✓ 结果已保存到: {output_dir}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='改进版Prompting Baseline实验')
    parser.add_argument('--model_type', type=str, default='deepseek',
                        choices=['deepseek', 'gpt4o'],
                        help='LLM模型类型')
    parser.add_argument('--mode', type=str, default='zero-shot',
                        choices=['zero-shot', 'few-shot'],
                        help='Prompting模式')
    parser.add_argument('--data_dir', type=str, default='dataset_split_result_v4',
                        help='数据目录')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录')
    
    args = parser.parse_args()
    
    # 设置输出目录
    if args.output_dir is None:
        args.output_dir = f"../prompting_{args.model_type}_{args.mode}_v2"
    
    # 数据路径
    test_csv = os.path.join(args.data_dir, 'test.csv')
    train_csv = os.path.join(args.data_dir, 'train.csv')
    
    # 运行实验
    metrics = run_improved_prompting(
        test_csv,
        train_csv,
        args.model_type,
        args.mode,
        args.output_dir
    )
    
    print("\n" + "=" * 80)
    print("实验完成！")
    print("=" * 80)
    print(f"准确率: {metrics['accuracy']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")


if __name__ == '__main__':
    main()
