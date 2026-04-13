#!/usr/bin/env python3
"""
错误分析工具: 提取和分析模型分类错误的样本

功能:
1. 提取测试集/验证集上的错误预测样本
2. 保存详细的上下文信息(说话者、本句、前5句历史)
3. 按真实观点标签分组,便于分析每类的错误模式
4. 区分错误类型(仅对话行为错、仅观点演化错、两者都错)
"""

import pandas as pd
import numpy as np
import os
from typing import List, Dict, Tuple


# 标签映射（与当前多任务模型使用的 12 类对话行为保持一致）
DIALOGUE_ACT_LABELS = {
    0: "None (Teacher)",
    1: "Keeping Together",
    2: "Relate",
    3: "Restating",
    4: "Revoicing",
    5: "Press Accuracy",
    6: "Press Reasoning",
    7: "None (Student)",
    8: "Relating to Another Student",
    9: "Asking for Info",
    10: "Making a Claim",
    11: "Providing Evidence/Reasoning"
}

OPINION_LABELS = {
    0: "Irrelevant",
    1: "New",
    2: "Strengthened",
    3: "Weakened",
    4: "Adopted",
    5: "Refuted"
}


def extract_context_window(df: pd.DataFrame, idx: int, window_size: int = 5) -> str:
    current_dialogue_id = df.loc[idx, 'dialogue_id'] if 'dialogue_id' in df.columns else None

    context_sentences = []
    search_idx = idx - 1
    collected = 0
    
    while search_idx >= 0 and collected < window_size:
      
        if current_dialogue_id is not None:
            if df.loc[search_idx, 'dialogue_id'] != current_dialogue_id:
                break
        
        speaker = df.loc[search_idx, 'Speaker']
        sentence = df.loc[search_idx, 'Sentence']
        context_sentences.insert(0, f"{speaker}: {sentence}")
        
        collected += 1
        search_idx -= 1
    

    if collected < window_size:
        context_sentences.insert(0, "[对话开始]")
    
    return "\n".join(context_sentences)


def analyze_errors(
    predictions: List[int],
    labels: List[int],
    dialogue_act_preds: List[int],
    dialogue_act_labels: List[int],
    data_df: pd.DataFrame,
    output_path: str,
    split_name: str = "test"
) -> Dict:

    assert len(predictions) == len(labels) == len(data_df), \
        f"长度不匹配: preds={len(predictions)}, labels={len(labels)}, df={len(data_df)}"
    
    error_records = []
    

    for idx in range(len(predictions)):
        opinion_pred = predictions[idx]
        opinion_true = labels[idx]
        dialogue_act_pred = dialogue_act_preds[idx]
        dialogue_act_true = dialogue_act_labels[idx]
 
        opinion_correct = (opinion_pred == opinion_true)
        dialogue_act_correct = (dialogue_act_pred == dialogue_act_true)
        
        
        if not opinion_correct or not dialogue_act_correct:
            if not opinion_correct and not dialogue_act_correct:
                error_type = "both_wrong"
            elif not dialogue_act_correct:
                error_type = "dialogue_act_wrong"
            else:
                error_type = "opinion_wrong"
          
            row = data_df.iloc[idx]
            speaker = row['Speaker']
            sentence = row['Sentence']
            

            context = extract_context_window(data_df, idx, window_size=5)
            
    
            error_record = {
                '数据集': split_name,
                '样本索引': idx,
                '真实观点标签': OPINION_LABELS[opinion_true],
                '真实观点标签ID': opinion_true,
                '说话者': speaker,
                '本句内容': sentence,
                '上下文(前5句)': context,
                '真实对话行为': DIALOGUE_ACT_LABELS[dialogue_act_true],
                '真实对话行为ID': dialogue_act_true,
                '预测对话行为': DIALOGUE_ACT_LABELS[dialogue_act_pred],
                '预测对话行为ID': dialogue_act_pred,
                '预测观点演化': OPINION_LABELS[opinion_pred],
                '预测观点演化ID': opinion_pred,
                '对话行为是否正确': '✓' if dialogue_act_correct else '✗',
                '观点演化是否正确': '✓' if opinion_correct else '✗',
                '错误类型': error_type
            }
            
            error_records.append(error_record)
    
   
    error_df = pd.DataFrame(error_records)
    
  
    if len(error_df) > 0:
        error_df = error_df.sort_values(
            by=['真实观点标签ID', '错误类型', '样本索引'],
            ascending=[True, True, True]
        )
    

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    error_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    

    stats = {
        'total_samples': len(predictions),
        'total_errors': len(error_df),
        'error_rate': len(error_df) / len(predictions) if len(predictions) > 0 else 0,
        'both_wrong': len(error_df[error_df['错误类型'] == 'both_wrong']),
        'dialogue_act_wrong_only': len(error_df[error_df['错误类型'] == 'dialogue_act_wrong']),
        'opinion_wrong_only': len(error_df[error_df['错误类型'] == 'opinion_wrong']),
        'dialogue_act_accuracy': sum([1 for d_pred, d_true in zip(dialogue_act_preds, dialogue_act_labels) 
                                      if d_pred == d_true]) / len(predictions),
        'opinion_accuracy': sum([1 for o_pred, o_true in zip(predictions, labels) 
                                if o_pred == o_true]) / len(predictions)
    }
    

    stats['errors_by_opinion_class'] = {}
    for opinion_id, opinion_name in OPINION_LABELS.items():
        class_errors = error_df[error_df['真实观点标签ID'] == opinion_id]
        class_total = sum([1 for label in labels if label == opinion_id])
        stats['errors_by_opinion_class'][opinion_name] = {
            'total': class_total,
            'errors': len(class_errors),
            'error_rate': len(class_errors) / class_total if class_total > 0 else 0
        }
    
    return stats


def print_error_stats(stats: Dict):
    """打印错误统计信息"""
    print("\n" + "="*80)
    print("错误分析统计")
    print("="*80)
    print(f"总样本数: {stats['total_samples']}")
    print(f"错误样本数: {stats['total_errors']} ({stats['error_rate']*100:.2f}%)")
    print(f"\n任务准确率:")
    print(f"  对话行为识别: {stats['dialogue_act_accuracy']*100:.2f}%")
    print(f"  观点演化分类: {stats['opinion_accuracy']*100:.2f}%")
    print(f"\n错误类型分布:")
    print(f"  两个任务都错: {stats['both_wrong']} ({stats['both_wrong']/stats['total_errors']*100:.2f}%)")
    print(f"  仅对话行为错: {stats['dialogue_act_wrong_only']} ({stats['dialogue_act_wrong_only']/stats['total_errors']*100:.2f}%)")
    print(f"  仅观点演化错: {stats['opinion_wrong_only']} ({stats['opinion_wrong_only']/stats['total_errors']*100:.2f}%)")
    
    print(f"\n各观点类别错误率:")
    for opinion_name, class_stats in stats['errors_by_opinion_class'].items():
        print(f"  {opinion_name:15s}: {class_stats['errors']:4d}/{class_stats['total']:4d} "
              f"({class_stats['error_rate']*100:5.2f}%)")
    print("="*80)


if __name__ == "__main__":
    print("错误分析工具已加载")
    print("请在训练脚本中调用 analyze_errors() 函数")
