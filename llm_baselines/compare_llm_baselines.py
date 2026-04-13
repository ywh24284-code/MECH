#!/usr/bin/env python3
"""
LLM基线对比分析脚本

对比 Llama-3.1-8B, Qwen2-7B 与混合模型的性能
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import re

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def parse_test_report(report_file):
    if not os.path.exists(report_file):
        return None
    
    with open(report_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    acc_match = re.search(r'准确率:\s*(\d+\.\d+)', content)
    f1_match = re.search(r'Macro-F1:\s*(\d+\.\d+)', content)
    
    if acc_match and f1_match:
        return {
            'accuracy': float(acc_match.group(1)),
            'macro_f1': float(f1_match.group(1))
        }
    return None


def parse_hybrid_metrics(metrics_file):
    if not os.path.exists(metrics_file):
        return None
    
    with open(metrics_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    metrics = {}
    for line in lines:
        if 'Accuracy:' in line or '准确率:' in line:
            match = re.search(r'(\d+\.\d+)%', line)
            if match:
                metrics['accuracy'] = float(match.group(1)) / 100
        elif 'Macro-F1:' in line or 'Macro F1:' in line:
            match = re.search(r'(\d+\.\d+)', line)
            if match:
                metrics['macro_f1'] = float(match.group(1))
    
    return metrics if metrics else None


def main():
    print("\n" + "=" * 80)
    print("LLM基线对比分析")
    print("=" * 80)

    experiments = [
        {
            'name': 'Llama-3.1-8B (QLoRA)',
            'report': '../llm_baseline_llama/test_report.txt',
            'params': '8B (1% tuned)',
            'type': 'LLM Fine-tuning'
        },
        {
            'name': 'Qwen2-7B (QLoRA)',
            'report': '../llm_baseline_qwen/test_report.txt',
            'params': '7B (1% tuned)',
            'type': 'LLM Fine-tuning'
        },
        {
            'name': 'DeBERTa-v3 (单任务)',
            'report': '../baseline_deberta/test_report.txt',
            'params': '184M',
            'type': 'PLM Baseline'
        },
        {
            'name': '混合模型 (多任务)',
            'report': '../results_group2_proposed/metrics.txt',
            'params': '184M + API',
            'type': 'Hybrid (Proposed)',
            'is_hybrid': True
        },
    ]
    
    results = []
    
    for exp in experiments:
        print(f"\n解析: {exp['name']}")
        
        is_hybrid = exp.get('is_hybrid', False)
        
        if is_hybrid:
            metrics = parse_hybrid_metrics(exp['report'])
        else:
            metrics = parse_test_report(exp['report'])
        
        if metrics:
            results.append({
                '模型': exp['name'],
                '类型': exp['type'],
                '参数量': exp['params'],
                '准确率': f"{metrics['accuracy']:.4f}",
                'Macro-F1': f"{metrics['macro_f1']:.4f}",
                '准确率(%)': metrics['accuracy'] * 100,
                'F1值': metrics['macro_f1']
            })
            print(f"  ✓ 准确率: {metrics['accuracy']:.4f}, Macro-F1: {metrics['macro_f1']:.4f}")
        else:
            print(f"  ✗ 文件不存在或解析失败: {exp['report']}")
    
    if not results:
        print("\n[错误] 没有找到任何有效的实验结果")
        return
    
    df = pd.DataFrame(results)
    

    print("\n" + "=" * 80)
    print("对比结果")
    print("=" * 80)
    display_df = df[['模型', '类型', '参数量', '准确率', 'Macro-F1']]
    print(display_df.to_string(index=False))
    

    output_csv = '../llm_baseline_comparison.csv'
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n✓ 结果已保存: {output_csv}")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    

    colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
    axes[0].bar(range(len(df)), df['准确率(%)'], color=colors[:len(df)])
    axes[0].set_xticks(range(len(df)))
    axes[0].set_xticklabels(df['模型'], rotation=20, ha='right')
    axes[0].set_ylabel('准确率 (%)', fontsize=12)
    axes[0].set_title('准确率对比', fontsize=14, fontweight='bold')
    axes[0].set_ylim([0, 100])
    axes[0].grid(True, alpha=0.3, axis='y')
    

    for i, y in enumerate(df['准确率(%)']):
        axes[0].text(i, y + 1, f'{y:.2f}%', ha='center', fontsize=10)
    

    axes[1].bar(range(len(df)), df['F1值'], color=colors[:len(df)])
    axes[1].set_xticks(range(len(df)))
    axes[1].set_xticklabels(df['模型'], rotation=20, ha='right')
    axes[1].set_ylabel('Macro-F1', fontsize=12)
    axes[1].set_title('Macro-F1对比', fontsize=14, fontweight='bold')
    axes[1].set_ylim([0, 1.0])
    axes[1].grid(True, alpha=0.3, axis='y')
    

    for i, y in enumerate(df['F1值']):
        axes[1].text(i, y + 0.02, f'{y:.4f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    output_png = '../llm_baseline_comparison.png'
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"✓ 对比图已保存: {output_png}")
    
    if len(df) >= 2:
        print("\n" + "=" * 80)
        print("性能提升分析")
        print("=" * 80)
        

        hybrid = df[df['类型'] == 'Hybrid (Proposed)'].iloc[0] if any(df['类型'] == 'Hybrid (Proposed)') else None
        
        if hybrid is not None:
            print(f"\n混合模型 vs LLM基线:")
            
            for i, row in df.iterrows():
                if 'LLM Fine-tuning' in row['类型']:
                    acc_diff = hybrid['准确率(%)'] - row['准确率(%)']
                    f1_diff = hybrid['F1值'] - row['F1值']
                    
                    print(f"\nvs {row['模型']}:")
                    print(f"  准确率提升: {acc_diff:+.2f} 个百分点")
                    print(f"  Macro-F1提升: {f1_diff:+.4f}")
                    print(f"  相对提升: {acc_diff/row['准确率(%)']*100:+.1f}%")


if __name__ == '__main__':
    main()
