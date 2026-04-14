#!/usr/bin/env python3
"""
LLM Baseline Comparison Script

Compare Llama-3.1-8B, Qwen2-7B and Hybrid model performance
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
    
    acc_match = re.search(r'Accuracy:\s*(\d+\.\d+)', content)
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
        if 'Accuracy:' in line:
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
    print("LLM Baseline Comparison")
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
            'name': 'DeBERTa-v3 (Single-task)',
            'report': '../baseline_deberta/test_report.txt',
            'params': '184M',
            'type': 'PLM Baseline'
        },
        {
            'name': 'Hybrid (Multi-task)',
            'report': '../results_group2_proposed/metrics.txt',
            'params': '184M + API',
            'type': 'Hybrid (Proposed)',
            'is_hybrid': True
        },
    ]
    
    results = []
    
    for exp in experiments:
        print(f"\nParsing: {exp['name']}")
        
        is_hybrid = exp.get('is_hybrid', False)
        
        if is_hybrid:
            metrics = parse_hybrid_metrics(exp['report'])
        else:
            metrics = parse_test_report(exp['report'])
        
        if metrics:
            results.append({
                'Model': exp['name'],
                'Type': exp['type'],
                'Params': exp['params'],
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Macro-F1': f"{metrics['macro_f1']:.4f}",
                'Accuracy(%)': metrics['accuracy'] * 100,
                'F1': metrics['macro_f1']
            })
            print(f"  > Accuracy: {metrics['accuracy']:.4f}, Macro-F1: {metrics['macro_f1']:.4f}")
        else:
            print(f"  x File not found or parsing failed: {exp['report']}")
    
    if not results:
        print("\n[Error] No valid experiment results found")
        return
    
    df = pd.DataFrame(results)
    

    print("\n" + "=" * 80)
    print("Comparison Results")
    print("=" * 80)
    display_df = df[['Model', 'Type', 'Params', 'Accuracy', 'Macro-F1']]
    print(display_df.to_string(index=False))
    

    output_csv = '../llm_baseline_comparison.csv'
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n> Results saved: {output_csv}")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    

    colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
    axes[0].bar(range(len(df)), df['Accuracy(%)'], color=colors[:len(df)])
    axes[0].set_xticks(range(len(df)))
    axes[0].set_xticklabels(df['Model'], rotation=20, ha='right')
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0].set_ylim([0, 100])
    axes[0].grid(True, alpha=0.3, axis='y')
    

    for i, y in enumerate(df['Accuracy(%)']):
        axes[0].text(i, y + 1, f'{y:.2f}%', ha='center', fontsize=10)
    

    axes[1].bar(range(len(df)), df['F1'], color=colors[:len(df)])
    axes[1].set_xticks(range(len(df)))
    axes[1].set_xticklabels(df['Model'], rotation=20, ha='right')
    axes[1].set_ylabel('Macro-F1', fontsize=12)
    axes[1].set_title('Macro-F1 Comparison', fontsize=14, fontweight='bold')
    axes[1].set_ylim([0, 1.0])
    axes[1].grid(True, alpha=0.3, axis='y')
    

    for i, y in enumerate(df['F1']):
        axes[1].text(i, y + 0.02, f'{y:.4f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    output_png = '../llm_baseline_comparison.png'
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"> Comparison chart saved: {output_png}")
    
    if len(df) >= 2:
        print("\n" + "=" * 80)
        print("Performance Improvement Analysis")
        print("=" * 80)
        

        hybrid = df[df['Type'] == 'Hybrid (Proposed)'].iloc[0] if any(df['Type'] == 'Hybrid (Proposed)') else None
        
        if hybrid is not None:
            print(f"\nHybrid model vs LLM baselines:")

            for i, row in df.iterrows():
                if 'LLM Fine-tuning' in row['Type']:
                    acc_diff = hybrid['Accuracy(%)'] - row['Accuracy(%)']
                    f1_diff = hybrid['F1'] - row['F1']

                    print(f"\nvs {row['Model']}:")
                    print(f"  Accuracy improvement: {acc_diff:+.2f} percentage points")
                    print(f"  Macro-F1 improvement: {f1_diff:+.4f}")
                    print(f"  Relative improvement: {acc_diff/row['Accuracy(%)']*100:+.1f}%")


if __name__ == '__main__':
    main()
