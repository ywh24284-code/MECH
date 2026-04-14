#!/usr/bin/env python3
"""
PLM Baseline Comparison Script

Compares BERT-base, RoBERTa-large and hybrid model performance
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import re

# Font config
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def parse_test_report(report_file):
    """Parse test report file"""
    if not os.path.exists(report_file):
        return None
    
    with open(report_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract accuracy and F1
    acc_match = re.search(r'Accuracy:\s*(\d+\.\d+)', content)
    f1_match = re.search(r'Macro-F1:\s*(\d+\.\d+)', content)
    
    if acc_match and f1_match:
        return {
            'accuracy': float(acc_match.group(1)),
            'macro_f1': float(f1_match.group(1))
        }
    return None


def parse_hybrid_metrics(metrics_file):
    """Parse hybrid model metrics.txt"""
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
    print("PLM Baseline Comparison")
    print("=" * 80)
    
    # Experiment config
    experiments = [
        {
            'name': 'BERT-base',
            'report': '../baseline_bert/test_report.txt',
            'params': '110M',
            'type': 'PLM Baseline'
        },
        {
            'name': 'RoBERTa-large',
            'report': '../baseline_roberta/test_report.txt',
            'params': '355M',
            'type': 'PLM Baseline'
        },
        {
            'name': 'DeBERTa-v3-base',
            'report': '../baseline_deberta/test_report.txt',
            'params': '184M',
            'type': 'PLM Baseline'
        },
        {
            'name': 'Hybrid (Single-task)',
            'report': '../results_group1_baseline/metrics.txt',
            'params': '184M + API',
            'type': 'Hybrid (Single-task)',
            'is_hybrid': True
        },
        {
            'name': 'Hybrid (Multi-task)',
            'report': '../results_group2_proposed/metrics.txt',
            'params': '184M + API',
            'type': 'Hybrid (Multi-task)',
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
            print(f"  ✓ Accuracy: {metrics['accuracy']:.4f}, Macro-F1: {metrics['macro_f1']:.4f}")
        else:
            print(f"  ✗ File not found or parse failed: {exp['report']}")
    
    if not results:
        print("\n[Error] No valid experiment results found")
        return
    
    df = pd.DataFrame(results)
    
    # Display table
    print("\n" + "=" * 80)
    print("Comparison Results")
    print("=" * 80)
    display_df = df[['Model', 'Type', 'Params', 'Accuracy', 'Macro-F1']]
    print(display_df.to_string(index=False))
    
    # Save CSV
    output_csv = '../baseline_comparison.csv'
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n✓ Results saved: {output_csv}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy Comparison
    colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71']
    axes[0].bar(df['Model'], df['Accuracy(%)'], color=colors[:len(df)])
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0].set_ylim([0, 100])
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (x, y) in enumerate(zip(df['Model'], df['Accuracy(%)'])):
        axes[0].text(i, y + 1, f'{y:.2f}%', ha='center', fontsize=10)
    
    # Rotate x-axis labels
    axes[0].tick_params(axis='x', rotation=15)

    # Macro-F1 Comparison
    axes[1].bar(df['Model'], df['F1'], color=colors[:len(df)])
    axes[1].set_ylabel('Macro-F1', fontsize=12)
    axes[1].set_title('Macro-F1 Comparison', fontsize=14, fontweight='bold')
    axes[1].set_ylim([0, 1.0])
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (x, y) in enumerate(zip(df['Model'], df['F1'])):
        axes[1].text(i, y + 0.02, f'{y:.4f}', ha='center', fontsize=10)
    
    # Rotate x-axis labels
    axes[1].tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    output_png = '../baseline_comparison.png'
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison chart saved: {output_png}")
    
    # Performance improvement analysis
    if len(df) >= 3:
        print("\n" + "=" * 80)
        print("Performance Improvement Analysis")
        print("=" * 80)
        
        # Find hybrid model (multi-task)
        hybrid_multi = df[df['Model'].str.contains('Multi-task')].iloc[0] if any(df['Model'].str.contains('Multi-task')) else None

        if hybrid_multi is not None:
            print(f"\nHybrid (Multi-task) vs PLM Baselines:")

            for i, row in df.iterrows():
                if 'PLM Baseline' in row['Type']:
                    acc_diff = hybrid_multi['Accuracy(%)'] - row['Accuracy(%)']
                    f1_diff = hybrid_multi['F1'] - row['F1']

                    print(f"\nvs {row['Model']}:")
                    print(f"  Accuracy improvement: {acc_diff:+.2f} percentage points")
                    print(f"  Macro-F1 improvement: {f1_diff:+.4f}")
                    print(f"  Relative improvement: {acc_diff/row['Accuracy(%)']*100:+.1f}%")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
