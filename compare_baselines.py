#!/usr/bin/env python3
"""
PLM基线对比分析脚本

对比 BERT-base, RoBERTa-large 与混合模型的性能
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import re

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def parse_test_report(report_file):
    """解析测试报告文件"""
    if not os.path.exists(report_file):
        return None
    
    with open(report_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取准确率和F1
    acc_match = re.search(r'准确率:\s*(\d+\.\d+)', content)
    f1_match = re.search(r'Macro-F1:\s*(\d+\.\d+)', content)
    
    if acc_match and f1_match:
        return {
            'accuracy': float(acc_match.group(1)),
            'macro_f1': float(f1_match.group(1))
        }
    return None


def parse_hybrid_metrics(metrics_file):
    """解析混合模型的metrics.txt"""
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
    print("PLM基线对比分析")
    print("=" * 80)
    
    # 实验配置
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
            'name': '混合模型 (单任务)',
            'report': '../results_group1_baseline/metrics.txt',
            'params': '184M + API',
            'type': 'Hybrid (Single-task)',
            'is_hybrid': True
        },
        {
            'name': '混合模型 (多任务)',
            'report': '../results_group2_proposed/metrics.txt',
            'params': '184M + API',
            'type': 'Hybrid (Multi-task)',
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
    
    # 显示表格
    print("\n" + "=" * 80)
    print("对比结果")
    print("=" * 80)
    display_df = df[['模型', '类型', '参数量', '准确率', 'Macro-F1']]
    print(display_df.to_string(index=False))
    
    # 保存CSV
    output_csv = '../baseline_comparison.csv'
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n✓ 结果已保存: {output_csv}")
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 准确率对比
    colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71']
    axes[0].bar(df['模型'], df['准确率(%)'], color=colors[:len(df)])
    axes[0].set_ylabel('准确率 (%)', fontsize=12)
    axes[0].set_title('准确率对比', fontsize=14, fontweight='bold')
    axes[0].set_ylim([0, 100])
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, (x, y) in enumerate(zip(df['模型'], df['准确率(%)'])):
        axes[0].text(i, y + 1, f'{y:.2f}%', ha='center', fontsize=10)
    
    # 旋转x轴标签
    axes[0].tick_params(axis='x', rotation=15)
    
    # Macro-F1对比
    axes[1].bar(df['模型'], df['F1值'], color=colors[:len(df)])
    axes[1].set_ylabel('Macro-F1', fontsize=12)
    axes[1].set_title('Macro-F1对比', fontsize=14, fontweight='bold')
    axes[1].set_ylim([0, 1.0])
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, (x, y) in enumerate(zip(df['模型'], df['F1值'])):
        axes[1].text(i, y + 0.02, f'{y:.4f}', ha='center', fontsize=10)
    
    # 旋转x轴标签
    axes[1].tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    output_png = '../baseline_comparison.png'
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"✓ 对比图已保存: {output_png}")
    
    # 性能提升分析
    if len(df) >= 3:
        print("\n" + "=" * 80)
        print("性能提升分析")
        print("=" * 80)
        
        # 找到混合模型（多任务）
        hybrid_multi = df[df['模型'].str.contains('多任务')].iloc[0] if any(df['模型'].str.contains('多任务')) else None
        
        if hybrid_multi is not None:
            print(f"\n混合模型 (多任务) vs PLM基线:")
            
            for i, row in df.iterrows():
                if 'PLM Baseline' in row['类型']:
                    acc_diff = hybrid_multi['准确率(%)'] - row['准确率(%)']
                    f1_diff = hybrid_multi['F1值'] - row['F1值']
                    
                    print(f"\nvs {row['模型']}:")
                    print(f"  准确率提升: {acc_diff:+.2f} 个百分点")
                    print(f"  Macro-F1提升: {f1_diff:+.4f}")
                    print(f"  相对提升: {acc_diff/row['准确率(%)']*100:+.1f}%")
    
    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()
