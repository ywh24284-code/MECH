#!/usr/bin/env python3
"""
Prompting Baseline Comparison script
Compare DeepSeek/GPT-4o (Zero-shot/Few-shot) with hybrid model performance
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import re


def parse_metrics(metrics_file):
    """Parse metrics.json"""
    if not os.path.exists(metrics_file):
        return None
    
    with open(metrics_file, 'r') as f:
        return json.load(f)


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
        elif 'API Call Rate:' in line or 'API' in line and '%' in line:
            match = re.search(r'(\d+\.\d+)%', line)
            if match:
                metrics['api_call_rate'] = float(match.group(1)) / 100
    
    return metrics


def main():
    print("\n" + "=" * 80)
    print("Prompting Baseline Comparison")
    print("=" * 80)
    
    # Experiment configuration
    experiments = [
        {
            'name': 'DeepSeek Zero-shot',
            'metrics': '../prompting_deepseek_zeroshot/metrics.json',
            'type': 'Prompting',
            'model': 'DeepSeek-v3',
            'mode': 'Zero-shot'
        },
        {
            'name': 'DeepSeek Few-shot',
            'metrics': '../prompting_deepseek_fewshot/metrics.json',
            'type': 'Prompting',
            'model': 'DeepSeek-v3',
            'mode': 'Few-shot'
        },
        {
            'name': 'GPT-4o Zero-shot',
            'metrics': '../prompting_gpt4o_zeroshot/metrics.json',
            'type': 'Prompting',
            'model': 'GPT-4o',
            'mode': 'Zero-shot'
        },
        {
            'name': 'GPT-4o Few-shot',
            'metrics': '../prompting_gpt4o_fewshot/metrics.json',
            'type': 'Prompting',
            'model': 'GPT-4o',
            'mode': 'Few-shot'
        },
        {
            'name': 'Hybrid Model',
            'metrics': '../results_group2_proposed/metrics.txt',
            'type': 'Hybrid',
            'model': 'DeBERTa + DeepSeek',
            'mode': 'Risk Routing',
            'is_hybrid': True
        },
    ]
    
    results = []
    
    for exp in experiments:
        print(f"\nParsing: {exp['name']}")
        
        is_hybrid = exp.get('is_hybrid', False)
        
        if is_hybrid:
            metrics = parse_hybrid_metrics(exp['metrics'])
        else:
            metrics = parse_metrics(exp['metrics'])
        
        if metrics:
            results.append({
                'Method': exp['name'],
                'Type': exp['type'],
                'Model': exp['model'],
                'Mode': exp['mode'],
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Macro-F1': f"{metrics['macro_f1']:.4f}",
                'API Call Rate': f"{metrics.get('api_call_rate', 1.0)*100:.1f}%",
                'Accuracy(%)': metrics['accuracy'] * 100,
                'F1': metrics['macro_f1'],
                'API Rate': metrics.get('api_call_rate', 1.0)
            })
            print(f"  ✓ Accuracy: {metrics['accuracy']:.4f}, Macro-F1: {metrics['macro_f1']:.4f}, API: {metrics.get('api_call_rate', 1.0)*100:.1f}%")
        else:
            print(f"  ✗ File not found or parse failed: {exp['metrics']}")
    
    if not results:
        print("\n[Error] No valid experiment results found")
        return
    
    df = pd.DataFrame(results)
    
    # Display table
    print("\n" + "=" * 80)
    print("Comparison Results")
    print("=" * 80)
    display_df = df[['Method', 'Model', 'Mode', 'Accuracy', 'Macro-F1', 'API Call Rate']]
    print(display_df.to_string(index=False))
    
    # Save CSV
    output_csv = '../prompting_comparison.csv'
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n✓ Results saved: {output_csv}")
    
    # Plot - performance comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    colors = ['#3498db', '#e74c3c', '#9b59b6', '#f39c12', '#2ecc71']
    
    # Accuracy comparison
    axes[0].bar(range(len(df)), df['Accuracy(%)'], color=colors[:len(df)])
    axes[0].set_xticks(range(len(df)))
    axes[0].set_xticklabels(df['Method'], rotation=30, ha='right')
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0].set_ylim([0, 100])
    axes[0].grid(True, alpha=0.3, axis='y')
    
    for i, y in enumerate(df['Accuracy(%)']):
        axes[0].text(i, y + 1, f'{y:.2f}%', ha='center', fontsize=9)
    
    axes[1].bar(range(len(df)), df['F1'], color=colors[:len(df)])
    axes[1].set_xticks(range(len(df)))
    axes[1].set_xticklabels(df['Method'], rotation=30, ha='right')
    axes[1].set_ylabel('Macro-F1', fontsize=12)
    axes[1].set_title('Macro-F1 Comparison', fontsize=14, fontweight='bold')
    axes[1].set_ylim([0, 1.0])
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for i, y in enumerate(df['F1']):
        axes[1].text(i, y + 0.02, f'{y:.4f}', ha='center', fontsize=9)
    
  
    axes[2].bar(range(len(df)), df['API Rate'] * 100, color=colors[:len(df)])
    axes[2].set_xticks(range(len(df)))
    axes[2].set_xticklabels(df['Method'], rotation=30, ha='right')
    axes[2].set_ylabel('API Call Rate (%)', fontsize=12)
    axes[2].set_title('API Call Rate Comparison', fontsize=14, fontweight='bold')
    axes[2].set_ylim([0, 110])
    axes[2].grid(True, alpha=0.3, axis='y')
    
    for i, y in enumerate(df['API Rate'] * 100):
        axes[2].text(i, y + 2, f'{y:.1f}%', ha='center', fontsize=9)
    
    plt.tight_layout()
    output_png = '../prompting_comparison.png'
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison chart saved: {output_png}")
    plt.close()

    print("\n" + "=" * 80)
    print("Cost-Efficiency Analysis")
    print("=" * 80)
    
    hybrid = df[df['Type'] == 'Hybrid'].iloc[0] if any(df['Type'] == 'Hybrid') else None
    
    if hybrid is not None:
        print(f"\nHybrid Model vs Prompting Baselines:")
        print(f"\n{'Method':<20} {'Acc. Gain':<12} {'API Saved':<12} {'Efficiency'}")
        print("-" * 70)
        
        for i, row in df.iterrows():
            if row['Type'] == 'Prompting':
                acc_diff = hybrid['Accuracy(%)'] - row['Accuracy(%)']
                api_save = row['API Rate'] - hybrid['API Rate']
                
                # Efficiency = performance gain / API increase (higher is better)
                if api_save > 0:
                    efficiency = acc_diff / (api_save * 100)
                else:
                    efficiency = float('inf') if acc_diff > 0 else 0
                
                print(f"{row['Method']:<20} {acc_diff:+6.2f}%      {api_save*100:+6.1f}%      {efficiency:+.4f}")
        
        print("\nInterpretation:")
        print("- Acc. gain: positive means hybrid model is better")
        print("- API saved: positive means Prompting uses more API")
        print("- Efficiency: hybrid model achieves higher performance with fewer API calls")
    
  
    cost_analysis_path = '../cost_efficiency_analysis.txt'
    with open(cost_analysis_path, 'w', encoding='utf-8') as f:
        f.write("Cost-Efficiency Analysis\n")
        f.write("=" * 80 + "\n\n")
        
        if hybrid is not None:
            f.write(f"Baseline: Hybrid Model\n")
            f.write(f"  Accuracy: {hybrid['Accuracy(%)']:.2f}%\n")
            f.write(f"  Macro-F1: {hybrid['F1']:.4f}\n")
            f.write(f"  API Call Rate: {hybrid['API Rate']*100:.1f}%\n\n")

            f.write("Comparison with Prompting Baselines:\n\n")
            
            for i, row in df.iterrows():
                if row['Type'] == 'Prompting':
                    acc_diff = hybrid['Accuracy(%)'] - row['Accuracy(%)']
                    f1_diff = hybrid['F1'] - row['F1']
                    api_save = row['API Rate'] - hybrid['API Rate']
                    
                    f.write(f"{row['Method']}:\n")
                    f.write(f"  Accuracy gap: {acc_diff:+.2f} percentage points\n")
                    f.write(f"  Macro-F1 gap: {f1_diff:+.4f}\n")
                    f.write(f"  API call gap: {api_save*100:+.1f}% (positive = Prompting uses more)\n")
                    f.write(f"  Cost ratio: {row['API Rate']/hybrid['API Rate']:.2f}x (Prompting/Hybrid)\n\n")
    
    print(f"\n✓ Cost-efficiency analysis saved: {cost_analysis_path}")

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
