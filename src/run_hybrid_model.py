#!/usr/bin/env python3
"""
混合模型快速启动脚本
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from transformers import AutoTokenizer, AutoModel, AutoConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def train_discriminative_model(data_dir='dataset_split_result_v4'):
    """训练判别模型"""
    print("=" * 80)
    print("Step 1: 训练判别模型")
    print("=" * 80)

    if not os.path.exists(data_dir):
        print(f"[错误] 数据目录不存在: {data_dir}")
        print("请确保数据已准备好")
        return False

    import discriminative_model_training
    try:
        discriminative_model_training.main()
        print("\n" + "=" * 80)
        print("✓ 判别模型训练完成!")
        print("=" * 80)
        return True
    except Exception as e:
        print(f"\n[错误] 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_single_prediction():
    print("=" * 80)
    print("Step 2: 测试混合模型 (单句预测)")
    print("=" * 80)

    from hybrid_opinion_classifier import HybridOpinionClassifier

    model_dir = "../discriminative_model_outputs_v4_fix"
    model_file = os.path.join(model_dir, 'best_model.pth')
    config_file = os.path.join(model_dir, 'config.json')

    if not os.path.exists(config_file):
        print(f"[错误] 配置文件不存在: {config_file}")
        print("请先运行: python3 run_hybrid_model.py --mode train")
        return False

    if not os.path.exists(model_file):
        print(f"[错误] 模型文件不存在: {model_file}")
        print("请先运行: python3 run_hybrid_model.py --mode train")
        return False

    print("\n正在加载模型...")
    classifier = HybridOpinionClassifier(
        discriminative_model_dir=model_dir,
        use_generative=True,
        generative_model="deepseek-v3",
        cascade_threshold=0.85,
        cascade_threshold_irrelevant=0.75,  
        decision_threshold=0.7,  
        topk_threshold=0.2, 
        prefer_generative=True,  
        context_window=5
    )

    # 测试示例
    examples = [
        {
            'sentence': 'I would like someone to tell me where you would place one.',
            'speaker': 'T',
            'context': [
                {'speaker': 'T', 'sentence': 'Today we are talking about fractions.', 'label': 'New'}
            ]
        },
        {
            'sentence': 'I think we should use the number line.',
            'speaker': 'S',
            'context': [
                {'speaker': 'T', 'sentence': 'Where would you place one?', 'label': 'New'}
            ]
        }
    ]

    print("\n" + "=" * 80)
    print("测试示例")
    print("=" * 80)

    for i, example in enumerate(examples, 1):
        print(f"\n【示例 {i}】")
        print(f"输入: [{example['speaker']}] {example['sentence']}")

        prev_speaker = example['context'][-1]['speaker'] if example['context'] else None

        result = classifier.classify_single(
            example['sentence'],
            example['speaker'],
            example['context'],
            prev_speaker
        )

        print(f"\n最终预测: {result['final_label']}")
        print(f"\n判别模型:")
        print(f"  - 预测: {result['discriminative_result']['predicted_label']}")
        print(f"  - 置信度: {result['discriminative_result']['confidence_score']:.4f}")
        print(f"  - Top-3排名:")
        for rank_info in result['discriminative_result']['top_k_rankings'][:3]:
            print(f"    {rank_info['rank']}. {rank_info['label']}: {rank_info['probability']:.4f}")

        if result['generative_result']:
            print(f"\n生成模型:")
            print(f"  - 预测: {result['generative_result']['predicted_label']}")

        print(f"\n决策信息:")
        print(f"  - 策略: {result['decision_info']['strategy']}")
        print(f"  - 理由: {result['decision_info']['reason']}")
        print("-" * 80)

    return True


def batch_process(model_dir=None, data_dir='dataset_split_result_v4', output_dir=None, enable_risk_routing='auto'):
    print("=" * 80)
    print("Step 3: 批量处理CSV文件")
    print("=" * 80)

    from hybrid_opinion_classifier import HybridOpinionClassifier

    if model_dir is None:
        model_dir = "../discriminative_model_outputs_v4_fix" 
    input_csv = f"{data_dir}/test.csv"

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_csv = f"{output_dir}/hybrid_predictions.csv"
    elif 'fewshot' in model_dir:
        output_csv = f"../hybrid_predictions_{os.path.basename(model_dir)}.csv"
    else:
        output_csv = "../hybrid_predictions.csv"

    if not os.path.exists(input_csv):
        print(f"[错误] 输入文件不存在: {input_csv}")
        return False

    model_file = os.path.join(model_dir, 'best_model.pth')
    config_file = os.path.join(model_dir, 'config.json')

    if not os.path.exists(config_file):
        print(f"[错误] 配置文件不存在: {config_file}")
        print("请先运行: python3 run_hybrid_model.py --mode train")
        return False

    if not os.path.exists(model_file):
        print(f"[错误] 模型文件不存在: {model_file}")
        print("请先运行: python3 run_hybrid_model.py --mode train")
        return False

    df = pd.read_csv(input_csv)

    use_generative = True 

    if 'class' in df.columns:
        classes = df['class'].unique()
        print(f"\n数据集包含 {len(classes)} 堂课")
        print(f"总样本数: {len(df)}")

        class_counts = df['class'].value_counts().sort_index()
        print(f"\n前10堂课的样本分布:")
        for cls, count in class_counts.head(10).items():
            print(f"  课堂 {cls}: {count} 条")

        print("\n" + "=" * 80)
        print("选择处理模式:")
        print("  1. 测试单堂课（快速测试，低成本）")
        print("  2. 测试前N堂课（中等规模测试）")
        print("  3. 处理全部数据（完整评估，高成本）")
        print("  4. 仅使用判别模型（免费，快速）")
        print("=" * 80)

        choice = input("\n请选择模式 (1/2/3/4): ").strip()

        if choice == '1':
            class_id = input(f"请输入要测试的课堂ID (例如 {classes[0]}): ").strip()
            try:
                class_id = int(class_id)
                df_subset = df[df['class'] == class_id].copy()
                if len(df_subset) == 0:
                    print(f"[错误] 课堂 {class_id} 不存在")
                    return False
                use_generative = True
                output_csv = f"../hybrid_predictions_class_{class_id}.csv"
                print(f"\n将测试课堂 {class_id}，共 {len(df_subset)} 条样本")
            except ValueError:
                print("[错误] 无效的课堂ID")
                return False

        elif choice == '2':
            n = input("请输入要测试的课堂数量 (例如 3): ").strip()
            try:
                n = int(n)
                selected_classes = sorted(classes)[:n]
                df_subset = df[df['class'].isin(selected_classes)].copy()
                use_generative = True
                output_csv = f"../hybrid_predictions_top{n}_classes.csv"
                print(f"\n将测试前 {n} 堂课，共 {len(df_subset)} 条样本")
                print(f"课堂ID: {selected_classes}")
            except ValueError:
                print("[错误] 无效的数量")
                return False

        elif choice == '3':
            df_subset = df.copy()
            use_generative = True
            confirm = input(f"\n⚠️  将处理全部 {len(df)} 条样本，可能产生较高API费用。确认? (yes/no): ").strip().lower()
            if confirm != 'yes':
                print("已取消")
                return False

        elif choice == '4':
            df_subset = df.copy()
            use_generative = False
            print(f"\n将使用仅判别模型处理全部 {len(df)} 条样本（免费）")

        else:
            print("[错误] 无效的选择")
            return False
    else:
        df_subset = df.copy()
        use_generative = False 
        print(f"\n数据集无class列，将处理全部 {len(df)} 条样本")
        use_gen_input = input("是否启用生成模型? (y/n): ").strip().lower()
        if use_gen_input == 'y':
            use_generative = True

    use_knn_icl = False
    knn_datastore_path = "./knn_datastore.npz"
    knn_k = 3

    if use_generative:
        print("\n" + "-" * 50)
        print("配置运行模式:")
        print("  1. 标准混合模式 (默认): 启用级联路由，只对难样本调用 LLM (省钱、快)")
        print("  2. 强制全量模式 (调试): 禁用快速通道，对所有样本调用 LLM (验证 LLM 性能)")
        print("  3. kNN-ICL 增强模式 (实验): 启用 kNN 检索，为 LLM 提供上下文示例")
        print("-" * 50)
        mode_choice = input("请选择 (1/2/3) [默认1]: ").strip()

        if mode_choice == '2':
            c_thresh = 2.0
            c_thresh_irr = 2.0
            print("\n>> 已启用【强制全量模式】：将对所有样本调用 DeepSeek")
        elif mode_choice == '3':
            c_thresh = 0.96
            c_thresh_irr = 0.80
            use_knn_icl = True
            if not os.path.exists(knn_datastore_path):
                print(f"\n[警告] kNN 向量库不存在: {knn_datastore_path}")
                print("请先运行: python 师哥的实验/run_knn_icl.py --step 1")
                build_now = input("是否现在构建向量库? (y/n): ").strip().lower()
                if build_now == 'y':
                    import subprocess
                    subprocess.run(["python", "师哥的实验/build_knn_datastore.py"])
                    if not os.path.exists(knn_datastore_path):
                        print("[错误] 向量库构建失败，将禁用 kNN-ICL")
                        use_knn_icl = False
                else:
                    print("将禁用 kNN-ICL")
                    use_knn_icl = False
            if use_knn_icl:
                print(f"\n>> 已启用【kNN-ICL 增强模式】：将检索 Top-{knn_k} 相似示例")
        else:
            c_thresh = 0.96
            c_thresh_irr = 0.80
            print(f"\n>> 已启用【标准混合模式】：高阈值策略 (Content>{c_thresh}, Irr>{c_thresh_irr})")
    else:
        c_thresh = 0.96
        c_thresh_irr = 0.80

    if model_dir and ('fewshot' in model_dir or '20pct' in model_dir or '40pct' in model_dir or 
                      '60pct' in model_dir or '80pct' in model_dir):
        print("\n>> 检测到少样本模型，应用策略B（性能优先）...")
        

        if '20pct' in model_dir:
            base_threshold = 0.75
            irr_threshold = 0.70
            data_pct = 20
            print(f">> 少样本模式 (20%): 高阈值策略，大部分样本走 LLM (Content>{base_threshold}, Irr>{irr_threshold})")
        elif '40pct' in model_dir:
            base_threshold = 0.80
            irr_threshold = 0.75
            data_pct = 40
            print(f">> 少样本模式 (40%): 高阈值策略 (Content>{base_threshold}, Irr>{irr_threshold})")
        elif '60pct' in model_dir:
            base_threshold = 0.85
            irr_threshold = 0.80
            data_pct = 60
            print(f">> 少样本模式 (60%): 高阈值策略 (Content>{base_threshold}, Irr>{irr_threshold})")
        elif '80pct' in model_dir:
            base_threshold = 0.90
            irr_threshold = 0.85
            data_pct = 80
            print(f">> 少样本模式 (80%): 高阈值策略 (Content>{base_threshold}, Irr>{irr_threshold})")
        else:
            # 通用 fewshot 标识，使用保守策略
            base_threshold = 0.75
            irr_threshold = 0.70
            data_pct = None
            print(f">> 少样本模式 (通用): 高阈值策略 (Content>{base_threshold}, Irr>{irr_threshold})")
        
        c_thresh = base_threshold
        c_thresh_irr = irr_threshold
        
        if data_pct:
            print(f"   预期 API 调用率: ~{100 - (data_pct * 0.3):.0f}% (性能优先策略)")
    elif model_dir and 'singletask' not in model_dir:
        c_thresh = 0.95
        c_thresh_irr = 0.85
        print(f"\n>> 全量模型: 标准高阈值策略 (Content>{c_thresh}, Irr>{c_thresh_irr})")
    print("\n正在加载模型...")
    
  
    if enable_risk_routing == 'auto':
        final_enable_risk_routing = True  # 默认启用
        print(f">> 风险路由: 自动模式 (将根据模型类型决定)")
    elif enable_risk_routing == 'true':
        final_enable_risk_routing = True
        print(f">> 风险路由: 强制启用")
    else:  # 'false'
        final_enable_risk_routing = False
        print(f">> 风险路由: 强制禁用")
    
    classifier = HybridOpinionClassifier(
        discriminative_model_dir=model_dir,
        use_generative=use_generative,
        generative_model="deepseek-v3",
        cascade_threshold=c_thresh,
        cascade_threshold_irrelevant=c_thresh_irr,
        decision_threshold=0.0,  
        topk_threshold=0.0, 
        prefer_generative=True,
        context_window=5,
        use_knn_icl=use_knn_icl,
        knn_datastore_path=knn_datastore_path if use_knn_icl else None,
        knn_k=knn_k, 
        enable_risk_routing=final_enable_risk_routing
    )

    temp_csv = "../temp_subset.csv"
    df_subset.to_csv(temp_csv, index=False)

    print(f"\n正在处理...")
    try:
        classifier.classify_file(
            input_csv=temp_csv,
            output_csv=output_csv,
            text_column='Sentence',
            speaker_column='Speaker',
            label_column='label',
            context_window=5
        )
    except Exception as e:
        print(f"\n[错误] 处理过程中出错: {e}")
        import traceback
        traceback.print_exc()

    if os.path.exists(temp_csv):
        os.remove(temp_csv)

    print("\n" + "=" * 80)
    print(f"✓ 批量处理完成!")
    print(f"  输出文件: {output_csv}")
    print(f"  处理样本数: {len(df_subset)}")
    print("=" * 80)
    return True


def evaluate_performance(output_dir=None):
    print("=" * 80)
    print("Step 4: 评估混合模型性能")
    print("=" * 80)

    import glob
    if output_dir:
        search_patterns = [
            f"{output_dir}/hybrid_predictions*.csv",
            f"{output_dir}/*.csv"
        ]
    else:
        search_patterns = [
            "../results_*/hybrid_predictions*.csv",  
            "../hybrid_predictions*.csv"  
        ]

    files = []
    for pattern in search_patterns:
        files.extend(glob.glob(pattern))

    if not files:
        print(f"[错误] 未找到结果文件")
        print(f"搜索路径: {search_patterns}")
        print("请先运行: python 师哥的实验/run_hybrid_model.py --mode batch")
        return False

    result_csv = max(files, key=os.path.getmtime)

    print(f"\n分析文件: {result_csv}")
    print(f"   文件大小: {os.path.getsize(result_csv) / 1024:.1f} KB")


    df = pd.read_csv(result_csv)
    print(f"数据集大小: {len(df)}")

    if 'true_label' not in df.columns or 'final_label' not in df.columns:
        print("[错误] 缺少必要的列: true_label 或 final_label")
        return False
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support,
        confusion_matrix, classification_report, matthews_corrcoef
    )
    from sklearn.preprocessing import LabelEncoder
    y_true = df['true_label']
    y_pred = df['final_label']
 
    print("\n" + "=" * 80)
    print("整体性能指标")
    print("=" * 80)
    accuracy = accuracy_score(y_true, y_pred)

    label_encoder = LabelEncoder()
    label_encoder.fit(list(y_true) + list(y_pred)) 
    y_true_encoded = label_encoder.transform(y_true)
    y_pred_encoded = label_encoder.transform(y_pred)
    mcc = matthews_corrcoef(y_true_encoded, y_pred_encoded)

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0
    )
    print(f"  准确率 (Accuracy):     {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"  MCC (Matthews相关系数): {mcc:.4f}")
    print(f"  宏平均精确率 (Macro Precision): {macro_precision:.4f}")
    print(f"  宏平均召回率 (Macro Recall):    {macro_recall:.4f}")
    print(f"  宏平均F1 (Macro F1):            {macro_f1:.4f}")
    print("\n" + "=" * 80)
    print("每类别性能指标")
    print("=" * 80)
    labels = ['Irrelevant', 'New', 'Strengthened', 'Weakened', 'Adopted', 'Refuted']
    present_labels = [l for l in labels if l in y_true.values or l in y_pred.values]

    print(f"\n{'类别':<15} {'精确率':<10} {'召回率':<10} {'F1分数':<10} {'样本数':<8}")
    print("-" * 60)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=present_labels, zero_division=0
    )

    for i, label in enumerate(present_labels):
        print(f"{label:<15} {precision[i]:<10.4f} {recall[i]:<10.4f} {f1[i]:<10.4f} {support[i]:<8}")
    print("-" * 60)
    print(f"{'宏平均':<15} {precision.mean():<10.4f} {recall.mean():<10.4f} {f1.mean():<10.4f} {support.sum():<8}")

    print("\n" + "=" * 80)
    print("混淆矩阵")
    print("=" * 80)

    cm = confusion_matrix(y_true, y_pred, labels=present_labels)

    header = "真实\\预测"
    print(f"\n{header:<15}", end="")
    for label in present_labels:
        print(f"{label[:10]:<12}", end="")
    print()
    print("-" * (15 + 12 * len(present_labels)))

    for i, label in enumerate(present_labels):
        print(f"{label:<15}", end="")
        for j in range(len(present_labels)):
            print(f"{cm[i][j]:<12}", end="")
        print()

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=present_labels,
        yticklabels=present_labels,
        cbar_kws={'label': 'Count'},
        linewidths=0.5,
        linecolor='gray'
    )
    plt.title('Confusion Matrix (Absolute Counts)', fontsize=16, pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()

    cm_count_path = os.path.join(output_dir, 'confusion_matrix_count.png')
    plt.savefig(cm_count_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 混淆矩阵(计数)已保存: {cm_count_path}")
    plt.close()

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2%',
        cmap='YlOrRd',
        xticklabels=present_labels,
        yticklabels=present_labels,
        cbar_kws={'label': 'Proportion'},
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor='gray'
    )
    plt.title('Confusion Matrix (Normalized by True Label)', fontsize=16, pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()

    cm_norm_path = os.path.join(output_dir, 'confusion_matrix_normalized.png')
    plt.savefig(cm_norm_path, dpi=300, bbox_inches='tight')
    print(f"✓ 混淆矩阵(归一化)已保存: {cm_norm_path}")
    plt.close()

    if 'strategy' in df.columns:
        print("\n" + "=" * 80)
        print("决策策略分析")
        print("=" * 80)

        strategy_counts = df['strategy'].value_counts()

        print(f"\n{'策略':<30} {'数量':<10} {'占比':<10} {'准确率':<10}")
        print("-" * 60)

        for strategy in strategy_counts.index:
            strategy_df = df[df['strategy'] == strategy]
            count = len(strategy_df)
            pct = count / len(df) * 100
            acc = (strategy_df['true_label'] == strategy_df['final_label']).mean()
            print(f"{strategy:<30} {count:<10} {pct:<10.1f}% {acc:<10.4f}")

    if 'disc_confidence' in df.columns:
        print("\n" + "=" * 80)
        print("判别模型置信度分析")
        print("=" * 80)

        correct_mask = (df['true_label'] == df['final_label'])

        print(f"\n  整体平均置信度:   {df['disc_confidence'].mean():.4f}")
        print(f"  中位数:           {df['disc_confidence'].median():.4f}")
        print(f"  标准差:           {df['disc_confidence'].std():.4f}")
        print(f"  最小值:           {df['disc_confidence'].min():.4f}")
        print(f"  最大值:           {df['disc_confidence'].max():.4f}")

        print(f"\n  正确预测平均置信度: {df[correct_mask]['disc_confidence'].mean():.4f}")
        print(f"  错误预测平均置信度: {df[~correct_mask]['disc_confidence'].mean():.4f}")

        print(f"\n  置信度区间分布:")
        bins = [0, 0.5, 0.7, 0.8, 0.9, 1.0]
        for i in range(len(bins) - 1):
            mask = (df['disc_confidence'] >= bins[i]) & (df['disc_confidence'] < bins[i + 1])
            count = mask.sum()
            if count > 0:
                acc = (df[mask]['true_label'] == df[mask]['final_label']).mean()
                print(f"    [{bins[i]:.1f}, {bins[i + 1]:.1f}): {count:>4} 样本, 准确率 {acc:.4f}")

    if 'disc_label' in df.columns and 'gen_label' in df.columns:
        print("\n" + "=" * 80)
        print(" 判别模型 vs 生成模型对比")
        print("=" * 80)

        disc_acc = (df['disc_label'] == df['true_label']).mean()
        print(f"\n  判别模型单独准确率: {disc_acc:.4f} ({disc_acc * 100:.2f}%)")

        gen_mask = df['gen_label'].notna()
        if gen_mask.sum() > 0:
            gen_acc = (df[gen_mask]['gen_label'] == df[gen_mask]['true_label']).mean()
            print(f"  生成模型单独准确率: {gen_acc:.4f} ({gen_acc * 100:.2f}%)")

        hybrid_acc = accuracy
        print(f"  混合模型准确率:     {hybrid_acc:.4f} ({hybrid_acc * 100:.2f}%)")

        improvement = (hybrid_acc - disc_acc) * 100
        print(f"\n  混合模型提升: {improvement:+.2f} 个百分点")

    print("\n" + "=" * 80)
    print("✓ 评估完成!")
    print("=" * 80)
    return True


def run_group_experiments(model_dir=None, data_dir='dataset_split_result_v4', base_output_dir='results_groups'):
    print("=" * 80)
    print("三组对比实验：验证对话行为对观点演化的辅助作用")
    print("=" * 80)
    
    from hybrid_opinion_classifier import HybridOpinionClassifier
    import time
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = f"{base_output_dir}_{timestamp}"
    os.makedirs(base_output_dir, exist_ok=True)
    
    input_csv = f"{data_dir}/test.csv"
    if not os.path.exists(input_csv):
  
        input_csv = "../temp_subset.csv"
        if not os.path.exists(input_csv):
            print(f"[错误] 未找到测试数据: {input_csv}")
            return False
    
    print(f"\n 数据路径: {input_csv}")
    print(f" 输出目录: {base_output_dir}")

    df = pd.read_csv(input_csv)
    total_samples = len(df)
    print(f" 总样本数: {total_samples}")
    print("\n" + "=" * 80)
    print("实验配置概览:")
    print("  Group 1 (Baseline):  单任务模型 + 不启用风险路由")
    print("  Group 2 (Proposed):  多任务模型 + 启用风险路由 (核心方法)")
    print("  Group 3 (Oracle):    多任务模型 + Oracle DA 注入 (上界)")
    print("=" * 80)
    
    confirm = input(f"\n 将运行三组完整实验 ({total_samples} 样本 × 3 组)，确认? (yes/no): ").strip().lower()
    if confirm != 'yes':
        print("已取消")
        return False
    
    print("\n" + "=" * 80)
    print("第一组实验：Baseline (单任务判别 + 纯混合)")
    print("=" * 80)
    
    group1_dir = f"{base_output_dir}/group1_baseline"
    os.makedirs(group1_dir, exist_ok=True)
    
    single_task_model_dir = input("\n请输入单任务模型路径 (例如: ../discriminative_model_outputs_singletask): ").strip()
    if not os.path.exists(single_task_model_dir):
        print(f" 模型路径不存在: {single_task_model_dir}")
        print("跳过第一组实验")
        group1_success = False
    else:
        print(f"使用模型: {single_task_model_dir}")
        print("配置: 不启用风险路由")
        
        classifier_g1 = HybridOpinionClassifier(
            discriminative_model_dir=single_task_model_dir,
            use_generative=True,
            generative_model="deepseek-v3",
            cascade_threshold=0.96,
            cascade_threshold_irrelevant=0.80,
            prefer_generative=True,
            context_window=5,
            enable_risk_routing=False  
        )
        
        output_csv_g1 = f"{group1_dir}/predictions.csv"
        print(f"\n 开始运行 Group 1...")
        try:
            classifier_g1.classify_file(
                input_csv=input_csv,
                output_csv=output_csv_g1,
                use_oracle_da=False  
            )
            print("\n Group 1 完成，开始评估...")
            evaluate_performance(output_dir=group1_dir)
            group1_success = True
        except Exception as e:
            print(f"\n Group 1 失败: {e}")
            import traceback
            traceback.print_exc()
            group1_success = False
    
   
    print("\n" + "=" * 80)
    print(" 第二组实验：Proposed (多任务判别 + 风险路由) - 核心方法")
    print("=" * 80)
    
    group2_dir = f"{base_output_dir}/group2_proposed"
    os.makedirs(group2_dir, exist_ok=True)
    
    if model_dir is None:
        multi_task_model_dir = input("\n请输入多任务模型路径 (例如: ../discriminative_model_outputs_multitask): ").strip()
    else:
        multi_task_model_dir = model_dir
    
    if not os.path.exists(multi_task_model_dir):
        print(f"[错误] 模型路径不存在: {multi_task_model_dir}")
        print("跳过第二组实验")
        group2_success = False
    else:
        print(f"使用模型: {multi_task_model_dir}")
        print("配置: 启用风险路由 (利用预测的DA概率)")
        
        classifier_g2 = HybridOpinionClassifier(
            discriminative_model_dir=multi_task_model_dir,
            use_generative=True,
            generative_model="deepseek-v3",
            cascade_threshold=0.96,
            cascade_threshold_irrelevant=0.80,
            prefer_generative=True,
            context_window=5,
            enable_risk_routing=True  
        )
        
        output_csv_g2 = f"{group2_dir}/predictions.csv"
        print(f"\n 开始运行 Group 2...")
        try:
            classifier_g2.classify_file(
                input_csv=input_csv,
                output_csv=output_csv_g2,
                use_oracle_da=False  
            )
            print("\n Group 2 完成，开始评估...")
            evaluate_performance(output_dir=group2_dir)
            group2_success = True
        except Exception as e:
            print(f"\n Group 2 失败: {e}")
            import traceback
            traceback.print_exc()
            group2_success = False
    

    print("\n" + "=" * 80)
    print(" 第三组实验：Oracle (多任务判别 + Oracle DA 注入) - 性能上界")
    print("=" * 80)
    
    group3_dir = f"{base_output_dir}/group3_oracle"
    os.makedirs(group3_dir, exist_ok=True)
    

    df_check = pd.read_csv(input_csv)
    if 'Act Tag' not in df_check.columns:
        print(f"[警告] 数据中未找到 'Act Tag' 列，无法运行 Oracle 实验")
        print("跳过第三组实验")
        group3_success = False
    else:
        print(f"使用模型: {multi_task_model_dir if os.path.exists(multi_task_model_dir) else '需要重新指定'}")
        print("配置: 启用风险路由 + Oracle DA 注入")
        
        if not os.path.exists(multi_task_model_dir):
            multi_task_model_dir = input("\n请输入多任务模型路径: ").strip()
        
        if not os.path.exists(multi_task_model_dir):
            print(f"[错误] 模型路径不存在")
            group3_success = False
        else:
            classifier_g3 = HybridOpinionClassifier(
                discriminative_model_dir=multi_task_model_dir,
                use_generative=True,
                generative_model="deepseek-v3",
                cascade_threshold=0.96,
                cascade_threshold_irrelevant=0.80,
                prefer_generative=True,
                context_window=5,
                enable_risk_routing=True 
            )
            
            output_csv_g3 = f"{group3_dir}/predictions.csv"
            print(f"\n 开始运行 Group 3 (Oracle 模式)...")
            try:
                classifier_g3.classify_file(
                    input_csv=input_csv,
                    output_csv=output_csv_g3,
                    use_oracle_da=True,  
                    act_tag_column='Act Tag'
                )
                print("\n Group 3 完成，开始评估...")
                evaluate_performance(output_dir=group3_dir)
                group3_success = True
            except Exception as e:
                print(f"\n Group 3 失败: {e}")
                import traceback
                traceback.print_exc()
                group3_success = False
    
    print("\n" + "=" * 80)
    print(" 三组实验汇总报告")
    print("=" * 80)
    
    summary = []
    if group1_success:
        summary.append("Group 1 (Baseline):  完成")
    else:
        summary.append("Group 1 (Baseline):  失败或跳过")
    
    if group2_success:
        summary.append("Group 2 (Proposed): 完成")
    else:
        summary.append("Group 2 (Proposed): 失败或跳过")
    
    if group3_success:
        summary.append("Group 3 (Oracle):   完成")
    else:
        summary.append("Group 3 (Oracle):   失败或跳过")
    
    for line in summary:
        print(f"  {line}")
    
    print(f"\n所有结果保存在: {base_output_dir}/")
    print("\n提示：可以使用以下命令对比结果：")
    print(f"python code3/compare_experiments.py --dir {base_output_dir}")
    
    print("=" * 80)
    
    return True


def run_oracle_experiment(model_dir=None, data_dir='dataset_split_result_v4', output_dir='results_oracle_experiment'):
    print("=" * 80)
    print("Step 3 (Special): 运行 Oracle DA 实验 (上帝视角)")
    print("=" * 80)

    from hybrid_opinion_classifier import HybridOpinionClassifier

    if model_dir is None:
        model_dir = "../discriminative_model_outputs_v4_fix" 

    input_csv = f"{data_dir}/test.csv"
    os.makedirs(output_dir, exist_ok=True)
    output_csv = f"{output_dir}/hybrid_predictions_oracle.csv"

    print(f"模型路径: {model_dir}")
    print(f"数据路径: {input_csv}")
    print(f"输出路径: {output_csv}")

    classifier = HybridOpinionClassifier(
        discriminative_model_dir=model_dir,
        use_generative=True,
        generative_model="deepseek-v3",
        cascade_threshold=0.96,
        cascade_threshold_irrelevant=0.80,
        prefer_generative=True,
        context_window=5,
        enable_risk_routing=True  
    )

    print("\n 开始运行 Oracle 模式...")
    classifier.classify_file(
        input_csv=input_csv,
        output_csv=output_csv,
        use_oracle_da=True,  
        act_tag_column='Act Tag' 
    )

    print("\n自动进行评估...")
    evaluate_performance(output_dir=output_dir)




def main():
    parser = argparse.ArgumentParser(
        description='混合模型快速启动脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'test', 'batch', 'eval', 'all', 'oracle', 'groups'],
        default='test',
        help='运行模式 (默认: test)'
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default=None,
        help='判别模型目录 (例如: ../discriminative_model_outputs_fewshot_20pct)'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='dataset_split_result_v4',
        help='数据目录 (默认: dataset_split_result_v4)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='输出目录 (默认: ../results_*)'
    )
    parser.add_argument(
        '--enable_risk_routing',
        type=str,
        default='auto',
        choices=['auto', 'true', 'false'],
        help='是否启用风险路由: auto(自动检测模型类型)/true(强制启用)/false(强制禁用)'
    )

    args = parser.parse_args()

    print("\n")
    print("=" * 80)
    print("混合观点分类模型 - 快速启动")
    print("=" * 80)
    print(f"模式: {args.mode}")
    print("=" * 80)
    print("\n")

    success = True

    if args.mode == 'train' or args.mode == 'all':
        success = train_discriminative_model(data_dir=args.data_dir)
        if not success:
            print("\n训练失败，退出")
            return

    if args.mode == 'test' or args.mode == 'all':
        success = test_single_prediction()
        if not success:
            print("\n测试失败")

    if args.mode == 'batch' or args.mode == 'all':
        success = batch_process(
            model_dir=args.model_dir, 
            data_dir=args.data_dir, 
            output_dir=args.output_dir,
            enable_risk_routing=args.enable_risk_routing  # 【新增】传递参数
        )
        if not success:
            print("\n批量处理失败")

    if args.mode == 'eval' or args.mode == 'all':
        success = evaluate_performance(output_dir=args.output_dir)
        if not success:
            print("\n评估失败")

    if args.mode == 'oracle':
        out_dir = args.output_dir if args.output_dir else "results_oracle_exp"
        run_oracle_experiment(model_dir=args.model_dir, data_dir=args.data_dir, output_dir=out_dir)

    if args.mode == 'groups':
        # 运行三组对比实验
        out_dir = args.output_dir if args.output_dir else "results_groups"
        run_group_experiments(model_dir=args.model_dir, data_dir=args.data_dir, base_output_dir=out_dir)

    print("\n")
    print("=" * 80)
    print("完成!")
    print("=" * 80)


if __name__ == '__main__':
    main()
