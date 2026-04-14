#!/usr/bin/env python3
"""
Hybrid model launcher script
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


def train_discriminative_model(data_dir='data'):
    """Train discriminative model"""
    print("=" * 80)
    print("Step 1: Train discriminative model")
    print("=" * 80)

    if not os.path.exists(data_dir):
        print(f"[Error] Data directory not found: {data_dir}")
        print("Please ensure data is prepared")
        return False

    import train_multi_task_model as discriminative_model_training
    try:
        discriminative_model_training.main()
        print("\n" + "=" * 80)
        print("✓ Discriminative model training complete!")
        print("=" * 80)
        return True
    except Exception as e:
        print(f"\n[Error] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_single_prediction():
    print("=" * 80)
    print("Step 2: Test hybrid model (single prediction)")
    print("=" * 80)

    from hybrid_opinion_classifier import HybridOpinionClassifier

    model_dir = "outputs/multi_task_model"
    model_file = os.path.join(model_dir, 'best_model.pth')
    config_file = os.path.join(model_dir, 'config.json')

    if not os.path.exists(config_file):
        print(f"[Error] Config file not found: {config_file}")
        print("Please run first: python3 run_hybrid_model.py --mode train")
        return False

    if not os.path.exists(model_file):
        print(f"[Error] Model file not found: {model_file}")
        print("Please run first: python3 run_hybrid_model.py --mode train")
        return False

    print("\nLoading model...")
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

    # Test examples
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
    print("Test examples")
    print("=" * 80)

    for i, example in enumerate(examples, 1):
        print(f"\n[Example {i}]")
        print(f"Input: [{example['speaker']}] {example['sentence']}")

        prev_speaker = example['context'][-1]['speaker'] if example['context'] else None

        result = classifier.classify_single(
            example['sentence'],
            example['speaker'],
            example['context'],
            prev_speaker
        )

        print(f"\nFinal prediction: {result['final_label']}")
        print(f"\nDiscriminative model:")
        print(f"  - Prediction: {result['discriminative_result']['predicted_label']}")
        print(f"  - Confidence: {result['discriminative_result']['confidence_score']:.4f}")
        print(f"  - Top-3 rankings:")
        for rank_info in result['discriminative_result']['top_k_rankings'][:3]:
            print(f"    {rank_info['rank']}. {rank_info['label']}: {rank_info['probability']:.4f}")

        if result['generative_result']:
            print(f"\nGenerative model:")
            print(f"  - Prediction: {result['generative_result']['predicted_label']}")

        print(f"\nDecision info:")
        print(f"  - Strategy: {result['decision_info']['strategy']}")
        print(f"  - Reason: {result['decision_info']['reason']}")
        print("-" * 80)

    return True


def batch_process(model_dir=None, data_dir='data', output_dir=None, enable_risk_routing='auto',
                   process_mode='all', class_id=None, num_classes=None,
                   hybrid_mode='standard', auto_confirm=False):
    print("=" * 80)
    print("Step 3: Batch processing CSV files")
    print("=" * 80)

    from hybrid_opinion_classifier import HybridOpinionClassifier

    if model_dir is None:
        model_dir = "outputs/multi_task_model"
    input_csv = f"{data_dir}/test.csv"

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_csv = f"{output_dir}/hybrid_predictions.csv"
    elif 'fewshot' in model_dir:
        output_csv = f"../hybrid_predictions_{os.path.basename(model_dir)}.csv"
    else:
        output_csv = "../hybrid_predictions.csv"

    if not os.path.exists(input_csv):
        print(f"[Error] Input file not found: {input_csv}")
        return False

    model_file = os.path.join(model_dir, 'best_model.pth')
    config_file = os.path.join(model_dir, 'config.json')

    if not os.path.exists(config_file):
        print(f"[Error] Config file not found: {config_file}")
        print("Please run first: python3 run_hybrid_model.py --mode train")
        return False

    if not os.path.exists(model_file):
        print(f"[Error] Model file not found: {model_file}")
        print("Please run first: python3 run_hybrid_model.py --mode train")
        return False

    df = pd.read_csv(input_csv)

    use_generative = True

    if 'class' in df.columns:
        classes = df['class'].unique()
        print(f"\nDataset contains {len(classes)} classes")
        print(f"Total samples: {len(df)}")

        class_counts = df['class'].value_counts().sort_index()
        print(f"\nSample distribution of first 10 classes:")
        for cls, count in class_counts.head(10).items():
            print(f"  Class {cls}: {count} samples")

        if process_mode == 'single':
            if class_id is None:
                print("[Error] --class_id is required when --process_mode is single")
                return False
            df_subset = df[df['class'] == class_id].copy()
            if len(df_subset) == 0:
                print(f"[Error] Class {class_id} does not exist")
                return False
            use_generative = True
            output_csv = f"../hybrid_predictions_class_{class_id}.csv"
            print(f"\nWill test class {class_id}, {len(df_subset)} samples total")

        elif process_mode == 'topn':
            if num_classes is None:
                print("[Error] --num_classes is required when --process_mode is topn")
                return False
            n = num_classes
            selected_classes = sorted(classes)[:n]
            df_subset = df[df['class'].isin(selected_classes)].copy()
            use_generative = True
            output_csv = f"../hybrid_predictions_top{n}_classes.csv"
            print(f"\nWill test first {n} classes, {len(df_subset)} samples total")
            print(f"Class IDs: {selected_classes}")

        elif process_mode == 'all':
            df_subset = df.copy()
            use_generative = True
            if not auto_confirm:
                confirm = input(f"\nWill process all {len(df)} samples, may incur high API costs. Confirm? (yes/no): ").strip().lower()
                if confirm != 'yes':
                    print("Cancelled")
                    return False
            else:
                print(f"\nWill process all {len(df)} samples (auto-confirmed)")

        elif process_mode == 'disc_only':
            df_subset = df.copy()
            use_generative = False
            print(f"\nWill use discriminative model only to process all {len(df)} samples (free)")

        else:
            print(f"[Error] Invalid process_mode: {process_mode}")
            return False
    else:
        df_subset = df.copy()
        use_generative = process_mode != 'disc_only'
        print(f"\nDataset has no class column, will process all {len(df)} samples")
        if use_generative:
            print("Generative model enabled")
        else:
            print("Discriminative model only")

    if use_generative:
        if hybrid_mode == 'full_llm':
            c_thresh = 2.0
            c_thresh_irr = 2.0
            print("\n>> Force full-LLM mode enabled: will call DeepSeek for all samples")
        else:
            c_thresh = 0.96
            c_thresh_irr = 0.80
            print(f"\n>> Standard hybrid mode enabled: high threshold strategy (Content>{c_thresh}, Irr>{c_thresh_irr})")
    else:
        c_thresh = 0.96
        c_thresh_irr = 0.80

    if model_dir and ('fewshot' in model_dir or '20pct' in model_dir or '40pct' in model_dir or
                      '60pct' in model_dir or '80pct' in model_dir):
        print("\n>> Few-shot model detected, applying strategy B (performance priority)...")


        if '20pct' in model_dir:
            base_threshold = 0.75
            irr_threshold = 0.70
            data_pct = 20
            print(f">> Few-shot mode (20%): high threshold strategy, most samples go to LLM (Content>{base_threshold}, Irr>{irr_threshold})")
        elif '40pct' in model_dir:
            base_threshold = 0.80
            irr_threshold = 0.75
            data_pct = 40
            print(f">> Few-shot mode (40%): high threshold strategy (Content>{base_threshold}, Irr>{irr_threshold})")
        elif '60pct' in model_dir:
            base_threshold = 0.85
            irr_threshold = 0.80
            data_pct = 60
            print(f">> Few-shot mode (60%): high threshold strategy (Content>{base_threshold}, Irr>{irr_threshold})")
        elif '80pct' in model_dir:
            base_threshold = 0.90
            irr_threshold = 0.85
            data_pct = 80
            print(f">> Few-shot mode (80%): high threshold strategy (Content>{base_threshold}, Irr>{irr_threshold})")
        else:
            # Generic fewshot identifier, use conservative strategy
            base_threshold = 0.75
            irr_threshold = 0.70
            data_pct = None
            print(f">> Few-shot mode (generic): high threshold strategy (Content>{base_threshold}, Irr>{irr_threshold})")

        c_thresh = base_threshold
        c_thresh_irr = irr_threshold

        if data_pct:
            print(f"   Expected API call rate: ~{100 - (data_pct * 0.3):.0f}% (performance priority strategy)")
    elif model_dir and 'singletask' not in model_dir:
        c_thresh = 0.95
        c_thresh_irr = 0.85
        print(f"\n>> Full model: standard high threshold strategy (Content>{c_thresh}, Irr>{c_thresh_irr})")
    print("\nLoading model...")


    if enable_risk_routing == 'auto':
        final_enable_risk_routing = True  # Enabled by default
        print(f">> Risk routing: auto mode (will decide based on model type)")
    elif enable_risk_routing == 'true':
        final_enable_risk_routing = True
        print(f">> Risk routing: force enabled")
    else:  # 'false'
        final_enable_risk_routing = False
        print(f">> Risk routing: force disabled")

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
        enable_risk_routing=final_enable_risk_routing
    )

    temp_csv = "../temp_subset.csv"
    df_subset.to_csv(temp_csv, index=False)

    print(f"\nProcessing...")
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
        print(f"\n[Error] Error during processing: {e}")
        import traceback
        traceback.print_exc()

    if os.path.exists(temp_csv):
        os.remove(temp_csv)

    print("\n" + "=" * 80)
    print(f"✓ Batch processing complete!")
    print(f"  Output file: {output_csv}")
    print(f"  Samples processed: {len(df_subset)}")
    print("=" * 80)
    return True


def evaluate_performance(output_dir=None):
    print("=" * 80)
    print("Step 4: Evaluate hybrid model performance")
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
        print(f"[Error] Result files not found")
        print(f"Search paths: {search_patterns}")
        print("Please run first: python run_hybrid_model.py --mode batch")
        return False

    result_csv = max(files, key=os.path.getmtime)

    print(f"\nAnalyzing file: {result_csv}")
    print(f"   File size: {os.path.getsize(result_csv) / 1024:.1f} KB")


    df = pd.read_csv(result_csv)
    print(f"Dataset size: {len(df)}")

    if 'true_label' not in df.columns or 'final_label' not in df.columns:
        print("[Error] Missing required columns: true_label or final_label")
        return False
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support,
        confusion_matrix, classification_report, matthews_corrcoef
    )
    from sklearn.preprocessing import LabelEncoder
    y_true = df['true_label']
    y_pred = df['final_label']

    print("\n" + "=" * 80)
    print("Overall Performance Metrics")
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
    print(f"  Accuracy:              {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"  MCC (Matthews Corr):   {mcc:.4f}")
    print(f"  Macro Precision:       {macro_precision:.4f}")
    print(f"  Macro Recall:          {macro_recall:.4f}")
    print(f"  Macro F1:              {macro_f1:.4f}")
    print("\n" + "=" * 80)
    print("Per-class Performance Metrics")
    print("=" * 80)
    labels = ['Irrelevant', 'New', 'Strengthened', 'Weakened', 'Adopted', 'Refuted']
    present_labels = [l for l in labels if l in y_true.values or l in y_pred.values]

    print(f"\n{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Samples':<8}")
    print("-" * 60)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=present_labels, zero_division=0
    )

    for i, label in enumerate(present_labels):
        print(f"{label:<15} {precision[i]:<10.4f} {recall[i]:<10.4f} {f1[i]:<10.4f} {support[i]:<8}")
    print("-" * 60)
    print(f"{'Macro Avg':<15} {precision.mean():<10.4f} {recall.mean():<10.4f} {f1.mean():<10.4f} {support.sum():<8}")

    print("\n" + "=" * 80)
    print("Confusion Matrix")
    print("=" * 80)

    cm = confusion_matrix(y_true, y_pred, labels=present_labels)

    header = "True\\Pred"
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
    print(f"\n✓ Confusion matrix (counts) saved: {cm_count_path}")
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
    print(f"✓ Confusion matrix (normalized) saved: {cm_norm_path}")
    plt.close()

    if 'strategy' in df.columns:
        print("\n" + "=" * 80)
        print("Decision Strategy Analysis")
        print("=" * 80)

        strategy_counts = df['strategy'].value_counts()

        print(f"\n{'Strategy':<30} {'Count':<10} {'Ratio':<10} {'Accuracy':<10}")
        print("-" * 60)

        for strategy in strategy_counts.index:
            strategy_df = df[df['strategy'] == strategy]
            count = len(strategy_df)
            pct = count / len(df) * 100
            acc = (strategy_df['true_label'] == strategy_df['final_label']).mean()
            print(f"{strategy:<30} {count:<10} {pct:<10.1f}% {acc:<10.4f}")

    if 'disc_confidence' in df.columns:
        print("\n" + "=" * 80)
        print("Discriminative Model Confidence Analysis")
        print("=" * 80)

        correct_mask = (df['true_label'] == df['final_label'])

        print(f"\n  Overall mean confidence: {df['disc_confidence'].mean():.4f}")
        print(f"  Median:                  {df['disc_confidence'].median():.4f}")
        print(f"  Std dev:                 {df['disc_confidence'].std():.4f}")
        print(f"  Min:                     {df['disc_confidence'].min():.4f}")
        print(f"  Max:                     {df['disc_confidence'].max():.4f}")

        print(f"\n  Correct prediction mean confidence: {df[correct_mask]['disc_confidence'].mean():.4f}")
        print(f"  Wrong prediction mean confidence:   {df[~correct_mask]['disc_confidence'].mean():.4f}")

        print(f"\n  Confidence interval distribution:")
        bins = [0, 0.5, 0.7, 0.8, 0.9, 1.0]
        for i in range(len(bins) - 1):
            mask = (df['disc_confidence'] >= bins[i]) & (df['disc_confidence'] < bins[i + 1])
            count = mask.sum()
            if count > 0:
                acc = (df[mask]['true_label'] == df[mask]['final_label']).mean()
                print(f"    [{bins[i]:.1f}, {bins[i + 1]:.1f}): {count:>4} samples, accuracy {acc:.4f}")

    if 'disc_label' in df.columns and 'gen_label' in df.columns:
        print("\n" + "=" * 80)
        print(" Discriminative model vs Generative model comparison")
        print("=" * 80)

        disc_acc = (df['disc_label'] == df['true_label']).mean()
        print(f"\n  Discriminative model accuracy: {disc_acc:.4f} ({disc_acc * 100:.2f}%)")

        gen_mask = df['gen_label'].notna()
        if gen_mask.sum() > 0:
            gen_acc = (df[gen_mask]['gen_label'] == df[gen_mask]['true_label']).mean()
            print(f"  Generative model accuracy:     {gen_acc:.4f} ({gen_acc * 100:.2f}%)")

        hybrid_acc = accuracy
        print(f"  Hybrid model accuracy:         {hybrid_acc:.4f} ({hybrid_acc * 100:.2f}%)")

        improvement = (hybrid_acc - disc_acc) * 100
        print(f"\n  Hybrid model improvement: {improvement:+.2f} percentage points")

    print("\n" + "=" * 80)
    print("✓ Evaluation complete!")
    print("=" * 80)
    return True


def run_group_experiments(model_dir=None, data_dir='data', base_output_dir='results_groups',
                          single_task_model_dir=None, auto_confirm=False):
    print("=" * 80)
    print("Three-group comparative experiments: Validating the auxiliary role of DA for OE")
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
            print(f"[Error] Test data not found: {input_csv}")
            return False

    print(f"\n Data path: {input_csv}")
    print(f" Output directory: {base_output_dir}")

    df = pd.read_csv(input_csv)
    total_samples = len(df)
    print(f" Total samples: {total_samples}")
    print("\n" + "=" * 80)
    print("Experiment Configuration Overview:")
    print("  Group 1 (Baseline):  Single-task model + risk routing disabled")
    print("  Group 2 (Proposed):  Multi-task model + risk routing enabled (core method)")
    print("  Group 3 (Oracle):    Multi-task model + Oracle DA injection (upper bound)")
    print("=" * 80)

    if not auto_confirm:
        confirm = input(f"\n Will run three full experiments ({total_samples} samples x 3 groups), confirm? (yes/no): ").strip().lower()
        if confirm != 'yes':
            print("Cancelled")
            return False
    else:
        print(f"\n Will run three full experiments ({total_samples} samples x 3 groups) (auto-confirmed)")

    print("\n" + "=" * 80)
    print("Group 1 Experiment: Baseline (single-task discriminative + pure hybrid)")
    print("=" * 80)

    group1_dir = f"{base_output_dir}/group1_baseline"
    os.makedirs(group1_dir, exist_ok=True)

    if single_task_model_dir is None:
        single_task_model_dir = input("\nEnter single-task model path (e.g.: ../discriminative_model_outputs_singletask): ").strip()
    if not os.path.exists(single_task_model_dir):
        print(f" Model path not found: {single_task_model_dir}")
        print("Skipping Group 1 experiment")
        group1_success = False
    else:
        print(f"Using model: {single_task_model_dir}")
        print("Config: risk routing disabled")

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
        print(f"\n Starting Group 1...")
        try:
            classifier_g1.classify_file(
                input_csv=input_csv,
                output_csv=output_csv_g1,
                use_oracle_da=False
            )
            print("\n Group 1 complete, starting evaluation...")
            evaluate_performance(output_dir=group1_dir)
            group1_success = True
        except Exception as e:
            print(f"\n Group 1 failed: {e}")
            import traceback
            traceback.print_exc()
            group1_success = False


    print("\n" + "=" * 80)
    print(" Group 2 Experiment: Proposed (multi-task discriminative + risk routing) - core method")
    print("=" * 80)

    group2_dir = f"{base_output_dir}/group2_proposed"
    os.makedirs(group2_dir, exist_ok=True)

    if model_dir is None:
        multi_task_model_dir = input("\nEnter multi-task model path (e.g.: ../discriminative_model_outputs_multitask): ").strip()
    else:
        multi_task_model_dir = model_dir

    if not os.path.exists(multi_task_model_dir):
        print(f"[Error] Model path not found: {multi_task_model_dir}")
        print("Skipping Group 2 experiment")
        group2_success = False
    else:
        print(f"Using model: {multi_task_model_dir}")
        print("Config: risk routing enabled (using predicted DA probabilities)")

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
        print(f"\n Starting Group 2...")
        try:
            classifier_g2.classify_file(
                input_csv=input_csv,
                output_csv=output_csv_g2,
                use_oracle_da=False
            )
            print("\n Group 2 complete, starting evaluation...")
            evaluate_performance(output_dir=group2_dir)
            group2_success = True
        except Exception as e:
            print(f"\n Group 2 failed: {e}")
            import traceback
            traceback.print_exc()
            group2_success = False


    print("\n" + "=" * 80)
    print(" Group 3 Experiment: Oracle (multi-task discriminative + Oracle DA injection) - upper bound")
    print("=" * 80)

    group3_dir = f"{base_output_dir}/group3_oracle"
    os.makedirs(group3_dir, exist_ok=True)


    df_check = pd.read_csv(input_csv)
    if 'Act Tag' not in df_check.columns:
        print(f"[Warning] 'Act Tag' column not found in data, cannot run Oracle experiment")
        print("Skipping Group 3 experiment")
        group3_success = False
    else:
        print(f"Using model: {multi_task_model_dir if os.path.exists(multi_task_model_dir) else 'needs to be specified'}")
        print("Config: risk routing enabled + Oracle DA injection")

        if not os.path.exists(multi_task_model_dir):
            multi_task_model_dir = input("\nEnter multi-task model path: ").strip()

        if not os.path.exists(multi_task_model_dir):
            print(f"[Error] Model path not found")
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
            print(f"\n Starting Group 3 (Oracle mode)...")
            try:
                classifier_g3.classify_file(
                    input_csv=input_csv,
                    output_csv=output_csv_g3,
                    use_oracle_da=True,
                    act_tag_column='Act Tag'
                )
                print("\n Group 3 complete, starting evaluation...")
                evaluate_performance(output_dir=group3_dir)
                group3_success = True
            except Exception as e:
                print(f"\n Group 3 failed: {e}")
                import traceback
                traceback.print_exc()
                group3_success = False

    print("\n" + "=" * 80)
    print(" Three-group experiment summary report")
    print("=" * 80)

    summary = []
    if group1_success:
        summary.append("Group 1 (Baseline):  Done")
    else:
        summary.append("Group 1 (Baseline):  Failed or skipped")

    if group2_success:
        summary.append("Group 2 (Proposed): Done")
    else:
        summary.append("Group 2 (Proposed): Failed or skipped")

    if group3_success:
        summary.append("Group 3 (Oracle):   Done")
    else:
        summary.append("Group 3 (Oracle):   Failed or skipped")

    for line in summary:
        print(f"  {line}")

    print(f"\nAll results saved in: {base_output_dir}/")
    print("\nTip: You can compare results using:")
    print(f"python code3/compare_experiments.py --dir {base_output_dir}")

    print("=" * 80)

    return True


def run_oracle_experiment(model_dir=None, data_dir='data', output_dir='results_oracle_experiment'):
    print("=" * 80)
    print("Step 3 (Special): Run Oracle DA experiment (oracle perspective)")
    print("=" * 80)

    from hybrid_opinion_classifier import HybridOpinionClassifier

    if model_dir is None:
        model_dir = "outputs/multi_task_model"

    input_csv = f"{data_dir}/test.csv"
    os.makedirs(output_dir, exist_ok=True)
    output_csv = f"{output_dir}/hybrid_predictions_oracle.csv"

    print(f"Model path: {model_dir}")
    print(f"Data path: {input_csv}")
    print(f"Output path: {output_csv}")

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

    print("\n Starting Oracle mode...")
    classifier.classify_file(
        input_csv=input_csv,
        output_csv=output_csv,
        use_oracle_da=True,
        act_tag_column='Act Tag'
    )

    print("\nRunning automatic evaluation...")
    evaluate_performance(output_dir=output_dir)




def main():
    parser = argparse.ArgumentParser(
        description='Hybrid model launcher script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=""
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'test', 'batch', 'eval', 'all', 'oracle', 'groups'],
        default='test',
        help='Run mode (default: test)'
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default=None,
        help='Discriminative model directory (e.g.: ../discriminative_model_outputs_fewshot_20pct)'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data',
        help='Data directory (default: data)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory (default: ../results_*)'
    )
    parser.add_argument(
        '--enable_risk_routing',
        type=str,
        default='auto',
        choices=['auto', 'true', 'false'],
        help='Enable risk routing: auto (auto-detect model type) / true (force enable) / false (force disable)'
    )
    parser.add_argument(
        '--process_mode',
        type=str,
        default='all',
        choices=['single', 'topn', 'all', 'disc_only'],
        help='Batch processing mode: single (single class) / topn (top-N classes) / all (all data) / disc_only (discriminative only)'
    )
    parser.add_argument(
        '--class_id',
        type=int,
        default=None,
        help='Class ID to test (used with --process_mode single)'
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        default=None,
        help='Number of classes to test (used with --process_mode topn)'
    )
    parser.add_argument(
        '--hybrid_mode',
        type=str,
        default='standard',
        choices=['standard', 'full_llm'],
        help='Hybrid run mode: standard (cascade routing) / full_llm (force LLM for all samples)'
    )
    parser.add_argument(
        '--single_task_model_dir',
        type=str,
        default=None,
        help='Single-task model directory for group experiments (Group 1 baseline)'
    )
    parser.add_argument(
        '-y', '--yes',
        action='store_true',
        default=False,
        help='Auto-confirm prompts (skip confirmation dialogs)'
    )

    args = parser.parse_args()

    print("\n")
    print("=" * 80)
    print("Hybrid Opinion Classifier - Launcher")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print("=" * 80)
    print("\n")

    success = True

    if args.mode == 'train' or args.mode == 'all':
        success = train_discriminative_model(data_dir=args.data_dir)
        if not success:
            print("\nTraining failed, exiting")
            return

    if args.mode == 'test' or args.mode == 'all':
        success = test_single_prediction()
        if not success:
            print("\nTest failed")

    if args.mode == 'batch' or args.mode == 'all':
        success = batch_process(
            model_dir=args.model_dir,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            enable_risk_routing=args.enable_risk_routing,
            process_mode=args.process_mode,
            class_id=args.class_id,
            num_classes=args.num_classes,
            hybrid_mode=args.hybrid_mode,
            auto_confirm=args.yes
        )
        if not success:
            print("\nBatch processing failed")

    if args.mode == 'eval' or args.mode == 'all':
        success = evaluate_performance(output_dir=args.output_dir)
        if not success:
            print("\nEvaluation failed")

    if args.mode == 'oracle':
        out_dir = args.output_dir if args.output_dir else "results_oracle_exp"
        run_oracle_experiment(model_dir=args.model_dir, data_dir=args.data_dir, output_dir=out_dir)

    if args.mode == 'groups':
        # Run three-group comparative experiments
        out_dir = args.output_dir if args.output_dir else "results_groups"
        run_group_experiments(
            model_dir=args.model_dir,
            data_dir=args.data_dir,
            base_output_dir=out_dir,
            single_task_model_dir=args.single_task_model_dir,
            auto_confirm=args.yes
        )

    print("\n")
    print("=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == '__main__':
    main()
