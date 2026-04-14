#!/usr/bin/env python3
"""
Error Analysis Tool: Extract and analyze model classification errors.

Features:
1. Extract mispredicted samples from test/validation sets
2. Save detailed context information (speaker, sentence, 5-turn history)
3. Group by true opinion label for per-class error pattern analysis
4. Distinguish error types (DA-only wrong, OE-only wrong, both wrong)
"""

import pandas as pd
import numpy as np
import os
from typing import List, Dict, Tuple


# Label mappings (consistent with the 12-class DA taxonomy used in the multi-task model)
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
        context_sentences.insert(0, "[Dialogue Start]")

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
        f"Length mismatch: preds={len(predictions)}, labels={len(labels)}, df={len(data_df)}"

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
                'split': split_name,
                'sample_index': idx,
                'true_opinion_label': OPINION_LABELS[opinion_true],
                'true_opinion_id': opinion_true,
                'speaker': speaker,
                'sentence': sentence,
                'context_5_turns': context,
                'true_dialogue_act': DIALOGUE_ACT_LABELS[dialogue_act_true],
                'true_dialogue_act_id': dialogue_act_true,
                'pred_dialogue_act': DIALOGUE_ACT_LABELS[dialogue_act_pred],
                'pred_dialogue_act_id': dialogue_act_pred,
                'pred_opinion': OPINION_LABELS[opinion_pred],
                'pred_opinion_id': opinion_pred,
                'da_correct': 'Y' if dialogue_act_correct else 'N',
                'oe_correct': 'Y' if opinion_correct else 'N',
                'error_type': error_type
            }

            error_records.append(error_record)

    error_df = pd.DataFrame(error_records)

    if len(error_df) > 0:
        error_df = error_df.sort_values(
            by=['true_opinion_id', 'error_type', 'sample_index'],
            ascending=[True, True, True]
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    error_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    stats = {
        'total_samples': len(predictions),
        'total_errors': len(error_df),
        'error_rate': len(error_df) / len(predictions) if len(predictions) > 0 else 0,
        'both_wrong': len(error_df[error_df['error_type'] == 'both_wrong']),
        'dialogue_act_wrong_only': len(error_df[error_df['error_type'] == 'dialogue_act_wrong']),
        'opinion_wrong_only': len(error_df[error_df['error_type'] == 'opinion_wrong']),
        'dialogue_act_accuracy': sum([1 for d_pred, d_true in zip(dialogue_act_preds, dialogue_act_labels)
                                      if d_pred == d_true]) / len(predictions),
        'opinion_accuracy': sum([1 for o_pred, o_true in zip(predictions, labels)
                                if o_pred == o_true]) / len(predictions)
    }

    stats['errors_by_opinion_class'] = {}
    for opinion_id, opinion_name in OPINION_LABELS.items():
        class_errors = error_df[error_df['true_opinion_id'] == opinion_id]
        class_total = sum([1 for label in labels if label == opinion_id])
        stats['errors_by_opinion_class'][opinion_name] = {
            'total': class_total,
            'errors': len(class_errors),
            'error_rate': len(class_errors) / class_total if class_total > 0 else 0
        }

    return stats


def print_error_stats(stats: Dict):
    """Print error statistics."""
    print("\n" + "=" * 80)
    print("Error Analysis Statistics")
    print("=" * 80)
    print(f"Total samples: {stats['total_samples']}")
    print(f"Error samples: {stats['total_errors']} ({stats['error_rate']*100:.2f}%)")
    print(f"\nTask accuracy:")
    print(f"  Dialogue Act recognition: {stats['dialogue_act_accuracy']*100:.2f}%")
    print(f"  Opinion Evolution classification: {stats['opinion_accuracy']*100:.2f}%")
    print(f"\nError type distribution:")
    print(f"  Both tasks wrong: {stats['both_wrong']} ({stats['both_wrong']/stats['total_errors']*100:.2f}%)")
    print(f"  DA only wrong:    {stats['dialogue_act_wrong_only']} ({stats['dialogue_act_wrong_only']/stats['total_errors']*100:.2f}%)")
    print(f"  OE only wrong:    {stats['opinion_wrong_only']} ({stats['opinion_wrong_only']/stats['total_errors']*100:.2f}%)")

    print(f"\nPer-class error rate:")
    for opinion_name, class_stats in stats['errors_by_opinion_class'].items():
        print(f"  {opinion_name:15s}: {class_stats['errors']:4d}/{class_stats['total']:4d} "
              f"({class_stats['error_rate']*100:5.2f}%)")
    print("=" * 80)


if __name__ == "__main__":
    print("Error analysis tool loaded.")
    print("Call analyze_errors() from your training script.")
