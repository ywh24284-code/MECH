# PLM Baseline Experiments

This folder contains discriminative baseline scripts for comparison.

## Files

- `train_plm_baseline.py`: Train baseline classifiers
- `compare_baselines.py`: Aggregate and compare baseline results

## Example

```bash
python baseline_experiments/train_plm_baseline.py \
    --model_type roberta \
    --model_path path/to/roberta-large \
    --output_dir baseline_roberta \
    --batch_size 8 \
    --num_epochs 12 \
    --learning_rate 1e-5
```
