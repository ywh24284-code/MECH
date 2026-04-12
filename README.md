# Opinion Tracker (Open Source Release)

This package contains the code for the multi-task opinion tracking model and all baseline comparison experiments.

## Project Structure

- `src/`: Main model training, inference, architecture, and error analysis
- `baseline_experiments/`: PLM-based discriminative baselines
- `llm_baselines/`: QLoRA fine-tuning baselines for LLMs
- `prompting_baselines/`: Prompt-based baselines

## Included Core Files

- `src/train_multi_task_model.py`
- `src/run_hybrid_model.py`
- `src/multi_task_model.py`
- `src/hybrid_opinion_classifier.py`
- `src/error_analysis.py`

## Quick Start

1. Create environment and install dependencies:

```bash
pip install -r requirements.txt
```

2. Train the multi-task model:

```bash
python src/train_multi_task_model.py --task_type multi --model_type deberta
```

3. Run hybrid pipeline:

```bash
python src/run_hybrid_model.py --mode train
```

## Data

Dataset files are not included in this release. Place your dataset under your own data directory and pass paths through command-line arguments.

## License

This project is released under the license in `LICENSE`.
