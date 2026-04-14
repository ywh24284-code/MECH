# MECH: A Cost-Effective Multi-Task Cascade Framework for Classroom Opinion Evolution Recognition

This is the official implementation of the paper:

> **MECH: A Cost-Effective Multi-Task Cascade Framework for Classroom Opinion Evolution Recognition**  
> Yancui Li, Xiaoyu Zhou, Guoyi Miao, Fang Kong  
> *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026)*

## Overview

MECH is a hybrid cascade framework that combines a lightweight discriminative expert (DeBERTa-v3) with a generative reasoning expert (DeepSeek-v3) for classroom opinion evolution recognition. The key innovation is a **Semantic-Aware Risk Router** that uses Dialogue Act (DA) signals from multi-task learning to construct a "semantic safety net," routing implicit or ambiguous samples to the LLM while efficiently filtering simple ones.

### Key Results

| Method | Macro-F1 | Accuracy | API Cost |
|--------|----------|----------|----------|
| DeBERTa-v3 (Single-task) | 0.5810 | 0.7488 | 0% |
| GPT-4o (Zero-shot) | 0.5688 | 0.7156 | 100% |
| DeepSeek-v3 (Zero-shot) | 0.5963 | 0.6830 | 100% |
| **MECH (Ours)** | **0.6828** | **0.7855** | **55.6%** |

## Project Structure

```
MECH/
├── data/                          # COED dataset (train/val/test splits)
├── src/                           # Core MECH framework
│   ├── multi_task_model.py        # Multi-task model architecture (DeBERTa + DA Head + OE Head)
│   ├── train_multi_task_model.py  # Training script for discriminative expert
│   ├── hybrid_opinion_classifier.py  # Hybrid cascade classifier (core)
│   ├── run_hybrid_model.py        # CLI entry point for training / inference / evaluation
│   ├── error_analysis.py          # Error analysis utilities
│   └── utils.py                   # Shared utilities (EarlyStopping, etc.)
├── baseline_experiments/          # PLM baselines (BERT, RoBERTa, DeBERTa)
├── llm_baselines/                 # LLM fine-tuning baselines (Llama, Qwen via QLoRA)
├── prompting_baselines/           # Zero-shot / Few-shot prompting baselines
├── requirements.txt               # Python dependencies
└── README.md
```

## Quick Start

### 1. Installation

```bash
git clone https://github.com/ywh24284-code/MECH.git
cd MECH
pip install -r requirements.txt
```

For the hybrid inference pipeline, set up your LLM API key:

```bash
# Create a .env file in the project root
echo "OPENAI_API_KEY=your_api_key_here" > .env
echo "OPENAI_BASE_URL=https://api.deepseek.com/v1" >> .env
```

### 2. Train the Discriminative Multi-task Model

```bash
python src/train_multi_task_model.py \
  --task_type multi \
  --data_dir data \
  --output_dir outputs/multi_task_model
```

### 3. Run the MECH Hybrid Pipeline

```bash
python src/run_hybrid_model.py \
  --mode batch \
  --model_dir outputs/multi_task_model \
  --data_dir data \
  --output_dir outputs/results \
  --enable_risk_routing true \
  --process_mode all \
  --yes
```

### 4. Evaluate Results

```bash
python src/run_hybrid_model.py \
  --mode eval \
  --output_dir outputs/results
```

### 5. Run Ablation Experiments (Table 1 in the paper)

```bash
python src/run_hybrid_model.py \
  --mode groups \
  --model_dir outputs/multi_task_model \
  --single_task_model_dir outputs/single_task_model \
  --data_dir data \
  --output_dir outputs/ablation \
  --yes
```

## Baseline Reproduction

### PLM Baselines (Table 2)

```bash
# BERT
python baseline_experiments/train_plm_baseline.py \
  --model_type bert --model_path bert-base-uncased --data_dir data

# RoBERTa
python baseline_experiments/train_plm_baseline.py \
  --model_type roberta --model_path roberta-large --data_dir data

# DeBERTa
python baseline_experiments/train_plm_baseline.py \
  --model_type deberta --model_path microsoft/deberta-v3-base --data_dir data
```

### LLM Fine-tuning Baselines (Table 2)

```bash
# Llama-3.1-8B with QLoRA
python llm_baselines/train_llm_qlora_v2.py \
  --model_type llama --model_path meta-llama/Llama-3.1-8B --data_dir data

# Qwen2-7B with QLoRA
python llm_baselines/train_llm_qlora_v2.py \
  --model_type qwen --model_path Qwen/Qwen2-7B --data_dir data
```

### Prompting Baselines (Table 2)

```bash
# DeepSeek Zero-shot
python prompting_baselines/run_prompting_v2.py \
  --model_type deepseek --mode zero-shot --data_dir data

# GPT-4o Few-shot
python prompting_baselines/run_prompting_v2.py \
  --model_type gpt4o --mode few-shot --data_dir data
```

## Dataset (COED)

The Classroom Opinion Evolution Dataset (COED) is provided in `data/` with train/val/test splits. It contains 14,672 utterances from 100 K-12 mathematics classrooms, annotated with both Dialogue Act and Opinion Evolution labels.

**Opinion Evolution Labels (6 classes):** Irrelevant, New, Strengthened, Weakened, Adopted, Refuted

**Dialogue Act Labels (12 classes):** Teacher acts (None, Keeping Together, Relate, Restating, Revoicing, Press Accuracy, Press Reasoning) and Student acts (None, Relating to Another Student, Asking for Info, Making a Claim, Providing Evidence/Reasoning)

## Citation

```bibtex
@inproceedings{li2026mech,
  title={MECH: A Cost-Effective Multi-Task Cascade Framework for Classroom Opinion Evolution Recognition},
  author={Li, Yancui and Zhou, Xiaoyu and Miao, Guoyi and Kong, Fang},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2026}
}
```

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
