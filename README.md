MECH: A Cost-Effective Multi-Task Cascade Framework for Classroom Opinion Evolution Recognition
This is the official implementation of the paper: "MECH: A Cost-Effective Multi-Task Cascade Framework for Classroom Opinion Evolution Recognition", accepted by ACL 2026.

🌟 Overview
MECH is a hybrid cascade framework designed for classroom dialogue analysis. By integrating the Continuous Opinions and Discrete Actions (CODA) theory, it optimizes both recognition accuracy and API consumption.

Key Features:
Multi-task Cascade: Combines lightweight PLMs with LLMs using a smart routing mechanism.

Cost-Efficient: Achieves state-of-the-art performance while significantly reducing LLM API costs.

Theory-Grounded: Built upon the CODA theoretical framework for classroom opinion evolution.

📂 Project Structure
src/: Core implementation of the MECH framework (architecture, routing, and inference).

data/: The COED dataset (Classroom Opinion Evolution Dataset) used in our experiments.

llm_baselines/: Codes for QLoRA fine-tuning and diagnosis.

prompting_baselines/: Prompt engineering and Zero/Few-shot experiments.

baseline_experiments/: Traditional discriminative baselines (RoBERTa, DeBERTa).

🚀 Quick Start
Installation
Bash
git clone https://github.com/ywh24284-code/MECH.git
cd MECH
pip install -r requirements.txt
Training & Inference
Train the Multi-task Model (Internal Router):

Bash
python src/train_multi_task_model.py --task_type multi --model_type deberta
Run the MECH Hybrid Pipeline:

Bash
python src/run_hybrid_model.py --mode train
📊 Dataset (COED)
The dataset is provided in the data/ directory. It includes:

train.csv, val.csv, test.csv: Expert-annotated classroom dialogues with Dialogue Act (DA) and Opinion Evolution (OE) labels.

📝 Citation
If you find our work or code useful, please cite our paper:

代码段
@inproceedings{li2026mech,
  title={MECH: A Cost-Effective Multi-Task Cascade Framework for Classroom Opinion Evolution Recognition},
  author={Li, Yancui and Zhou,xiaoyu and Miao, Guoyi and Kong, Fang },
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2026}
}
