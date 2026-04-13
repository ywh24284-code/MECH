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
1. Installation
Clone the repository and install the required dependencies:

Bash
git clone https://github.com/ywh24284-code/MECH.git
cd MECH
pip install -r requirements.txt
2. Training the Discriminative Multi-task Model
First, train the internal routing model (the lightweight PLM expert). This model will learn to predict both dialogue acts and opinion evolution to power the semantic-aware risk router.

Bash
python src/train_multi_task_model.py \
  --data_dir data \
  --output_dir multi_task_weighted_v2
3. Running the MECH Hybrid Pipeline
Once the discriminative model is trained, execute the hybrid cascade pipeline. This script will route high-confidence/low-risk samples to the local PLM and forward complex/high-risk samples to the LLM generative expert.

Bash
# Set the path to the model you just trained
MODEL_DIR="multi_task_weighted_v2" 

python src/run_hybrid_model.py \
  --mode batch \
  --model_dir "$MODEL_DIR" \
  --output_dir results_group2_0115 \
  --enable_risk_routing true
--mode batch: Runs evaluation on the entire dataset.

--model_dir: Specifies the directory of the trained multi-task PLM.

--output_dir: Directory where the hybrid predictions and evaluation metrics will be saved.

--enable_risk_routing true: Activates the core DA-based semantic risk routing mechanism proposed in our paper.

📊 Dataset (COED)
The dataset is provided in the data/ directory. It includes:

train.csv, val.csv, test.csv: Expert-annotated classroom dialogues with Dialogue Act (DA) and Opinion Evolution (OE) labels.

📝 Citation
If you find our work or code useful, please cite our paper:

@inproceedings{li2026mech,
  title={MECH: A Cost-Effective Multi-Task Cascade Framework for Classroom Opinion Evolution Recognition},
  author={Li, Yancui and Zhou, Xiaoyu and Miao, Guoyi and Kong, Fang},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2026}
}
