import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
import os
import json
import sys
from typing import Dict, List, Tuple, Optional
import warnings
import re

warnings.filterwarnings('ignore')

try:
    from multi_task_model import MultiTaskDialogueModel

    MULTI_TASK_AVAILABLE = True
except ImportError:
    MULTI_TASK_AVAILABLE = False
    print("[警告] 未找到 multi_task_model.py，多任务模型不可用")

try:
    from openai import OpenAI
    from dotenv import load_dotenv
    import time


    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    if api_key:
        openai_client = OpenAI(
            api_key=api_key,
            base_url=base_url if base_url else None
        )
        GENERATIVE_MODEL_AVAILABLE = True
    else:
        print("[Warning] OPENAI_API_KEY 未设置，将仅使用判别模型")
        openai_client = None
        GENERATIVE_MODEL_AVAILABLE = False

except ImportError as e:
    print(f"[Warning] 无法导入生成模型依赖 ({e})，将仅使用判别模型")
    print("  请安装: pip install openai python-dotenv")
    GENERATIVE_MODEL_AVAILABLE = False
    openai_client = None

class DialogueAwareModel(nn.Module):


    def __init__(self, model_path, num_classes=6, dropout=0.5):
        super().__init__()
        # 使用 AutoModel 支持 DeBERTa/RoBERTa 等多种模型
        self.encoder = AutoModel.from_pretrained(model_path)
        self.hidden_size = self.encoder.config.hidden_size
        self.num_classes = num_classes

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask, return_confidence=False, return_embedding=False):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_token_embedding = outputs.last_hidden_state[:, 0, :]

        if return_embedding:
            return cls_token_embedding

        logits = self.classifier(cls_token_embedding)

        if return_confidence:
            probs = F.softmax(logits, dim=-1)
            confidence_scores, predicted_classes = torch.max(probs, dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, k=self.num_classes, dim=-1)

            return logits, {
                'probabilities': probs,
                'confidence_scores': confidence_scores,
                'predicted_classes': predicted_classes,
                'top_k_probs': top_k_probs,
                'top_k_indices': top_k_indices,
                'cls_embedding': cls_token_embedding  # 同时返回 embedding
            }

        return logits



class DiscriminativeInference:

    def __init__(self, model_dir: str, device=None):
        self.model_dir = model_dir
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

 
        config_path = os.path.join(model_dir, 'config.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

    
        model_name_or_path = self.config.get('model_name', 'roberta-base')

        print(f"  模型路径: {model_name_or_path}")

 
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)

        context_window = self.config.get('context_window', 5)
        special_tokens_list = [
            "[TEACHER]", "[CURRENT]", "[OTHER]", "Unknown"
        ]
        turn_tokens = [f"[TURN_{i}]" for i in range(context_window + 2)]
        special_tokens_list.extend(turn_tokens)
        num_added = self.tokenizer.add_tokens(special_tokens_list)
        print(f"  添加 special tokens: {num_added} 个")

        checkpoint_path = os.path.join(model_dir, 'best_model.pth')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"模型文件不存在: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

  
        state_dict_keys = checkpoint['model_state_dict'].keys()
        is_multi_task = 'dialogue_act_head.0.weight' in state_dict_keys or 'shared_layer.1.weight' in state_dict_keys

        if is_multi_task:
            if not MULTI_TASK_AVAILABLE:
                raise ImportError("检测到多任务模型，但 multi_task_model.py 不可用")
            print("  检测到: 多任务模型 (MultiTaskDialogueModel)")
            self.is_multi_task = True

            has_feature_fusion = any('feature_fusion' in k for k in state_dict_keys)

            if has_feature_fusion:
                print(" 检测到增强版本模型（带feature_fusion），但当前代码仅支持基线版本")
                print("  尝试兼容性加载（跳过不匹配的层）...")

            # 加载多任务模型
            self.model = MultiTaskDialogueModel(
                model_path=model_name_or_path,
                num_dialogue_acts=self.config.get('num_dialogue_acts', 12),
                num_opinion_classes=self.config.get('num_opinion_classes', 6),
                dropout=self.config.get('dropout', 0.6),
                use_act_logits_in_opinion=self.config.get('use_act_logits_in_opinion', False)
            )
            # 调整 embedding
            self.model.encoder.resize_token_embeddings(len(self.tokenizer))
            print(f"  模型 embedding 大小: {self.model.encoder.embeddings.word_embeddings.weight.shape[0]}")

    
            if has_feature_fusion:
                print(f"  使用兼容模式加载权重（跳过额外的层）...")
                model_state = self.model.state_dict()
                checkpoint_state = checkpoint['model_state_dict']

                filtered_state = {}
                skipped_keys = []
                extra_keys = []

                for key in model_state.keys():
                    if key in checkpoint_state:
                        if model_state[key].shape == checkpoint_state[key].shape:
                            filtered_state[key] = checkpoint_state[key]
                        else:
                            skipped_keys.append(f"{key} (shape mismatch)")
                    else:
                        skipped_keys.append(f"{key} (not in checkpoint)")

                for key in checkpoint_state.keys():
                    if key not in model_state:
                        extra_keys.append(key)

                self.model.load_state_dict(filtered_state, strict=False)

                print(f"兼容性加载完成，加载了 {len(filtered_state)}/{len(model_state)} 个参数")
                if extra_keys:
                    print(f"Checkpoint中有 {len(extra_keys)} 个额外的参数（已忽略）:")
                    for key in extra_keys[:5]:
                        print(f"    - {key}")
                    if len(extra_keys) > 5:
                        print(f"    ... 还有 {len(extra_keys) - 5} 个")
                if skipped_keys:
                    print(f"  跳过了 {len(skipped_keys)} 个参数（将使用随机初始化）")
                    for key in skipped_keys[:5]:
                        print(f"    - {key}")
                    if len(skipped_keys) > 5:
                        print(f"    ... 还有 {len(skipped_keys) - 5} 个")
            else:
              
                try:
                    self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
                    print(f" 模型权重加载成功")
                except RuntimeError as e:
                    if 'size mismatch' in str(e) or 'Unexpected key' in str(e) or 'Missing key' in str(e):
                        print(f" 严格加载失败，使用兼容模式...")
               
                        model_state = self.model.state_dict()
                        checkpoint_state = checkpoint['model_state_dict']

                        filtered_state = {}
                        skipped_keys = []
                        for key in model_state.keys():
                            if key in checkpoint_state:
                                if model_state[key].shape == checkpoint_state[key].shape:
                                    filtered_state[key] = checkpoint_state[key]
                                else:
                                    skipped_keys.append(f"{key} (shape mismatch)")
                            else:
                                skipped_keys.append(f"{key} (not in checkpoint)")

                        self.model.load_state_dict(filtered_state, strict=False)

                        print(f"兼容性加载完成，加载了 {len(filtered_state)}/{len(model_state)} 个参数")
                        if skipped_keys:
                            print(f"跳过了 {len(skipped_keys)} 个参数（将使用随机初始化）")
                            for key in skipped_keys[:5]:
                                print(f"    - {key}")
                            if len(skipped_keys) > 5:
                                print(f"    ... 还有 {len(skipped_keys) - 5} 个")
                    else:
                        raise
            self.dialogue_act_names = [
                "None (Teacher)",
                "Keeping Together",
                "Relate",
                "Restating",
                "Revoicing",
                "Press Accuracy",
                "Press Reasoning",
                "None (Student)",
                "Relating to Another Student",
                "Asking for Info",
                "Making a Claim",
                "Providing Evidence/Reasoning"
            ]
        else:
            print("检测到: 单任务模型 (DialogueAwareModel)")
            self.is_multi_task = False
            self.model = DialogueAwareModel(
                model_path=model_name_or_path,
                num_classes=self.config.get('num_classes', 6),
                dropout=self.config.get('dropout', 0.6)
            )
            self.model.encoder.resize_token_embeddings(len(self.tokenizer))
            print(f"  模型 embedding 大小: {self.model.encoder.embeddings.word_embeddings.weight.shape[0]}")
            self.model.load_state_dict(checkpoint['model_state_dict'])

        self.model.to(self.device)
        self.model.eval()

        self.label_names = ['Irrelevant', 'New', 'Strengthened', 'Weakened', 'Adopted', 'Refuted']

        print(f"判别模型加载成功")
        print(f"设备: {self.device}")
        val_f1 = checkpoint.get('val_macro_f1', 0)
        if isinstance(val_f1, (int, float)):
            print(f"验证集Macro-F1: {val_f1:.4f}")
        else:
            print(f"验证集Macro-F1: {val_f1}")

    def _get_role_name(self, speaker_name, current_speaker_name):
        teacher_names = {'T', 'Teacher', 'Ms. G', 'Mrs. G'}
        s_name = str(speaker_name).strip()

        if s_name.startswith('T') or s_name in teacher_names:
            return "[TEACHER]"
        if s_name == str(current_speaker_name).strip():
            return "[CURRENT]"
        return "[OTHER]"

    def predict_single(self, text: str, speaker: str = None, context_history: List[Dict] = None) -> Dict:
        use_context = self.config.get('use_context', True)
        context_window = self.config.get('context_window', 5)
        use_turn_indicators = self.config.get('use_turn_indicators', True)

     
        target_text = f"[CURRENT] {text}"


        context_text = ""
        if use_context and context_history:
            context_parts = []

            recent_context = context_history[-context_window:] if len(
                context_history) > context_window else context_history

            for turn_idx, ctx in enumerate(recent_context):
                ctx_speaker = ctx.get('speaker', 'Unknown')
                ctx_sentence = ctx.get('sentence', '')

              
                role = self._get_role_name(ctx_speaker, speaker)

                if use_turn_indicators:
                    turn_marker = f"[TURN_{turn_idx}]"
                    context_parts.append(f"{turn_marker} {role} {ctx_sentence}")
                else:
                    context_parts.append(f"{role} {ctx_sentence}")

            context_text = " ".join(context_parts)

  
        if context_text:
            encoding = self.tokenizer(
                text=context_text,
                text_pair=target_text,
                max_length=self.config.get('max_length', 256),
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        else:
    
            encoding = self.tokenizer(
                text=target_text,
                max_length=self.config.get('max_length', 256),
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            if self.is_multi_task:
                outputs = self.model(input_ids, attention_mask, return_confidence=True, return_embedding=True)

                opinion_info = outputs['opinion_confidence']
                cls_embedding = outputs.get('cls_embedding', None)
                if cls_embedding is not None:
                    cls_embedding = cls_embedding[0].cpu().numpy()

                probs = opinion_info['probabilities'][0].cpu().numpy()
                predicted_class = opinion_info['predicted_classes'][0].item()
                confidence_score = opinion_info['confidence_scores'][0].item()
                top_k_probs = opinion_info['top_k_probs'][0].cpu().numpy()
                top_k_indices = opinion_info['top_k_indices'][0].cpu().numpy()

                dialogue_act_info = outputs['dialogue_act_confidence']
                dialogue_act_pred = dialogue_act_info['predicted_classes'][0].item()
                dialogue_act_conf = dialogue_act_info['confidence_scores'][0].item()
                dialogue_act_probs = dialogue_act_info['probabilities'][0].cpu().numpy()
            else:
                logits, confidence_info = self.model(input_ids, attention_mask, return_confidence=True)

                cls_embedding = confidence_info.get('cls_embedding', None)
                if cls_embedding is not None:
                    cls_embedding = cls_embedding.cpu().numpy()

                probs = confidence_info['probabilities'][0].cpu().numpy()
                predicted_class = confidence_info['predicted_classes'][0].item()
                confidence_score = confidence_info['confidence_scores'][0].item()
                top_k_probs = confidence_info['top_k_probs'][0].cpu().numpy()
                top_k_indices = confidence_info['top_k_indices'][0].cpu().numpy()

                dialogue_act_pred = None
                dialogue_act_conf = None
                dialogue_act_probs = None

        result = {
            'predicted_label': self.label_names[predicted_class],
            'predicted_class_id': predicted_class,
            'confidence_score': float(confidence_score),
            'all_probabilities': {
                self.label_names[i]: float(probs[i]) for i in range(len(self.label_names))
            },
            'top_k_rankings': [
                {
                    'rank': i + 1,
                    'label': self.label_names[int(idx)],
                    'class_id': int(idx),
                    'probability': float(prob)
                }
                for i, (idx, prob) in enumerate(zip(top_k_indices, top_k_probs))
            ]
        }

        if cls_embedding is not None:
            result['cls_embedding'] = cls_embedding

        if self.is_multi_task:
            result['dialogue_act'] = {
                'predicted_label': self.dialogue_act_names[dialogue_act_pred],
                'predicted_class_id': dialogue_act_pred,
                'confidence_score': float(dialogue_act_conf),
                'all_probabilities': {
                    self.dialogue_act_names[i]: float(dialogue_act_probs[i])
                    for i in range(len(self.dialogue_act_names))
                }
            }

        return result



VALID_LABELS = {
    "Irrelevant", "New", "Strengthened",
    "Weakened", "Adopted", "Refuted"
}


def call_llm_api(messages: List[Dict[str, str]], model: str = "deepseek-v3",
                 max_retries: int = 2) -> str:
   
    if not GENERATIVE_MODEL_AVAILABLE or openai_client is None:
        return "Error:API_Not_Available"

    for attempt in range(max_retries):
        try:
            response = openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=50
            )
            raw_label = response.choices[0].message.content.strip()

            label = extract_label_from_response(raw_label)

            if label in VALID_LABELS:
                return label
            else:
                print(f"    [Warning] LLM返回了无效标签: '{raw_label[:50]}...'. 默认记为 'Irrelevant'.")
                return "Irrelevant"

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    [Warning] API调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                time.sleep(2)
            else:
                print(f"    [Error] API调用最终失败: {e}")
                return "Error:API_Failure"

    return "Error:API_Failure"


def extract_label_from_response(raw_text: str) -> str:
    raw_text = raw_text.strip()

    if raw_text in VALID_LABELS:
        return raw_text

    label_names = ['Irrelevant', 'New', 'Strengthened', 'Weakened', 'Adopted', 'Refuted']

    lines = raw_text.split('\n')
    last_line = lines[-1].strip() if lines else ""

    for label in label_names:
        if last_line.lower() == label.lower():
            return label

    for label in label_names:
        if label.lower() in last_line.lower():
            return label

    for label in label_names:
        if label.lower() in raw_text.lower():
            return label

    return "Irrelevant" 


def get_classification_prompt_v1(
    context_str: str,
    current_speaker: str,
    current_sentence: str,
    consistency: str,
    act_hint: str = None,
    kb_hint: str = None,
) -> List[Dict[str, str]]:
    """旧版 Prompt：简洁标签说明 + 可选 DA 提示，不使用知识库强先验。"""
    system_prompt = f"""
    你是一个课堂对话分析专家。你的任务是根据"上下文"（前面的发言）和"当前发言"之间的关系，对"当前发言"进行分类。
    你必须从以下六个标签中选择一个：

    1.  **Irrelevant (无关)**: 发言与讨论的主题无关，或者是关于对话状态的元评论（如 "我听不懂了"），或者是程序性发言（如老师管理课堂秩序）。
    2.  **New (新观点)**: 引入了一个在上下文中未曾出现过的新主张、新论点或新证据。
    3.  **Strengthened (强化)**: 为上下文中已有的某个观点提供支持、同意、证据或更详细的阐述。
    4.  **Weakened (削弱)**: 针对上下文中的观点提出疑问、反例、不同意见或指出其局限性，但没有完全否定它。
    5.  **Adopted (采纳)**: 发言人明确表示同意或接受了上下文中 *另一位* 发言人的观点（通常在 'switch' 状态下发生）。
    6.  **Refuted (驳斥)**: 发言人明确、直接地否定了上下文中的观点（通常在 'switch' 状态下发生，例如以 "No..." 或 "Yeah, but..." 开头）。

    你的回复必须 *仅仅* 包含这六个标签中的一个词，不要有任何其他解释。
    """

    user_prompt = f"""
    **上下文 (Context):**
    {context_str if context_str else "（没有上下文，这是第一句发言）"}

    **待分类的发言 (Current Utterance):**
    - **发言人 (Speaker)**: {current_speaker} (注意：'T' 代表老师)
    - **发言人一致性 (Consistency)**: {consistency}
    - **句子 (Sentence)**: {current_sentence}
    """

    # 注入对话行为提示（软提示）
    if act_hint:
        user_prompt += (
            f"\n    - **已知对话行为 (Dialogue Act)**: {act_hint} "
            "(提示：请参考此意图进行判断，例如“提供证据”可能意味着强化或反驳)。\n"
        )

    # 旧版 prompt 不向 LLM 注入知识库规则，这里忽略 kb_hint

    user_prompt += "\n    请分类："

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt.strip()}
    ]


def get_classification_prompt(
    context_str: str,
    current_speaker: str,
    current_sentence: str,
    consistency: str,
    act_hint: str = None, 
    kb_hint: str = None

) -> List[Dict[str, str]]:
    """旧版 Prompt：简洁标签说明 + 可选 DA 提示，不使用知识库强先验。"""
    system_prompt = f"""
    你是一个课堂对话分析专家。你的任务是根据"上下文"（前面的发言）和"当前发言"之间的关系，对"当前发言"进行分类。
    你必须从以下六个标签中选择一个：

    1.  **Irrelevant (无关)**: 发言与讨论的主题无关，或者是关于对话状态的元评论（如 "我听不懂了"），或者是程序性发言（如老师管理课堂秩序）。
    2.  **New (新观点)**: 引入了一个在上下文中未曾出现过的新主张、新论点或新证据。
    3.  **Strengthened (强化)**: 为上下文中已有的某个观点提供支持、同意、证据或更详细的阐述。
    4.  **Weakened (削弱)**: 针对上下文中的观点提出疑问、反例、不同意见或指出其局限性，但没有完全否定它。
    5.  **Adopted (采纳)**: 发言人明确表示同意或接受了上下文中 *另一位* 发言人的观点（通常在 'switch' 状态下发生）。
    6.  **Refuted (驳斥)**: 发言人明确、直接地否定了上下文中的观点（通常在 'switch' 状态下发生，例如以 "No..." 或 "Yeah, but..." 开头）。

    你的回复必须 *仅仅* 包含这六个标签中的一个词，不要有任何其他解释。
    """

    user_prompt = f"""
    **上下文 (Context):**
    {context_str if context_str else "（没有上下文，这是第一句发言）"}

    **待分类的发言 (Current Utterance):**
    - **发言人 (Speaker)**: {current_speaker} (注意：'T' 代表老师)
    - **发言人一致性 (Consistency)**: {consistency}
    - **句子 (Sentence)**: {current_sentence}
    """

    # 注入对话行为提示（软提示）
    if act_hint:
        user_prompt += (
            f"\n    - **已知对话行为 (Dialogue Act)**: {act_hint} "
            "(提示：请参考此意图进行判断，例如“提供证据”可能意味着强化或反驳)。\n"
        )

    # 旧版 prompt 不向 LLM 注入知识库规则，这里忽略 kb_hint

    user_prompt += "\n    请分类："

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt.strip()}
    ]


def get_classification_prompt_with_icl_v1(
    context_str: str,
    current_speaker: str,
    current_sentence: str,
    consistency: str,
    icl_examples: List[Dict],
    act_hint: str = None,
    kb_hint: str = None,
 ) -> List[Dict[str, str]]:
    """旧版 ICL Prompt：示例 + 可选 DA 提示，不使用知识库强先验。"""
    system_prompt = f"""
    你是一个课堂对话分析专家。你的任务是根据"上下文"（前面的发言）和"当前发言"之间的关系，对"当前发言"进行分类。
    你必须从以下六个标签中选择一个：

    1.  **Irrelevant (无关)**: 发言与讨论的主题无关，或者是关于对话状态的元评论（如 "我听不懂了"），或者是程序性发言（如老师管理课堂秩序）。
    2.  **New (新观点)**: 引入了一个在上下文中未曾出现过的新主张、新论点或新证据。
    3.  **Strengthened (强化)**: 为上下文中已有的某个观点提供支持、同意、证据或更详细的阐述。
    4.  **Weakened (削弱)**: 针对上下文中的观点提出疑问、反例、不同意见或指出其局限性，但没有完全否定它。
    5.  **Adopted (采纳)**: 发言人明确表示同意或接受了上下文中 *另一位* 发言人的观点（通常在 'switch' 状态下发生）。
    6.  **Refuted (驳斥)**: 发言人明确、直接地否定了上下文中的观点（通常在 'switch' 状态下发生，例如以 "No..." 或 "Yeah, but..." 开头）。

    你的回复必须 *仅仅* 包含这六个标签中的一个词，不要有任何其他解释。

    以下是一些相似句子的分类示例，供你参考：
    """

    icl_examples_str = "\n"
    for i, example in enumerate(icl_examples, 1):
        sentence = example.get('sentence', '')
        label_name = example.get('label_name', 'Unknown')
        similarity = example.get('similarity', 0.0)
        icl_examples_str += f"\n**示例 {i}** (相似度: {similarity:.2f}):\n"
        icl_examples_str += f"  句子: {sentence}\n"
        icl_examples_str += f"  分类: {label_name}\n"

    system_prompt += icl_examples_str

    user_prompt = f"""
    **上下文 (Context):**
    {context_str if context_str else "（没有上下文，这是第一句话发言）"}

    **待分类的发言 (Current Utterance):**
    - **发言人 (Speaker)**: {current_speaker} (注意：'T' 代表老师)
    - **发言人一致性 (Consistency)**: {consistency}
    - **句子 (Sentence)**: {current_sentence}

    """


    if act_hint:
        user_prompt += (
            f"\n    - **已知对话行为 (Dialogue Act)**: {act_hint} "
            "(提示：请参考此意图进行判断)。\n"
        )

    # 旧版 ICL prompt 不向 LLM 注入知识库规则，这里忽略 kb_hint

    user_prompt += "\n    请参考上述示例，先简要分析（1-2句话），然后在最后一行输出标签（仅一个词）。"

    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt.strip()}
    ]


def get_classification_prompt_with_icl(
    context_str: str,
    current_speaker: str,
    current_sentence: str,
    consistency: str,
    icl_examples: List[Dict],
    act_hint: str = None, 
    kb_hint: str = None
) -> List[Dict[str, str]]:
    """【新增】构建带 kNN-ICL 的分类 Prompt (Legacy Version)"""
    system_prompt = f"""
    你是一个课堂对话分析专家。你的任务是根据"上下文"（前面的发言）和"当前发言"之间的关系，对"当前发言"进行分类。
    你必须从以下六个标签中选择一个：

    1.  **Irrelevant (无关)**: 发言与讨论的主题无关，或者是关于对话状态的元评论（如 "我听不懂了"），或者是程序性发言（如老师管理课堂秩序）。
    2.  **New (新观点)**: 引入了一个在上下文中未曾出现过的新主张、新论点或新证据。
    3.  **Strengthened (强化)**: 为上下文中已有的某个观点提供支持、同意、证据或更详细的阐述。
    4.  **Weakened (削弱)**: 针对上下文中的观点提出疑问、反例、不同意见或指出其局限性，但没有完全否定它。
    5.  **Adopted (采纳)**: 发言人明确表示同意或接受了上下文中 *另一位* 发言人的观点（通常在 'switch' 状态下发生）。
    6.  **Refuted (驳斥)**: 发言人明确、直接地否定了上下文中的观点（通常在 'switch' 状态下发生，例如以 "No..." 或 "Yeah, but..." 开头）。

    你的回复必须 *仅仅* 包含这六个标签中的一个词，不要有任何其他解释。

    以下是一些相似句子的分类示例，供你参考：
    """

    # 构建 ICL 示例部分
    icl_examples_str = "\n"
    for i, example in enumerate(icl_examples, 1):
        sentence = example.get('sentence', '')
        label_name = example.get('label_name', 'Unknown')
        similarity = example.get('similarity', 0.0)
        icl_examples_str += f"\n**示例 {i}** (相似度: {similarity:.2f}):\n"
        icl_examples_str += f"  句子: {sentence}\n"
        icl_examples_str += f"  分类: {label_name}\n"

    system_prompt += icl_examples_str

    user_prompt = f"""
    **上下文 (Context):**
    {context_str if context_str else "（没有上下文，这是第一句发言）"}

    **待分类的发言 (Current Utterance):**
    - **发言人 (Speaker)**: {current_speaker} (注意：'T' 代表老师)
    - **发言人一致性 (Consistency)**: {consistency}
    - **句子 (Sentence)**: {current_sentence}

    """

    if act_hint:
        user_prompt += f"\n    - **已知对话行为 (Dialogue Act)**: {act_hint} (提示：请参考此意图进行判断)\n"


    user_prompt += "\n    请参考上述示例，先简要分析（1-2句话），然后在最后一行输出标签（仅一个词）。"

    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt.strip()}
    ]


class GenerativeInference:

    def __init__(self, model_name: str = "deepseek-v3", context_window: int = 5, use_knn_icl: bool = False):
        if not GENERATIVE_MODEL_AVAILABLE:
            raise ImportError("生成模型未可用，请检查OPENAI_API_KEY和依赖")

        self.model_name = model_name
        self.context_window = context_window
        self.use_knn_icl = use_knn_icl 
        print(f"✓ 生成模型初始化完成")
        print(f"  模型: {model_name}")
        print(f"  上下文窗口: {context_window}")
        print(f"  kNN-ICL: {'启用' if use_knn_icl else '禁用'}")

    def predict_single(self, sentence: str, speaker: str, context_history: List[Dict],
                       previous_speaker: Optional[str] = None,
                       icl_examples: Optional[List[Dict]] = None,
                       oracle_da_label: str = None,  
                       kb_hint: Optional[str] = None,
                       prompt_version: str = "v1"  
                       ) -> Dict:
 
    
        if previous_speaker is None:
            consistency = "switch"  # 第一句话
        elif speaker == previous_speaker:
            consistency = "same"
        else:
            consistency = "switch"

   
        recent_context = context_history[-self.context_window:] if context_history else []
        context_lines = []
        for ctx in recent_context:
            ctx_speaker = ctx.get('speaker', '')
            ctx_sentence = ctx.get('sentence', '')
            if ctx_sentence:
                context_lines.append(f"{ctx_speaker}: {ctx_sentence}")
        context_str = "\n".join(context_lines)

        if self.use_knn_icl and icl_examples:
            if prompt_version == "v2":
                messages = get_classification_prompt_with_icl(
                    context_str, speaker, sentence, consistency, icl_examples,
                    act_hint=oracle_da_label,
                    kb_hint=kb_hint,
                )
            else:
                messages = get_classification_prompt_with_icl_v1(
                    context_str, speaker, sentence, consistency, icl_examples,
                    act_hint=oracle_da_label,
                    kb_hint=kb_hint,
                )
        else:
            if prompt_version == "v2":
                messages = get_classification_prompt(
                    context_str, speaker, sentence, consistency,
                    act_hint=oracle_da_label,
                    kb_hint=kb_hint,
                )
            else:
                messages = get_classification_prompt_v1(
                    context_str, speaker, sentence, consistency,
                    act_hint=oracle_da_label,
                    kb_hint=kb_hint,
                )

        predicted_label = call_llm_api(messages, model=self.model_name)

        result = {
            'predicted_label': predicted_label,
            'reasoning': f'LLM推理 (consistency={consistency})' if predicted_label not in ['Error:API_Failure',
                                                                                           'Error:API_Not_Available'] else 'API调用失败',
            'consistency': consistency
        }

        # 【新增】如果使用了 kNN-ICL，记录检索到的例子
        if self.use_knn_icl and icl_examples:
            result['icl_examples_used'] = len(icl_examples)
            result['icl_labels'] = [ex.get('label_name', 'Unknown') for ex in icl_examples]

        return result

class HybridDecisionMaker:
 

    def __init__(
            self,
            discriminative_threshold: float = 0.8, 
            agreement_weight: float = 0.5,
            topk_threshold: float = 0.2,  
            prefer_generative: bool = False 
    ):

        self.disc_threshold = discriminative_threshold
        self.agreement_weight = agreement_weight
        self.topk_threshold = topk_threshold
        self.prefer_generative = prefer_generative

    def make_decision(
            self,
            disc_result: Dict,
            gen_result: Dict,
            return_details: bool = True,
            risk_score: Optional[float] = None
    ) -> Tuple[str, Dict]:

        disc_label = disc_result['predicted_label']
        disc_confidence = disc_result['confidence_score']

        gen_label = gen_result['predicted_label'] if gen_result else None

        decision_info = {
            'discriminative_label': disc_label,
            'discriminative_confidence': disc_confidence,
            'generative_label': gen_label,
        }


        if gen_label is not None and disc_label == 'Irrelevant' and gen_label != 'Irrelevant':
            guard_threshold = 0.5
            if risk_score is not None and risk_score > 0.5:
                guard_threshold = 0.8

            if disc_confidence >= guard_threshold:
                final_decision = disc_label
                decision_info['strategy'] = 'guard_irrelevant_from_llm'
                extra_risk_info = f", risk={risk_score:.4f}" if risk_score is not None else ""
                decision_info['reason'] = (
                    f'Disc 预测 Irrelevant 且置信度较高 ({disc_confidence:.4f} >= {guard_threshold:.2f})，'
                    f'保守不改为 {gen_label}{extra_risk_info}'
                )
                decision_info['final_decision'] = final_decision
                if return_details:
                    return final_decision, decision_info
                else:
                    return final_decision
        if gen_label == 'Irrelevant' and disc_label != 'Irrelevant':
            if disc_confidence >= 0.8:
                final_decision = disc_label
                decision_info['strategy'] = 'guard_content_from_llm_irrelevant'
                decision_info['reason'] = (
                    f'Disc 预测观点类 {disc_label} 且置信度很高 ({disc_confidence:.4f})，'
                    f'仅在此情况下保守不压成 Irrelevant'
                )
                decision_info['final_decision'] = final_decision
                if return_details:
                    return final_decision, decision_info
                else:
                    return final_decision

        dialogue_act_decision = None
        if gen_result is not None:
            dialogue_act_decision = self._apply_dialogue_act_priors(disc_result, gen_result)

        if disc_label == gen_label:
            final_decision = disc_label
            decision_info['strategy'] = 'agreement'
            decision_info['reason'] = '判别模型与生成模型一致'

        elif gen_label not in ['Error:API_Failure', 'Error:API_Not_Available']:
            adaptive_high_thresh = 0.70 if disc_confidence < 0.6 else 0.90
            adaptive_mid_thresh = 0.60 if disc_confidence < 0.6 else 0.80
            adaptive_low_thresh = 0.65 if disc_confidence < 0.6 else 0.85

            if disc_label == 'Irrelevant' and disc_confidence > adaptive_high_thresh:
                final_decision = disc_label
                decision_info['strategy'] = 'trust_discriminative_irrelevant'
                decision_info['reason'] = f'Disc 高度确信是 Irrelevant ({disc_confidence:.4f})'

            elif gen_label == 'Irrelevant' and disc_label != 'Irrelevant' and disc_confidence > adaptive_mid_thresh:
                final_decision = disc_label
                decision_info['strategy'] = 'trust_discriminative_content'
                decision_info['reason'] = f'Disc 认为是观点 ({disc_label}, {disc_confidence:.4f})，LLM 过于保守'

    
            elif disc_label == 'Irrelevant' and gen_label != 'Irrelevant' and disc_confidence < 0.75:  
                final_decision = gen_label
                decision_info['strategy'] = 'trust_generative_content'
                decision_info['reason'] = f'LLM 发现了观点 ({gen_label})，Disc 不够确定 ({disc_confidence:.4f})'

            elif disc_label != 'Irrelevant' and gen_label != 'Irrelevant':
    
                if disc_confidence > adaptive_low_thresh:
                    final_decision = disc_label
                    decision_info['strategy'] = 'trust_discriminative_opinion'
                    decision_info['reason'] = f'观点细分由 Disc 决定 ({disc_label}, {disc_confidence:.4f})'
                else:
                    # 否则信任 LLM
                    final_decision = gen_label
                    decision_info['strategy'] = 'trust_generative_opinion'
                    decision_info['reason'] = f'Disc 不确定，信任 LLM ({gen_label})'

            else:
                final_decision = gen_label
                decision_info['strategy'] = 'trust_generative_default'
                decision_info['reason'] = f'默认信任 LLM ({gen_label})'

        else:
            final_decision = disc_label
            decision_info['strategy'] = 'fallback_discriminative'
            decision_info['reason'] = 'LLM 调用失败，降级使用判别模型'

        decision_info['final_decision'] = final_decision

        if return_details:
            return final_decision, decision_info
        else:
            return final_decision

    def _apply_dialogue_act_priors(
            self,
            disc_result: Dict,
            gen_result: Dict
    ) -> Optional[Tuple[str, str, str]]:
        if 'dialogue_act' not in disc_result:
            return None

        act_info = disc_result['dialogue_act']
        act_label = act_info['predicted_label'].lower()
        act_conf = act_info['confidence_score']
        opinion_label = disc_result['predicted_label']
        gen_label = gen_result.get('predicted_label', None)


        if act_conf > 0.85:
            if 'providing evidence' in act_label or 'reasoning' in act_label:
                if opinion_label == 'Strengthened':
                    return (
                        'Strengthened',
                        'dialogue_act_strong_prior_evidence',
                        f'对话行为"提供证据"强先验 → Strengthened (act_conf={act_conf:.3f})'
                    )

            if 'relating to another' in act_label or 'relate' in act_label:
                if opinion_label == 'Adopted':
                    return (
                        'Adopted',
                        'dialogue_act_strong_prior_relate',
                        f'对话行为"关联他人"强先验 → Adopted (act_conf={act_conf:.3f})'
                    )
                elif gen_label == 'Adopted':
                    return (
                        'Adopted',
                        'dialogue_act_strong_prior_relate_gen',
                        f'对话行为"关联他人"强先验 + LLM确认 → Adopted (act_conf={act_conf:.3f})'
                    )

            if 'making a claim' in act_label or 'claim' in act_label:
                if opinion_label in ['Adopted', 'Weakened']:
                    if gen_label in ['New', 'Strengthened', 'Refuted']:
                        return (
                            gen_label,
                            'dialogue_act_filter_claim',
                            f'对话行为"提出主张"排除{opinion_label}，信任LLM {gen_label} (act_conf={act_conf:.3f})'
                        )

        elif act_conf > 0.70:
            if 'providing evidence' in act_label or 'reasoning' in act_label:
                if opinion_label == 'Strengthened' and gen_label != 'Strengthened'
                    return (
                        'Strengthened',
                        'dialogue_act_medium_prior_evidence',
                        f'对话行为"提供证据"中等先验 + Disc一致 → Strengthened (act_conf={act_conf:.3f})'
                    )

    
            if 'asking' in act_label or 'press for' in act_label:
                if opinion_label == 'Strengthened':
                    if gen_label in ['New', 'Weakened', 'Irrelevant']:
                        return (
                            gen_label,
                            'dialogue_act_filter_question',
                            f'对话行为"提问/追问"排除Strengthened → {gen_label} (act_conf={act_conf:.3f})'
                        )
        return None

class HybridOpinionClassifier:
    """混合观点分类器 - 判别模型 + 生成模型 (级联路由优化版)"""

    def __init__(
            self,
            discriminative_model_dir: str,
            use_generative: bool = True,
            generative_model: str = "deepseek-v3",
            decision_threshold: float = 0.7,
            cascade_threshold: float = 0.65,  
            cascade_threshold_irrelevant: float = 0.6,  
            topk_threshold: float = 0.2,
            prefer_generative: bool = False,
            context_window: int = 5,
            use_knn_icl: bool = False, 
            knn_datastore_path: str = None,  
            knn_k: int = 3,  
            enable_risk_routing: bool = True  
    ):
        print("=" * 80)
        print("初始化混合观点分类器 (级联路由模式)")
        print("=" * 80)
        print(f"决策配置:")
        print(f"  - 级联截断阈值 (一般类别): {cascade_threshold}")
        print(f"  - 级联截断阈值 (Irrelevant): {cascade_threshold_irrelevant} [省钱优化!]")
        print(f"  - 联合决策阈值 (难样本): {decision_threshold}")
        print(f"  - 优先生成模型: {prefer_generative}")
        print(f"  - 风险路由 (Risk Routing): {'启用' if enable_risk_routing else '禁用'}")
        print(f"  - kNN-ICL: {'启用' if use_knn_icl else '禁用'}")
        if use_knn_icl:
            print(f"    * 向量库: {knn_datastore_path}")
            print(f"    * Top-K: {knn_k}")

        self.cascade_threshold = cascade_threshold
        self.cascade_threshold_irrelevant = cascade_threshold_irrelevant
        self.enable_risk_routing = enable_risk_routing  # 【新增】保存风险路由开关
        self.use_knn_icl = use_knn_icl
        self.knn_k = knn_k

        self.knowledge_base = None
        kb_dir = os.path.dirname(os.path.abspath(__file__))
        kb_candidates = [
            os.path.join(kb_dir, 'knowledge_base.json'),                    
            os.path.normpath(os.path.join(kb_dir, '..', 'knowledge_base.json'))  
        ]

        for kb_path in kb_candidates:
            if os.path.exists(kb_path):
                try:
                    with open(kb_path, 'r', encoding='utf-8') as f:
                        self.knowledge_base = json.load(f)
                    print(f"  ✓ 已加载知识库: {kb_path}")
                    print(f"    - 3-gram 规则数: {len(self.knowledge_base.get('sequence_3gram', {}))}")
                    print(f"    - 2-gram 规则数: {len(self.knowledge_base.get('sequence_2gram', {}))}")
                    print(f"    - 交互规则数:   {len(self.knowledge_base.get('interaction', {}))}")
                    print(f"    - 先验规则数:   {len(self.knowledge_base.get('priors', {}))}")
                except Exception as e:
                    print(f"[Warning] 知识库加载失败: {e}")
                    self.knowledge_base = None
                break

        self.kb_hint_match_count = 0      
        self.kb_hint_llm_count = 0       

        self.api_calls_total = 0
        self.fast_pass_total = 0
        self.fast_pass_by_irrelevant = 0
        self.fast_pass_by_high_conf = 0

        self.discriminative = DiscriminativeInference(discriminative_model_dir)


        self.knn_retriever = None
        if use_knn_icl:
            if knn_datastore_path is None or not os.path.exists(knn_datastore_path):
                print(f"[Warning] kNN 向量库不存在或未指定: {knn_datastore_path}")
                print("将禁用 kNN-ICL")
                self.use_knn_icl = False
            else:
                try:
                    from knn_retriever import KNNRetriever
                    self.knn_retriever = KNNRetriever(datastore_path=knn_datastore_path)
                except Exception as e:
                    print(f"[Warning] kNN 检索器加载失败: {e}")
                    print("将禁用 kNN-ICL")
                    self.use_knn_icl = False

        self.use_generative = use_generative and GENERATIVE_MODEL_AVAILABLE
        if self.use_generative:
            try:
                self.generative = GenerativeInference(
                    model_name=generative_model,
                    context_window=context_window,
                    use_knn_icl=self.use_knn_icl  
                )
            except Exception as e:
                print(f"[Warning] 生成模型初始化失败: {e}")
                print("将仅使用判别模型")
                self.use_generative = False
        else:
            self.generative = None
            print("仅使用判别模型 (生成模型未启用)")

        self.decision_maker = HybridDecisionMaker(
            discriminative_threshold=decision_threshold,
            topk_threshold=topk_threshold,
            prefer_generative=prefer_generative
        )

        print("=" * 80)
        print("混合分类器初始化完成")
        print("=" * 80)

    def classify_single(
            self,
            sentence: str,
            speaker: str,
            context_history: List[Dict] = None,
            previous_speaker: str = None,
            icl_examples: Optional[List[Dict]] = None,
            oracle_da_info: Dict = None,
            kb_hint: Optional[str] = None
    ) -> Dict:
       
        disc_result = self.discriminative.predict_single(sentence, speaker, context_history)
        disc_confidence = disc_result['confidence_score']
        disc_label = disc_result['predicted_label']

        oracle_risk_score = 0.0
        oracle_label_text = None
        if oracle_da_info:
            oracle_label_text = oracle_da_info['label']
            critical_keywords = [
                'claim', 'evidence', 'reason', 'relat',
                'disagree', 'agree', 'revoicing', 'ask'
            ]
            if any(k in oracle_label_text.lower() for k in critical_keywords):
                oracle_risk_score = 1.0

    
        predicted_risk_score = 0.0
        if 'dialogue_act' in disc_result and 'all_probabilities' in disc_result['dialogue_act']:
            da_probs = disc_result['dialogue_act']['all_probabilities']


            strong_high_risk = {
                "Providing Evidence/Reasoning",  
                "Making a Claim", 
                "Relating to Another Student",  
                "Revoicing",  
                "Restating", 
                "Press Reasoning",  
            }

            medium_high_risk = {
                "Press Accuracy",  
                "Asking for Info",  
            }

            for act_name, prob in da_probs.items():
                if act_name in strong_high_risk:
                    predicted_risk_score += prob * 1.0
                elif act_name in medium_high_risk:
                    predicted_risk_score += prob * 0.6
            

        if self.enable_risk_routing:
            final_risk_score = oracle_risk_score if oracle_da_info else predicted_risk_score
        else:
            final_risk_score = 0.0  

        should_skip_llm = False
        skip_reason = ""

        if not self.use_generative:
            should_skip_llm = True
            skip_reason = "生成模型未启用"
        elif self.enable_risk_routing and disc_label == 'Irrelevant' and final_risk_score <= 0.25:
            should_skip_llm = True
            skip_reason = f"Irrelevant 低风险快速通道 (Risk={final_risk_score:.2f})"
            self.fast_pass_by_irrelevant += 1
        elif disc_label == 'Irrelevant' and disc_confidence >= self.cascade_threshold_irrelevant:
            if final_risk_score > 0.25:
                should_skip_llm = False  
            else:
                should_skip_llm = True
                skip_reason = f"Irrelevant 快速通道 (Risk={final_risk_score:.2f})"
                self.fast_pass_by_irrelevant += 1
        elif disc_label != 'Irrelevant' and disc_confidence >= self.cascade_threshold:
            should_skip_llm = True
            skip_reason = f"高置信度快速通道 ({disc_confidence:.4f})"
            self.fast_pass_by_high_conf += 1

        # --- 分支 A: 快速通道 (Fast Pass) ---
        if should_skip_llm:
            self.fast_pass_total += 1  

            decision_info = {
                'discriminative_label': disc_label,
                'discriminative_confidence': disc_confidence,
                'generative_label': None,
                'strategy': 'cascade_fast_pass',
                'reason': skip_reason,
                'final_decision': disc_label,  
                'risk_score': final_risk_score 
            }

            return {
                'final_label': disc_label,
                'discriminative_result': disc_result,
                'generative_result': None,
                'decision_info': decision_info
            }

        # --- 分支 B: 慢速通道 (Slow Path) ---
        self.api_calls_total += 1

        if kb_hint:
            self.kb_hint_llm_count += 1
        if oracle_da_info:
            prompt_version = "v1"
            hint_to_pass = oracle_label_text
        else:
            prompt_version = "v1"
            hint_to_pass = None

        kb_hint_for_llm = None

        gen_result = self.generative.predict_single(
            sentence, speaker, context_history, previous_speaker,
            icl_examples=icl_examples,
            oracle_da_label=hint_to_pass,
            kb_hint=kb_hint_for_llm,
            prompt_version=prompt_version,
        )

        final_label, decision_info = self.decision_maker.make_decision(
            disc_result, gen_result, return_details=True, risk_score=final_risk_score
        )

        decision_info['risk_score'] = final_risk_score

        return {
            'final_label': final_label,
            'discriminative_result': disc_result,
            'generative_result': gen_result,
            'decision_info': decision_info
        }

    def classify_file(
            self,
            input_csv: str,
            output_csv: str,
            text_column: str = 'Sentence',
            speaker_column: str = 'Speaker',
            label_column: str = 'label',
            use_oracle_da: bool = False,  
            act_tag_column: str = 'Act Tag',  
            context_window: int = 5
    ):
      
        print(f"\n读取数据: {input_csv}")
        df = pd.read_csv(input_csv)
        print(f"样本数: {len(df)}")

        if use_oracle_da and act_tag_column not in df.columns:
            print(f"数据中未找到列 '{act_tag_column}'，无法使用 Oracle 模式！")
            use_oracle_da = False

        self.api_calls_total = 0
        self.fast_pass_total = 0
        self.fast_pass_by_irrelevant = 0
        self.fast_pass_by_high_conf = 0

        kb_available = (
            self.knowledge_base is not None
            and act_tag_column in df.columns
            and use_oracle_da
        )
    
        self.kb_hint_match_count = 0
        self.kb_hint_llm_count = 0


        kb_target_labels = {"New", "Strengthened", "Weakened", "Adopted", "Refuted"}
        kb_min_lift = 2.0  
        kb_min_prob = 0.5  

        if kb_available:

            df['ActTagClean'] = (
                df[act_tag_column]
                .astype(str)
                .str.lower()
                .str.strip()
                .str.replace(r'\s+', ' ', regex=True)
            )

            if 'class' in df.columns:
                grp = df.groupby('class')
                df['KB_prev_act'] = grp['ActTagClean'].shift(1)
                df['KB_prev_prev_act'] = grp['ActTagClean'].shift(2)
                df['KB_prev_speaker'] = grp[speaker_column].shift(1)
            else:
                df['KB_prev_act'] = df['ActTagClean'].shift(1)
                df['KB_prev_prev_act'] = df['ActTagClean'].shift(2)
                df['KB_prev_speaker'] = df[speaker_column].shift(1)

            if use_oracle_da:
                print("\n[Info] 已启用知识库提示 (Oracle DA + Knowledge Base)。")
            else:
                print("\n[Info] 已启用知识库提示 (使用数据中的 Act Tag 列构造序列；LLM 不直接看到真实 DA 标签)。")
            print("      将根据对话行为序列为部分样本构造 Knowledge Hint 并注入 LLM Prompt。")
        else:
            # 未实际启用知识库提示的几种情况
            if self.knowledge_base is None:
                print("\n[Info] 本次未使用知识库：未成功加载 knowledge_base.json。")
            elif act_tag_column not in df.columns:
                print(f"\n[Info] 本次未启用知识库提示：数据中找不到列 '{act_tag_column}'，或相关列缺失。")
            elif not use_oracle_da:
                # Proposed 场景：即便加载了知识库，也不会将 Knowledge Hint 注入 LLM
                print("\n[Info] 当前为 Proposed 模式：已加载 knowledge_base.json，但本轮不会向 LLM 注入 Knowledge Hint，仅使用判别模型与 LLM 自身语义推理。")
            else:
                print("\n[Info] 本次未启用知识库提示：配置不满足使用条件。")

        def build_kb_hint(row) -> Optional[str]:
            """根据当前行及其上下文的 Act Tag，在知识库中查找"高置信度"规则并返回描述。"""
            if not kb_available:
                return None

            kb = self.knowledge_base
            act_curr = row.get('ActTagClean', None)
            if pd.isna(act_curr) or not isinstance(act_curr, str) or not act_curr:
                return None

            prev_act = row.get('KB_prev_act', None)
            prev_prev_act = row.get('KB_prev_prev_act', None)
            prev_speaker = row.get('KB_prev_speaker', None)
            speaker = row.get(speaker_column, None)

            def is_valid_entry(entry: Dict) -> bool:
                """仅接受对内容类有显著提升的高置信规则，避免噪声先验。"""
                if not entry:
                    return False
                best_label = str(entry.get('best_label', ''))
                best_prob = float(entry.get('best_prob', 0.0))
                best_lift = float(entry.get('best_lift', 0.0))

                if best_label not in kb_target_labels:
                    return False
                if best_prob < kb_min_prob:
                    return False
                if best_lift < kb_min_lift:
                    return False
                return True

            if isinstance(prev_act, str) and isinstance(prev_prev_act, str):
                key3 = f"{prev_prev_act} -> {prev_act} -> {act_curr}"
                entry3 = kb.get('sequence_3gram', {}).get(key3, None)
                if is_valid_entry(entry3):
                    self.kb_hint_match_count += 1
                    return entry3.get('description', None)

            if isinstance(prev_act, str):
                key2 = f"{prev_act} -> {act_curr}"
                entry2 = kb.get('sequence_2gram', {}).get(key2, None)
                if is_valid_entry(entry2):
                    self.kb_hint_match_count += 1
                    return entry2.get('description', None)

   
            if isinstance(prev_act, str) and speaker != 'T' and isinstance(prev_speaker, str):
                if prev_speaker == 'T':
                    interaction_key = f"T->S:{prev_act}"
                else:
                    interaction_key = f"S->S:{prev_act}"

                entry_int = kb.get('interaction', {}).get(interaction_key, None)
                if is_valid_entry(entry_int):
                    self.kb_hint_match_count += 1
                    return entry_int.get('description', None)


            entry_prior = kb.get('priors', {}).get(act_curr, None)
            if is_valid_entry(entry_prior):
                self.kb_hint_match_count += 1
                return entry_prior.get('description', None)

            return None

        results = []
        context_history = []
        previous_speaker = None

        for idx, row in df.iterrows():
            sentence = row[text_column]
            speaker = row[speaker_column]


            oracle_da_info = None
            if use_oracle_da:
                raw_tag = str(row.get(act_tag_column, ""))
                oracle_da_info = {'label': raw_tag.lower()}
            kb_hint = build_kb_hint(row) if kb_available else None

            result = self.classify_single(
                sentence, speaker, context_history, previous_speaker,
                icl_examples=None,
                oracle_da_info=oracle_da_info,
                kb_hint=kb_hint
            )

            result_row = {
                'index': idx,
                'Speaker': speaker,
                'Sentence': sentence,
                'final_label': result['final_label'],
                'disc_label': result['discriminative_result']['predicted_label'],
                'disc_confidence': result['discriminative_result']['confidence_score'],
                'used_oracle_da': oracle_da_info['label'] if oracle_da_info else 'No',
                'strategy': result['decision_info'].get('strategy', 'unknown')
            }

            disc_res = result['discriminative_result']
            if 'dialogue_act' in disc_res:
                da_info = disc_res['dialogue_act']
                result_row['predicted_da'] = da_info.get('predicted_label', 'N/A')
                result_row['da_confidence'] = da_info.get('confidence_score', 0.0)

                # 提取概率最高的前3个DA（用于分析）
                if 'all_probabilities' in da_info:
                    da_probs = da_info['all_probabilities']
                    sorted_das = sorted(da_probs.items(), key=lambda x: x[1], reverse=True)[:3]
                    result_row['top3_das'] = '; '.join([f"{name}({prob:.2f})" for name, prob in sorted_das])
                else:
                    result_row['top3_das'] = 'N/A'
            else:
                result_row['predicted_da'] = 'N/A'
                result_row['da_confidence'] = 0.0
                result_row['top3_das'] = 'N/A'


            result_row['risk_score'] = result['decision_info'].get('risk_score', 0.0)


            if result.get('generative_result'):
                result_row['gen_label'] = result['generative_result']['predicted_label']
            else:
                result_row['gen_label'] = None 

            if result['decision_info']:
                result_row['strategy'] = result['decision_info']['strategy']
                result_row['reason'] = result['decision_info']['reason']

            if label_column in df.columns:
                true_label_id = row[label_column]
                result_row['true_label_id'] = true_label_id
                label_names = ['Irrelevant', 'New', 'Strengthened', 'Weakened', 'Adopted', 'Refuted']
                try:
                    if pd.isna(true_label_id):
                        true_label_name = "Unknown"
                    elif isinstance(true_label_id, (int, float)):
                        true_label_name = label_names[int(true_label_id)]
                    else:
                        true_label_name = true_label_id
                except:
                    true_label_name = str(true_label_id)

                result_row['true_label'] = true_label_name
                result_row['is_correct'] = (result['final_label'] == true_label_name)

            results.append(result_row)

            context_history.append({
                'speaker': speaker,
                'sentence': sentence,
                'label': result['final_label']
            })

            if len(context_history) > context_window:
                context_history.pop(0)
            previous_speaker = speaker

            if (idx + 1) % 20 == 0:
                savings_rate = (self.fast_pass_total / (idx + 1)) * 100 if (idx + 1) > 0 else 0
                print(
                    f"  进度: {idx + 1}/{len(df)} | API调用: {self.api_calls_total} | 快速通道: {self.fast_pass_total} (Irrelevant: {self.fast_pass_by_irrelevant}) | 节省率: {savings_rate:.1f}%")

        result_df = pd.DataFrame(results)
        result_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存: {output_csv}")

        total_samples = len(df)
        savings_rate = (self.fast_pass_total / total_samples * 100) if total_samples > 0 else 0

        print("\n" + "=" * 80)
        print("级联路由成本效益分析 (Cascade Routing Cost Analysis)")
        print("=" * 80)
        print(f"\n【总体统计】")
        print(f"  总样本数:              {total_samples}")
        print(f"  API调用次数 (花钱):  {self.api_calls_total:4d} ({self.api_calls_total / total_samples * 100:5.1f}%)")
        print(f"  快速通道次数 (省钱): {self.fast_pass_total:4d} ({savings_rate:5.1f}%)")

        print(f"\n【快速通道细分】")
        print(
            f"  Irrelevant类快速通过:  {self.fast_pass_by_irrelevant:4d} (阈值={self.cascade_threshold_irrelevant:.2f})")
        print(f"  高置信度快速通过:      {self.fast_pass_by_high_conf:4d} (阈值={self.cascade_threshold:.2f})")

        # 估算成本节省（假设每次API调用0.01元）
        cost_per_call = 0.01  # 元/次
        total_cost_original = total_samples * cost_per_call
        actual_cost = self.api_calls_total * cost_per_call
        saved_cost = total_cost_original - actual_cost

        print(f"\n【成本估算】(假设 {cost_per_call}元/调用)")
        print(f"  原始成本 (全部调用LLM): ¥{total_cost_original:.2f}")
        print(f"  实际成本 (级联路由):    ¥{actual_cost:.2f}")
        print(f"  节省成本:               ¥{saved_cost:.2f} ({savings_rate:.1f}%)")

        # 知识库使用情况总结（仅在本次运行启用了 KB 提示时打印）
        if kb_available:
            print(f"\n【知识库使用情况】")
            print(f"  匹配到知识库规则的样本数: {self.kb_hint_match_count}")
            print(f"  其中实际送入LLM并携带 Knowledge Hint 的样本数: {self.kb_hint_llm_count}")

        if label_column in df.columns:
            accuracy = result_df['is_correct'].mean()
            print(f"\n【分类性能】")
            print(f"  整体准确率: {accuracy:.4f}")


            fast_pass_results = result_df[result_df['strategy'] == 'cascade_fast_pass']
            slow_path_results = result_df[result_df['strategy'] != 'cascade_fast_pass']

            if len(fast_pass_results) > 0:
                fast_acc = fast_pass_results['is_correct'].mean()
                print(f"  快速通道准确率: {fast_acc:.4f} (n={len(fast_pass_results)})")

            if len(slow_path_results) > 0:
                slow_acc = slow_path_results['is_correct'].mean()
                print(f"  慢速通道准确率: {slow_acc:.4f} (n={len(slow_path_results)})")

            print(f"\n【决策策略分布】")
            strategy_counts = result_df['strategy'].value_counts()
            for strategy, count in strategy_counts.items():
                print(f"  {strategy:30s}: {count:4d} ({count / total_samples * 100:5.1f}%)")
