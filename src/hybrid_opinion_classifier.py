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

# 导入多任务模型
try:
    from multi_task_model import MultiTaskDialogueModel

    MULTI_TASK_AVAILABLE = True
except ImportError:
    MULTI_TASK_AVAILABLE = False
    print("[警告] 未找到 multi_task_model.py，多任务模型不可用")

# 导入生成模型
try:
    from openai import OpenAI
    from dotenv import load_dotenv
    import time

    # 加载环境变量
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    # 初始化 OpenAI 客户端（v1.0+ 语法）
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


# =============================================================================
# 1. 判别模型定义 (需要与训练时一致)
# =============================================================================

class DialogueAwareModel(nn.Module):
    """
    对话感知判别模型
    需要与discriminative_model_training.py中的模型结构完全一致
    """

    def __init__(self, model_path, num_classes=6, dropout=0.5):
        super().__init__()
        # 使用 AutoModel 支持 DeBERTa/RoBERTa 等多种模型
        self.encoder = AutoModel.from_pretrained(model_path)
        self.hidden_size = self.encoder.config.hidden_size
        self.num_classes = num_classes

        # 分类头 - 与训练脚本保持一致
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask, return_confidence=False, return_embedding=False):
        # 使用 encoder，并从 last_hidden_state 取 CLS token (第0位)
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_token_embedding = outputs.last_hidden_state[:, 0, :]

        # 【新增】如果只需要 embedding（用于 kNN 检索），直接返回
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


# =============================================================================
# 2. 判别模型推理器
# =============================================================================

class DiscriminativeInference:
    """判别模型推理类"""

    def __init__(self, model_dir: str, device=None):
        self.model_dir = model_dir
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载配置
        config_path = os.path.join(model_dir, 'config.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        # 获取模型路径（从config读取，支持本地路径）
        model_name_or_path = self.config.get('model_name', 'roberta-base')

        print(f"  模型路径: {model_name_or_path}")

        # 加载分词器（使用 AutoTokenizer 支持更多模型）
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)

        # 添加训练时使用的 special tokens（必须与训练时一致！）
        context_window = self.config.get('context_window', 5)
        special_tokens_list = [
            "[TEACHER]", "[CURRENT]", "[OTHER]", "Unknown"
        ]
        turn_tokens = [f"[TURN_{i}]" for i in range(context_window + 2)]
        special_tokens_list.extend(turn_tokens)
        num_added = self.tokenizer.add_tokens(special_tokens_list)
        print(f"  添加 special tokens: {num_added} 个")

        # 加载权重先检查模型类型
        checkpoint_path = os.path.join(model_dir, 'best_model.pth')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"模型文件不存在: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 检测是否为多任务模型
        state_dict_keys = checkpoint['model_state_dict'].keys()
        is_multi_task = 'dialogue_act_head.0.weight' in state_dict_keys or 'shared_layer.1.weight' in state_dict_keys

        if is_multi_task:
            if not MULTI_TASK_AVAILABLE:
                raise ImportError("检测到多任务模型，但 multi_task_model.py 不可用")
            print("  检测到: 多任务模型 (MultiTaskDialogueModel)")
            self.is_multi_task = True

            # 检测是否有 feature_fusion 层（增强版本 vs 基线版本）
            has_feature_fusion = any('feature_fusion' in k for k in state_dict_keys)

            if has_feature_fusion:
                print("  ⚠️ 检测到增强版本模型（带feature_fusion），但当前代码仅支持基线版本")
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

            # 加载权重（支持兼容性）
            # 如果检测到 feature_fusion 或为了安全起见，直接使用兼容模式
            if has_feature_fusion:
                print(f"  使用兼容模式加载权重（跳过额外的层）...")
                model_state = self.model.state_dict()
                checkpoint_state = checkpoint['model_state_dict']

                # 只加载当前模型中存在且形状匹配的参数
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

                # 检查checkpoint中有但模型中没有的键
                for key in checkpoint_state.keys():
                    if key not in model_state:
                        extra_keys.append(key)

                # 加载过滤后的参数
                self.model.load_state_dict(filtered_state, strict=False)

                print(f"  ✓ 兼容性加载完成，加载了 {len(filtered_state)}/{len(model_state)} 个参数")
                if extra_keys:
                    print(f"  ℹ️ Checkpoint中有 {len(extra_keys)} 个额外的参数（已忽略）:")
                    for key in extra_keys[:5]:
                        print(f"    - {key}")
                    if len(extra_keys) > 5:
                        print(f"    ... 还有 {len(extra_keys) - 5} 个")
                if skipped_keys:
                    print(f"  ⚠️ 跳过了 {len(skipped_keys)} 个参数（将使用随机初始化）")
                    for key in skipped_keys[:5]:
                        print(f"    - {key}")
                    if len(skipped_keys) > 5:
                        print(f"    ... 还有 {len(skipped_keys) - 5} 个")
            else:
                # 尝试严格加载
                try:
                    self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
                    print(f"  ✓ 模型权重加载成功")
                except RuntimeError as e:
                    if 'size mismatch' in str(e) or 'Unexpected key' in str(e) or 'Missing key' in str(e):
                        print(f"  ⚠️ 严格加载失败，使用兼容模式...")
                        # 过滤掉不匹配的键
                        model_state = self.model.state_dict()
                        checkpoint_state = checkpoint['model_state_dict']

                        # 只加载形状匹配的参数
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

                        # 加载过滤后的参数
                        self.model.load_state_dict(filtered_state, strict=False)

                        print(f"  ✓ 兼容性加载完成，加载了 {len(filtered_state)}/{len(model_state)} 个参数")
                        if skipped_keys:
                            print(f"  ⚠️ 跳过了 {len(skipped_keys)} 个参数（将使用随机初始化）")
                            for key in skipped_keys[:5]:
                                print(f"    - {key}")
                            if len(skipped_keys) > 5:
                                print(f"    ... 还有 {len(skipped_keys) - 5} 个")
                    else:
                        raise
            # 与训练数据中的 act_label 含义保持一致
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
            print("  检测到: 单任务模型 (DialogueAwareModel)")
            self.is_multi_task = False
            # 加载单任务模型
            self.model = DialogueAwareModel(
                model_path=model_name_or_path,
                num_classes=self.config.get('num_classes', 6),
                dropout=self.config.get('dropout', 0.6)
            )
            # 调整 embedding
            self.model.encoder.resize_token_embeddings(len(self.tokenizer))
            print(f"  模型 embedding 大小: {self.model.encoder.embeddings.word_embeddings.weight.shape[0]}")
            # 加载权重
            self.model.load_state_dict(checkpoint['model_state_dict'])

        self.model.to(self.device)
        self.model.eval()

        self.label_names = ['Irrelevant', 'New', 'Strengthened', 'Weakened', 'Adopted', 'Refuted']

        print(f"✓ 判别模型加载成功")
        print(f"  设备: {self.device}")
        val_f1 = checkpoint.get('val_macro_f1', 0)
        if isinstance(val_f1, (int, float)):
            print(f"  验证集Macro-F1: {val_f1:.4f}")
        else:
            print(f"  验证集Macro-F1: {val_f1}")

    def _get_role_name(self, speaker_name, current_speaker_name):
        """计算相对角色（与训练时保持一致）"""
        teacher_names = {'T', 'Teacher', 'Ms. G', 'Mrs. G'}
        s_name = str(speaker_name).strip()

        if s_name.startswith('T') or s_name in teacher_names:
            return "[TEACHER]"
        if s_name == str(current_speaker_name).strip():
            return "[CURRENT]"
        return "[OTHER]"

    def predict_single(self, text: str, speaker: str = None, context_history: List[Dict] = None) -> Dict:
        """
        预测单个样本（支持上下文，与训练时保持一致）

        Args:
            text: 输入文本（目标句子）
            speaker: 当前说话者
            context_history: 上下文历史 [{'speaker': 'T', 'sentence': '...', 'label': '...'}, ...]

        Returns:
            包含预测结果、置信度和 CLS embedding 的字典
        """
        # 构造与训练时一致的输入格式
        use_context = self.config.get('use_context', True)
        context_window = self.config.get('context_window', 5)
        use_turn_indicators = self.config.get('use_turn_indicators', True)

        # Target text: [CURRENT] 当前句子
        target_text = f"[CURRENT] {text}"

        # Context text: 构造上下文（如果提供）
        context_text = ""
        if use_context and context_history:
            context_parts = []
            # 只取最近的 context_window 条
            recent_context = context_history[-context_window:] if len(
                context_history) > context_window else context_history

            for turn_idx, ctx in enumerate(recent_context):
                ctx_speaker = ctx.get('speaker', 'Unknown')
                ctx_sentence = ctx.get('sentence', '')

                # 获取角色标记
                role = self._get_role_name(ctx_speaker, speaker)

                # 添加轮次标记（如果启用）
                if use_turn_indicators:
                    turn_marker = f"[TURN_{turn_idx}]"
                    context_parts.append(f"{turn_marker} {role} {ctx_sentence}")
                else:
                    context_parts.append(f"{role} {ctx_sentence}")

            context_text = " ".join(context_parts)

        # 分词（使用 text_pair 结构，与训练时一致）
        # 注意：如果没有上下文，只传 target_text 作为 text 参数
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
            # 没有上下文时，只用 target_text
            encoding = self.tokenizer(
                text=target_text,
                max_length=self.config.get('max_length', 256),
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # 推理
        with torch.no_grad():
            if self.is_multi_task:
                # 多任务模型返回多个输出
                outputs = self.model(input_ids, attention_mask, return_confidence=True, return_embedding=True)

                # 提取观点演化结果
                opinion_info = outputs['opinion_confidence']
                cls_embedding = outputs.get('cls_embedding', None)
                if cls_embedding is not None:
                    cls_embedding = cls_embedding[0].cpu().numpy()

                probs = opinion_info['probabilities'][0].cpu().numpy()
                predicted_class = opinion_info['predicted_classes'][0].item()
                confidence_score = opinion_info['confidence_scores'][0].item()
                top_k_probs = opinion_info['top_k_probs'][0].cpu().numpy()
                top_k_indices = opinion_info['top_k_indices'][0].cpu().numpy()

                # 提取对话行为结果（用于后续决策）
                dialogue_act_info = outputs['dialogue_act_confidence']
                dialogue_act_pred = dialogue_act_info['predicted_classes'][0].item()
                dialogue_act_conf = dialogue_act_info['confidence_scores'][0].item()
                dialogue_act_probs = dialogue_act_info['probabilities'][0].cpu().numpy()
            else:
                # 单任务模型
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

        # 添加 CLS embedding 用于 kNN 检索
        if cls_embedding is not None:
            result['cls_embedding'] = cls_embedding

        # 如果是多任务模型，添加对话行为信息
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


# =============================================================================
# 3. 生成模型包装器 (基于 classify_opinions.py)
# =============================================================================

# 定义有效标签
VALID_LABELS = {
    "Irrelevant", "New", "Strengthened",
    "Weakened", "Adopted", "Refuted"
}


def call_llm_api(messages: List[Dict[str, str]], model: str = "deepseek-v3",
                 max_retries: int = 2) -> str:
    """
    调用大模型API并处理基本错误
    (更新为 OpenAI v1.0+ API)
    """
    if not GENERATIVE_MODEL_AVAILABLE or openai_client is None:
        return "Error:API_Not_Available"

    for attempt in range(max_retries):
        try:
            # OpenAI v1.0+ 语法
            response = openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=50
            )
            raw_label = response.choices[0].message.content.strip()

            # 解析标签（处理多种格式）
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
    """
    从LLM响应中提取标签（支持思维链格式）
    处理多种格式: "New", "New (新观点)", "1. New" 等
    以及思维链输出（标签在最后一行）
    """
    raw_text = raw_text.strip()

    # 直接匹配
    if raw_text in VALID_LABELS:
        return raw_text

    # 标签名列表（按顺序）
    label_names = ['Irrelevant', 'New', 'Strengthened', 'Weakened', 'Adopted', 'Refuted']

    # 策略1: 优先提取最后一行（如果是思维链输出，标签在最后）
    lines = raw_text.split('\n')
    last_line = lines[-1].strip() if lines else ""

    # 检查最后一行是否是纯标签
    for label in label_names:
        if last_line.lower() == label.lower():
            return label

    # 策略2: 在最后一行中查找标签
    for label in label_names:
        if label.lower() in last_line.lower():
            return label

    # 策略3: 在整个响应中查找（向后兼容）
    for label in label_names:
        if label.lower() in raw_text.lower():
            return label

    return "Irrelevant"  # 默认返回


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
    act_hint: str = None,  # <--- 新增参数
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
    {context_str if context_str else "（没有上下文，这是第一句话发言）"}

    **待分类的发言 (Current Utterance):**
    - **发言人 (Speaker)**: {current_speaker} (注意：'T' 代表老师)
    - **发言人一致性 (Consistency)**: {consistency}
    - **句子 (Sentence)**: {current_sentence}

    """

    # 注入对话行为提示（软提示）
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
    act_hint: str = None,  # <--- [新增参数]
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

    # [新增] 注入对话行为提示
    if act_hint:
        user_prompt += f"\n    - **已知对话行为 (Dialogue Act)**: {act_hint} (提示：请参考此意图进行判断)\n"

    # Legacy prompt 忽略 kb_hint

    user_prompt += "\n    请参考上述示例，先简要分析（1-2句话），然后在最后一行输出标签（仅一个词）。"

    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt.strip()}
    ]


class GenerativeInference:
    """生成模型推理类 (基于 classify_opinions.py)

    通过 prompt_version 控制使用 v1（软先验）还是 v2（强先验）模板。
    """

    def __init__(self, model_name: str = "deepseek-v3", context_window: int = 5, use_knn_icl: bool = False):
        if not GENERATIVE_MODEL_AVAILABLE:
            raise ImportError("生成模型未可用，请检查OPENAI_API_KEY和依赖")

        self.model_name = model_name
        self.context_window = context_window
        self.use_knn_icl = use_knn_icl  # 【新增】是否使用 kNN-ICL
        print(f"✓ 生成模型初始化完成")
        print(f"  模型: {model_name}")
        print(f"  上下文窗口: {context_window}")
        print(f"  kNN-ICL: {'启用' if use_knn_icl else '禁用'}")

    def predict_single(self, sentence: str, speaker: str, context_history: List[Dict],
                       previous_speaker: Optional[str] = None,
                       icl_examples: Optional[List[Dict]] = None,
                       oracle_da_label: str = None,  # <--- 新增
                       kb_hint: Optional[str] = None,
                       prompt_version: str = "v1"  # v1: 软先验 (默认，用于 Proposed)；v2: 强先验 (用于 Oracle)
                       ) -> Dict:
        """
        使用生成模型预测

        Args:
            sentence: 当前句子
            speaker: 说话者
            context_history: 上下文历史 [{'speaker': 'T', 'sentence': '...', 'label': '...'}, ...]
            previous_speaker: 前一个说话者 (用于确定consistency)
            icl_examples: 【新增】检索到的 kNN 示例 [{'sentence': '...', 'label_name': 'New', 'similarity': 0.95}, ...]

        Returns:
            包含预测结果的字典
        """
        # 1. 确定发言人一致性（与 classify_opinions1.py 保持一致）
        if previous_speaker is None:
            consistency = "switch"  # 第一句话
        elif speaker == previous_speaker:
            consistency = "same"
        else:
            consistency = "switch"

        # 2. 构建上下文字符串 (只取最近的context_window条)
        recent_context = context_history[-self.context_window:] if context_history else []
        context_lines = []
        for ctx in recent_context:
            ctx_speaker = ctx.get('speaker', '')
            ctx_sentence = ctx.get('sentence', '')
            if ctx_sentence:
                context_lines.append(f"{ctx_speaker}: {ctx_sentence}")
        context_str = "\n".join(context_lines)

        # 3. 构建 Prompt
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

        # 4. 调用LLM
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


# =============================================================================
# 4. 联合决策策略
# =============================================================================

class HybridDecisionMaker:
    """混合模型联合决策器"""

    def __init__(
            self,
            discriminative_threshold: float = 0.8,  # 提高到0.8，降低对判别模型的信任
            agreement_weight: float = 0.5,
            topk_threshold: float = 0.2,  # 降低到0.2，更容易采纳生成模型
            prefer_generative: bool = False  # 新增：是否优先采纳生成模型
    ):
        """初始化联合决策器配置"""
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

        # [关键修改] 安全获取 gen_label
        # 如果 gen_result 是 None (走了快速通道)，则 gen_label 设为 None
        gen_label = gen_result['predicted_label'] if gen_result else None

        decision_info = {
            'discriminative_label': disc_label,
            'discriminative_confidence': disc_confidence,
            'generative_label': gen_label,
        }

        # =========================================================
        # 保守版优先级规则：在进入复杂策略前，先保护
        # "Irrelevant vs 内容类" 这两种最容易被误改的情况
        # =========================================================

        # 情况 A: 判别模型认为 Irrelevant，而 LLM 认为是内容类
        # 保守策略：
        #   - 对整体样本：当判别模型置信度达到中等及以上时（>=0.5），保护 Irrelevant，
        #   - 但若该样本风险分较高（来自 DA 模型，例如 Press Accuracy / Claim / Evidence），
        #     说明它更可能是“伪 Irrelevant”的内容类，此时提高阈值到 0.8，
        #     只有 Disc 极高置信度时才强行压制 LLM，给高风险样本更多让路空间。
        if gen_label is not None and disc_label == 'Irrelevant' and gen_label != 'Irrelevant':
            guard_threshold = 0.5
            # 若提供了风险分且风险较高，则提高 guard 阈值，降低对 Disc 的盲目信任
            if risk_score is not None and risk_score > 0.5:
                guard_threshold = 0.8

            if disc_confidence >= guard_threshold:
                final_decision = disc_label
                decision_info['strategy'] = 'guard_irrelevant_from_llm'
                # 在 reason 中附带当前阈值和风险分，便于后续分析
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

        # 情况 B: 判别模型认为是内容类，而 LLM 认为 Irrelevant
        # 保守策略（弱化版）：仅当判别模型置信度非常高时（>=0.8），
        # 才阻止 LLM 将其压成 Irrelevant，避免过度相信判别模型
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

        # =========================================================
        # 以下是原有的逻辑 (保留不动)
        # =========================================================

        # [原有的 _apply_dialogue_act_priors 调用可以保留，也可以注释掉，
        # 因为上面的 Oracle 逻辑已经覆盖了大部分情况]
        # [修复] 只有当生成模型真正运行了（gen_result 不为 None）时，才去应用先验规则
        # 如果走了快速通道，gen_result 是 None，没法比较，直接跳过
        dialogue_act_decision = None
        if gen_result is not None:
            dialogue_act_decision = self._apply_dialogue_act_priors(disc_result, gen_result)
        # ... (后续代码保持不变)

        # 策略 1: 两人意见一致 (Agreement) - 最稳
        if disc_label == gen_label:
            final_decision = disc_label
            decision_info['strategy'] = 'agreement'
            decision_info['reason'] = '判别模型与生成模型一致'

        # 策略 2: 意见不一致时，使用改进的投票策略（自适应阈值）
        elif gen_label not in ['Error:API_Failure', 'Error:API_Not_Available']:
            # 【自适应阈值】根据整体置信度水平调整
            # 如果 Disc 平均置信度很低（<0.6），说明是少样本模型，降低阈值
            # [优化] 提高阈值要求，减少对低置信度判别结果的过度信任
            adaptive_high_thresh = 0.70 if disc_confidence < 0.6 else 0.90  # 从0.55/0.85提高
            adaptive_mid_thresh = 0.60 if disc_confidence < 0.6 else 0.80  # 从0.45/0.70提高
            adaptive_low_thresh = 0.65 if disc_confidence < 0.6 else 0.85  # 从0.50/0.75提高

            # 子策略 2.1: 如果 Disc 置信度很高，且预测的是 Irrelevant
            # 说明这可能真的是无关发言，信任 Disc
            if disc_label == 'Irrelevant' and disc_confidence > adaptive_high_thresh:
                final_decision = disc_label
                decision_info['strategy'] = 'trust_discriminative_irrelevant'
                decision_info['reason'] = f'Disc 高度确信是 Irrelevant ({disc_confidence:.4f})'

            # 子策略 2.2: 如果 LLM 预测 Irrelevant，但 Disc 中等置信度预测观点类
            # 这种情况下，Disc 可能更准（因为 LLM 容易过度预测 Irrelevant）
            elif gen_label == 'Irrelevant' and disc_label != 'Irrelevant' and disc_confidence > adaptive_mid_thresh:
                final_decision = disc_label
                decision_info['strategy'] = 'trust_discriminative_content'
                decision_info['reason'] = f'Disc 认为是观点 ({disc_label}, {disc_confidence:.4f})，LLM 过于保守'

            # 子策略 2.3: 如果 Disc 预测 Irrelevant，但置信度不高，LLM 预测观点类
            # 信任 LLM（LLM 可能捕捉到了深层语义）
            # [优化] 放宽条件，让生成模型更容易被采纳
            elif disc_label == 'Irrelevant' and gen_label != 'Irrelevant' and disc_confidence < 0.75:  # 降低阈值
                final_decision = gen_label
                decision_info['strategy'] = 'trust_generative_content'
                decision_info['reason'] = f'LLM 发现了观点 ({gen_label})，Disc 不够确定 ({disc_confidence:.4f})'

            # 子策略 2.4: 两者都预测观点类（New/Strengthened等），但类别不同
            # 这种情况下，优先信任 Disc（因为它在观点细分上经过专门训练）
            elif disc_label != 'Irrelevant' and gen_label != 'Irrelevant':
                # 如果 Disc 置信度够高，信任 Disc
                if disc_confidence > adaptive_low_thresh:
                    final_decision = disc_label
                    decision_info['strategy'] = 'trust_discriminative_opinion'
                    decision_info['reason'] = f'观点细分由 Disc 决定 ({disc_label}, {disc_confidence:.4f})'
                else:
                    # 否则信任 LLM
                    final_decision = gen_label
                    decision_info['strategy'] = 'trust_generative_opinion'
                    decision_info['reason'] = f'Disc 不确定，信任 LLM ({gen_label})'

            # 子策略 2.5: 其他情况，默认信任 LLM
            else:
                final_decision = gen_label
                decision_info['strategy'] = 'trust_generative_default'
                decision_info['reason'] = f'默认信任 LLM ({gen_label})'

        # 策略 3: LLM 挂了，只能用 Disc 兜底
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
        """
        应用对话行为先验规则

        Returns:
            (final_label, strategy, reason) 或 None（无先验约束）
        """
        # 检查是否有对话行为信息（多任务模型）
        if 'dialogue_act' not in disc_result:
            return None

        act_info = disc_result['dialogue_act']
        act_label = act_info['predicted_label'].lower()
        act_conf = act_info['confidence_score']
        opinion_label = disc_result['predicted_label']
        gen_label = gen_result.get('predicted_label', None)

        # =========================================================
        # 强先验规则 (对话行为置信度 > 0.85)
        # =========================================================
        if act_conf > 0.85:
            # 规则1: Providing Evidence → Strengthened (85%+概率)
            if 'providing evidence' in act_label or 'reasoning' in act_label:
                if opinion_label == 'Strengthened':
                    return (
                        'Strengthened',
                        'dialogue_act_strong_prior_evidence',
                        f'对话行为"提供证据"强先验 → Strengthened (act_conf={act_conf:.3f})'
                    )

            # 规则2: Relating to Another → Adopted (80%+概率)
            if 'relating to another' in act_label or 'relate' in act_label:
                if opinion_label == 'Adopted':
                    return (
                        'Adopted',
                        'dialogue_act_strong_prior_relate',
                        f'对话行为"关联他人"强先验 → Adopted (act_conf={act_conf:.3f})'
                    )
                # 如果判别模型预测不是Adopted，但生成模型是，也信任
                elif gen_label == 'Adopted':
                    return (
                        'Adopted',
                        'dialogue_act_strong_prior_relate_gen',
                        f'对话行为"关联他人"强先验 + LLM确认 → Adopted (act_conf={act_conf:.3f})'
                    )

            # 规则3: Making a Claim → 排除Adopted/Weakened，优先New/Strengthened
            if 'making a claim' in act_label or 'claim' in act_label:
                # 如果判别模型预测Adopted或Weakened，很可能错误
                if opinion_label in ['Adopted', 'Weakened']:
                    # 检查生成模型是否给出了更合理的预测
                    if gen_label in ['New', 'Strengthened', 'Refuted']:
                        return (
                            gen_label,
                            'dialogue_act_filter_claim',
                            f'对话行为"提出主张"排除{opinion_label}，信任LLM {gen_label} (act_conf={act_conf:.3f})'
                        )

        # =========================================================
        # 中等先验规则 (对话行为置信度 0.70 - 0.85)
        # =========================================================
        elif act_conf > 0.70:
            # 规则4: Providing Evidence → 倾向Strengthened（弱约束）
            if 'providing evidence' in act_label or 'reasoning' in act_label:
                if opinion_label == 'Strengthened' and gen_label != 'Strengthened':
                    # 判别模型和对话行为一致，生成模型不同意 → 信任判别模型
                    return (
                        'Strengthened',
                        'dialogue_act_medium_prior_evidence',
                        f'对话行为"提供证据"中等先验 + Disc一致 → Strengthened (act_conf={act_conf:.3f})'
                    )

            # 规则5: Asking/Press for Accuracy → 不太可能是Strengthened
            if 'asking' in act_label or 'press for' in act_label:
                if opinion_label == 'Strengthened':
                    # 疑问/追问类对话行为，不太可能是强化观点
                    if gen_label in ['New', 'Weakened', 'Irrelevant']:
                        return (
                            gen_label,
                            'dialogue_act_filter_question',
                            f'对话行为"提问/追问"排除Strengthened → {gen_label} (act_conf={act_conf:.3f})'
                        )

        # 无先验约束
        return None


# =============================================================================
# 5. 完整推理管道
# =============================================================================

class HybridOpinionClassifier:
    """混合观点分类器 - 判别模型 + 生成模型 (级联路由优化版)"""

    def __init__(
            self,
            discriminative_model_dir: str,
            use_generative: bool = True,
            generative_model: str = "deepseek-v3",
            decision_threshold: float = 0.7,  # 联合决策阈值 (用于难样本)
            cascade_threshold: float = 0.65,  # 一般类别的级联截断阈值 [优化: 0.85→0.65]
            cascade_threshold_irrelevant: float = 0.6,  # [新增] Irrelevant类别专用阈值（更低） [优化: 0.75→0.6]
            topk_threshold: float = 0.2,
            prefer_generative: bool = False,
            context_window: int = 5,
            use_knn_icl: bool = False,  # 【新增】是否使用 kNN-ICL
            knn_datastore_path: str = None,  # 【新增】kNN 向量库路径
            knn_k: int = 3,  # 【新增】kNN 检索数量
            enable_risk_routing: bool = True  # 【新增】是否启用风险路由（对话行为辅助）
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

        # 知识库 (对话行为序列 -> 观点演化 统计规律)
        self.knowledge_base = None
        kb_dir = os.path.dirname(os.path.abspath(__file__))
        kb_candidates = [
            os.path.join(kb_dir, 'knowledge_base.json'),                     # 与本文件同目录
            os.path.normpath(os.path.join(kb_dir, '..', 'knowledge_base.json'))  # 项目根目录
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

        # 知识库使用统计（按 classify_file 一次运行统计）
        self.kb_hint_match_count = 0      # 匹配到任何知识库规则的样本数
        self.kb_hint_llm_count = 0        # 实际带 Knowledge Hint 发送给 LLM 的样本数

        # 成本统计计数器
        self.api_calls_total = 0
        self.fast_pass_total = 0
        self.fast_pass_by_irrelevant = 0
        self.fast_pass_by_high_conf = 0

        # 加载判别模型
        self.discriminative = DiscriminativeInference(discriminative_model_dir)

        # 【新增】加载 kNN 检索器
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

        # 加载生成模型
        self.use_generative = use_generative and GENERATIVE_MODEL_AVAILABLE
        if self.use_generative:
            try:
                self.generative = GenerativeInference(
                    model_name=generative_model,
                    context_window=context_window,
                    use_knn_icl=self.use_knn_icl  # 【新增】传递 kNN-ICL 标志
                )
            except Exception as e:
                print(f"[Warning] 生成模型初始化失败: {e}")
                print("将仅使用判别模型")
                self.use_generative = False
        else:
            self.generative = None
            print("仅使用判别模型 (生成模型未启用)")

        # 决策器
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
        """
        分类单个话语 (修复版：强制 Early Return，杜绝空值)
        """
        # 1. 判别模型推理
        disc_result = self.discriminative.predict_single(sentence, speaker, context_history)
        disc_confidence = disc_result['confidence_score']
        disc_label = disc_result['predicted_label']

        # 2. 路由逻辑准备
        # A. Oracle 风险分
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

        # B. 预测风险分
        predicted_risk_score = 0.0
        if 'dialogue_act' in disc_result and 'all_probabilities' in disc_result['dialogue_act']:
            da_probs = disc_result['dialogue_act']['all_probabilities']

            # 基于实证统计的高风险 DA 清单（按 da_name 精确匹配）
            strong_high_risk = {
                "Providing Evidence/Reasoning",  # 11
                "Making a Claim",  # 10
                "Relating to Another Student",  # 8
                "Revoicing",  # 4
                "Restating",  # 3
                "Press Reasoning",  # 6
            }

            medium_high_risk = {
                "Press Accuracy",  # 5
                "Asking for Info",  # 9
            }

            for act_name, prob in da_probs.items():
                if act_name in strong_high_risk:
                    predicted_risk_score += prob * 1.0
                elif act_name in medium_high_risk:
                    predicted_risk_score += prob * 0.6
                # 其他 DA 视为低风险，贡献 0

        # 决定使用哪个风险分（只有启用 Risk Routing 才计算）
        if self.enable_risk_routing:
            final_risk_score = oracle_risk_score if oracle_da_info else predicted_risk_score
        else:
            final_risk_score = 0.0  # 禁用风险路由时，风险分恒为0（不会触发拦截）

        # 3. 路由判断
        should_skip_llm = False
        skip_reason = ""

        if not self.use_generative:
            should_skip_llm = True
            skip_reason = "生成模型未启用"
        # 【保守版新增】若启用了风险路由, 判别模型预测 Irrelevant 且风险分很低,
        # 直接走快速通道，避免把明显的课堂管理/程序性话语送去 LLM
        elif self.enable_risk_routing and disc_label == 'Irrelevant' and final_risk_score <= 0.25:
            should_skip_llm = True
            skip_reason = f"Irrelevant 低风险快速通道 (Risk={final_risk_score:.2f})"
            self.fast_pass_by_irrelevant += 1
        elif disc_label == 'Irrelevant' and disc_confidence >= self.cascade_threshold_irrelevant:
            # 这里的 0.25 是你想要的温和阈值
            if final_risk_score > 0.25:
                should_skip_llm = False  # 有风险，不跳过
            else:
                should_skip_llm = True
                skip_reason = f"Irrelevant 快速通道 (Risk={final_risk_score:.2f})"
                self.fast_pass_by_irrelevant += 1
        elif disc_label != 'Irrelevant' and disc_confidence >= self.cascade_threshold:
            should_skip_llm = True
            skip_reason = f"高置信度快速通道 ({disc_confidence:.4f})"
            self.fast_pass_by_high_conf += 1

        # =========================================================
        # 4. 执行路径分支 (核心修复点！！！)
        # =========================================================

        # --- 分支 A: 快速通道 (Fast Pass) ---
        if should_skip_llm:
            self.fast_pass_total += 1  # 更新计数器

            # 构造决策信息
            decision_info = {
                'discriminative_label': disc_label,
                'discriminative_confidence': disc_confidence,
                'generative_label': None,
                'strategy': 'cascade_fast_pass',
                'reason': skip_reason,
                'final_decision': disc_label,  # 【关键】最终结果直接信判别模型
                'risk_score': final_risk_score  # 新增：记录风险分数
            }

            # 【绝对关键】直接 Return！不要往下走去调 make_decision！
            return {
                'final_label': disc_label,
                'discriminative_result': disc_result,
                'generative_result': None,
                'decision_info': decision_info
            }

        # --- 分支 B: 慢速通道 (Slow Path) ---
        # 只有没 Return 的才会走到这里，此时一定调用 LLM
        self.api_calls_total += 1

        # 若本样本有 kb_hint，且走到了 LLM，则计入“实际使用知识库”的样本数
        if kb_hint:
            self.kb_hint_llm_count += 1

        # 为了恢复旧版 LLM Prompt 行为：统一使用 v1 模板（软提示），
        # 仅向 LLM 传递对话行为 act_hint，不再注入知识库规则。
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

        # 联合决策 (gen_result 一定有值，放心调用)
        final_label, decision_info = self.decision_maker.make_decision(
            disc_result, gen_result, return_details=True, risk_score=final_risk_score
        )

        # 在决策信息中添加风险分数
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
            use_oracle_da: bool = False,  # <--- [新增参数]
            act_tag_column: str = 'Act Tag',  # <--- [新增参数] 默认为 'Act Tag'
            context_window: int = 5
    ):
        """
        批量分类CSV文件 (带成本统计)
        """
        print(f"\n读取数据: {input_csv}")
        df = pd.read_csv(input_csv)
        print(f"样本数: {len(df)}")

        if use_oracle_da and act_tag_column not in df.columns:
            print(f"⚠️ [警告] 数据中未找到列 '{act_tag_column}'，无法使用 Oracle 模式！")
            use_oracle_da = False

        # 重置计数器
        self.api_calls_total = 0
        self.fast_pass_total = 0
        self.fast_pass_by_irrelevant = 0
        self.fast_pass_by_high_conf = 0

        # 如果存在知识库且数据中有对话行为列，则预先构建用于匹配的 DA 序列特征
        # ⚠️ 当前设计：仅在 Oracle 模式下真正启用知识库提示
        kb_available = (
            self.knowledge_base is not None
            and act_tag_column in df.columns
            and use_oracle_da
        )
        # 每次运行前重置知识库使用计数
        self.kb_hint_match_count = 0
        self.kb_hint_llm_count = 0

        # 【知识库使用策略】仅对高置信、对内容类有帮助的规则启用提示
        # 可以根据后续实验调整阈值
        kb_target_labels = {"New", "Strengthened", "Weakened", "Adopted", "Refuted"}
        kb_min_lift = 2.0   # 至少提升 2 倍
        kb_min_prob = 0.5   # 条件概率至少 50%

        if kb_available:
            # 清洗 Act Tag，保持与 mine_knowledge_base.py 一致
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
                # 若没有 class 列，则退化为全局顺序
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

            # 1) 优先匹配 3-gram
            if isinstance(prev_act, str) and isinstance(prev_prev_act, str):
                key3 = f"{prev_prev_act} -> {prev_act} -> {act_curr}"
                entry3 = kb.get('sequence_3gram', {}).get(key3, None)
                if is_valid_entry(entry3):
                    self.kb_hint_match_count += 1
                    return entry3.get('description', None)

            # 2) 其次匹配 2-gram
            if isinstance(prev_act, str):
                key2 = f"{prev_act} -> {act_curr}"
                entry2 = kb.get('sequence_2gram', {}).get(key2, None)
                if is_valid_entry(entry2):
                    self.kb_hint_match_count += 1
                    return entry2.get('description', None)

            # 3) 再看交互模式
            #    - Teacher -> Student: 使用键 "T->S:{prev_da}"
            #    - Student -> Student: 使用键 "S->S:{prev_da}"
            if isinstance(prev_act, str) and speaker != 'T' and isinstance(prev_speaker, str):
                if prev_speaker == 'T':
                    interaction_key = f"T->S:{prev_act}"
                else:
                    interaction_key = f"S->S:{prev_act}"

                entry_int = kb.get('interaction', {}).get(interaction_key, None)
                if is_valid_entry(entry_int):
                    self.kb_hint_match_count += 1
                    return entry_int.get('description', None)

            # 4) 最后回退到单句先验
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

            # =========================================================
            # [新增] 读取人工标注数据
            # =========================================================
            oracle_da_info = None
            if use_oracle_da:
                # 读取 Act Tag (例如 "8 - press for accuracy")
                raw_tag = str(row.get(act_tag_column, ""))
                # 转小写以匹配规则 (规则库使用 'press', 'claim', 'evidence' 等小写词)
                oracle_da_info = {'label': raw_tag.lower()}

            # 知识库提示 (仅在 Oracle + KB 可用时启用)
            kb_hint = build_kb_hint(row) if kb_available else None

            # 分类（计数器在classify_single内部自动更新）
            result = self.classify_single(
                sentence, speaker, context_history, previous_speaker,
                icl_examples=None,
                oracle_da_info=oracle_da_info,
                kb_hint=kb_hint
            )

            # 记录结果
            result_row = {
                'index': idx,
                'Speaker': speaker,
                'Sentence': sentence,
                'final_label': result['final_label'],
                'disc_label': result['discriminative_result']['predicted_label'],
                'disc_confidence': result['discriminative_result']['confidence_score'],
                # 记录一下是否用了 Oracle
                'used_oracle_da': oracle_da_info['label'] if oracle_da_info else 'No',
                'strategy': result['decision_info'].get('strategy', 'unknown')
            }

            # 新增：提取 DA 预测信息
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

            # 新增：记录风险分数
            result_row['risk_score'] = result['decision_info'].get('risk_score', 0.0)

            # 安全处理生成模型结果 (可能为 None)
            if result.get('generative_result'):
                result_row['gen_label'] = result['generative_result']['predicted_label']
            else:
                result_row['gen_label'] = None  # 显式记录为空

            if result['decision_info']:
                result_row['strategy'] = result['decision_info']['strategy']
                result_row['reason'] = result['decision_info']['reason']

            if label_column in df.columns:
                true_label_id = row[label_column]
                result_row['true_label_id'] = true_label_id
                label_names = ['Irrelevant', 'New', 'Strengthened', 'Weakened', 'Adopted', 'Refuted']
                # 处理可能的数字或字符串标签
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

            # 更新上下文
            context_history.append({
                'speaker': speaker,
                'sentence': sentence,
                'label': result['final_label']
            })

            if len(context_history) > context_window:
                context_history.pop(0)
            previous_speaker = speaker

            # 进度显示 (包含实时成本统计)
            if (idx + 1) % 20 == 0:
                savings_rate = (self.fast_pass_total / (idx + 1)) * 100 if (idx + 1) > 0 else 0
                print(
                    f"  进度: {idx + 1}/{len(df)} | API调用: {self.api_calls_total} | 快速通道: {self.fast_pass_total} (Irrelevant: {self.fast_pass_by_irrelevant}) | 节省率: {savings_rate:.1f}%")

        # 保存结果
        result_df = pd.DataFrame(results)
        result_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存: {output_csv}")

        # 打印详细的成本效益分析
        total_samples = len(df)
        savings_rate = (self.fast_pass_total / total_samples * 100) if total_samples > 0 else 0

        print("\n" + "=" * 80)
        print("💰 级联路由成本效益分析 (Cascade Routing Cost Analysis)")
        print("=" * 80)
        print(f"\n【总体统计】")
        print(f"  总样本数:              {total_samples}")
        print(f"  API调用次数 (花钱💸):  {self.api_calls_total:4d} ({self.api_calls_total / total_samples * 100:5.1f}%)")
        print(f"  快速通道次数 (省钱💰): {self.fast_pass_total:4d} ({savings_rate:5.1f}%)")

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

            # 分别统计快速通道和慢速通道的准确率
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


# =============================================================================
# 6. 示例用法
# =============================================================================

def demo():
    """演示混合模型的使用"""

    # 配置
    model_dir = "../discriminative_model_outputs"  # 判别模型目录
    test_csv = "../dataset_split_result_v2/test.csv"  # 测试数据
    output_csv = "../hybrid_predictions.csv"  # 输出文件

    # 初始化混合分类器
    classifier = HybridOpinionClassifier(
        discriminative_model_dir=model_dir,
        use_generative=True,  # 启用生成模型
        use_knn_icl=True,  # 启用kNN-ICL
        decision_threshold=0.7  # 置信度阈值
    )

    # 示例1: 单句分类
    print("\n" + "=" * 80)
    print("示例1: 单句分类")
    print("=" * 80)

    sentence = "I would like someone to tell me where you would place one."
    speaker = "T"
    context = [
        {'speaker': 'T', 'sentence': 'Today we are talking about fractions.', 'label': 'New'},
        {'speaker': 'S', 'sentence': 'I think we should use the number line.', 'label': 'New'}
    ]

    result = classifier.classify_single(sentence, speaker, context)

    print(f"\n输入: [{speaker}] {sentence}")
    print(f"最终预测: {result['final_label']}")
    print(f"判别模型: {result['discriminative_result']['predicted_label']} "
          f"(置信度: {result['discriminative_result']['confidence_score']:.4f})")
    if result['generative_result']:
        print(f"生成模型: {result['generative_result']['predicted_label']}")
    print(f"决策策略: {result['decision_info']['strategy']}")
    print(f"决策理由: {result['decision_info']['reason']}")

    # 示例2: 批量分类
    print("\n" + "=" * 80)
    print("示例2: 批量分类文件")
    print("=" * 80)

    if os.path.exists(test_csv):
        classifier.classify_file(
            input_csv=test_csv,
            output_csv=output_csv,
            text_column='Sentence',
            speaker_column='Speaker',
            label_column='label',
            context_window=5
        )
    else:
        print(f"测试文件不存在: {test_csv}")


if __name__ == '__main__':
    demo()