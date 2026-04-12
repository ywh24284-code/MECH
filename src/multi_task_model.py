"""
多任务对话模型: 同时预测对话行为(Dialogue Act)和观点演化(Opinion Evolution)

用法:
  from multi_task_model import MultiTaskDialogueModel
  
  model = MultiTaskDialogueModel(
      model_path='deberta-v3-base',
      num_dialogue_acts=9,
      num_opinion_classes=6
  )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig


# =============================================================================
# 新增: Focal Loss 用于处理难分类样本
# =============================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # 类别权重
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 计算标准交叉熵损失 (不进行归约，以便后续加权)
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)  # 计算预测概率 pt
        # 计算 Focal Loss: (1 - pt)^gamma * log(pt)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss.sum()







class MultiTaskDialogueModel(nn.Module):
    """
    多任务学习模型
    
    架构:
      共享编码器(DeBERTa) → 共享层 → 分支1: 对话行为分类
                                      → 分支2: 观点演化分类
    
    优势:
      1. 对话行为和观点演化共享底层语义理解
      2. 对话行为作为中间表示,辅助观点演化分类
      3. 多任务正则化,提升泛化能力
    """
    
    def __init__(
        self, 
        model_path: str,
        num_dialogue_acts: int = 9,
        num_opinion_classes: int = 6,
        hidden_size: int = 512,
        dropout: float = 0.5,
        use_task_specific_layers: bool = True,
        use_act_logits_in_opinion: bool = False
    ):
        """
        Args:
            model_path: 预训练模型路径(如 'deberta-v3-base')
            num_dialogue_acts: 对话行为类别数
            num_opinion_classes: 观点演化类别数
            hidden_size: 共享层隐藏层大小
            dropout: Dropout比例
            use_task_specific_layers: 是否为每个任务添加专用层
        """
        super().__init__()
        
        # 加载预训练编码器
        self.encoder = AutoModel.from_pretrained(model_path)
        encoder_hidden_size = self.encoder.config.hidden_size
        
        self.num_dialogue_acts = num_dialogue_acts
        self.num_opinion_classes = num_opinion_classes
        self.use_task_specific_layers = use_task_specific_layers
        self.use_act_logits_in_opinion = use_act_logits_in_opinion
        
        # 共享特征提取层
        self.shared_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(encoder_hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout / 2)
        )
        
        # 任务1: 对话行为分类头 (增强深度版本)
        if use_task_specific_layers:
            self.dialogue_act_head = nn.Sequential(
                nn.Linear(hidden_size, 512),  # 增加到512
                nn.GELU(),
                nn.LayerNorm(512),
                nn.Dropout(dropout / 3),
                nn.Linear(512, 256),  # 新增一层
                nn.GELU(),
                nn.LayerNorm(256),
                nn.Dropout(dropout / 3),
                nn.Linear(256, num_dialogue_acts)
            )
        else:
            self.dialogue_act_head = nn.Linear(hidden_size, num_dialogue_acts)
        
        # 任务2: 观点演化分类头
        # 如果启用 use_act_logits_in_opinion，则在观点头中显式接入对话行为预测（softmax 概率）
        opinion_input_dim = hidden_size + (num_dialogue_acts if use_act_logits_in_opinion else 0)
        if use_task_specific_layers:
            self.opinion_head = nn.Sequential(
                nn.Linear(opinion_input_dim, 256),
                nn.GELU(),
                nn.LayerNorm(256),
                nn.Dropout(dropout / 3),
                nn.Linear(256, num_opinion_classes)
            )
        else:
            self.opinion_head = nn.Linear(opinion_input_dim, num_opinion_classes)
        
    def forward(
        self, 
        input_ids, 
        attention_mask, 
        return_embedding: bool = False,
        return_confidence: bool = False
    ):
        """
        前向传播
        
        Args:
            input_ids: 输入token ids
            attention_mask: 注意力掩码
            return_embedding: 是否返回共享embedding(用于kNN检索)
            return_confidence: 是否返回置信度信息
            
        Returns:
            dict包含:
                - dialogue_act_logits: 对话行为logits
                - opinion_logits: 观点演化logits
                - shared_embedding: 共享特征(可选)
                - confidence_info: 置信度信息(可选)
        """
        # 编码器前向传播
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [batch, hidden_size]
        
        # 共享特征提取
        shared_features = self.shared_layer(cls_embedding)  # [batch, 512]
        
        # 任务1: 对话行为分类
        dialogue_act_logits = self.dialogue_act_head(shared_features)

        # 任务2: 观点演化分类
        # 如果启用 use_act_logits_in_opinion，则将对话行为的 softmax 概率作为额外特征拼接到共享表示中
        if self.use_act_logits_in_opinion:
            act_probs_for_fusion = F.softmax(dialogue_act_logits, dim=-1)
            opinion_input = torch.cat([shared_features, act_probs_for_fusion], dim=-1)
        else:
            opinion_input = shared_features

        opinion_logits = self.opinion_head(opinion_input)
        
        result = {
            'dialogue_act_logits': dialogue_act_logits,
            'opinion_logits': opinion_logits
        }
        
        # 可选: 返回embedding
        if return_embedding:
            result['shared_embedding'] = shared_features
            result['cls_embedding'] = cls_embedding
        
        # 可选: 返回置信度信息
        if return_confidence:
            # 对话行为置信度
            act_probs = F.softmax(dialogue_act_logits, dim=-1)
            act_confidence, act_predicted = torch.max(act_probs, dim=-1)
            act_top_k_probs, act_top_k_indices = torch.topk(act_probs, k=min(3, self.num_dialogue_acts), dim=-1)
            
            # 观点演化置信度
            opinion_probs = F.softmax(opinion_logits, dim=-1)
            opinion_confidence, opinion_predicted = torch.max(opinion_probs, dim=-1)
            opinion_top_k_probs, opinion_top_k_indices = torch.topk(
                opinion_probs, k=self.num_opinion_classes, dim=-1
            )
            
            result['dialogue_act_confidence'] = {
                'probabilities': act_probs,
                'confidence_scores': act_confidence,
                'predicted_classes': act_predicted,
                'top_k_probs': act_top_k_probs,
                'top_k_indices': act_top_k_indices
            }
            
            result['opinion_confidence'] = {
                'probabilities': opinion_probs,
                'confidence_scores': opinion_confidence,
                'predicted_classes': opinion_predicted,
                'top_k_probs': opinion_top_k_probs,
                'top_k_indices': opinion_top_k_indices
            }
        
        return result


class MultiTaskLoss(nn.Module):
    """
    多任务联合损失函数
    支持手动传入类别权重
    """

    def __init__(
            self,
            dialogue_act_weight: float = 0.5,
            opinion_weight: float = 0.5,
            dialogue_act_class_weights: torch.Tensor = None,
            opinion_class_weights: torch.Tensor = None,  # 新增：允许接收外部传入的权重
            label_smoothing: float = 0.1,
            use_opinion_reweighting: bool = True  # 新增：允许接收开关参数
    ):
        super().__init__()
        self.dialogue_act_weight = dialogue_act_weight
        self.opinion_weight = opinion_weight
        self.label_smoothing = label_smoothing

        # 逻辑：如果外部传入了权重，优先使用；否则检查开关是否开启默认权重
        if opinion_class_weights is not None:
            # 使用传入的强力权重
            self.opinion_class_weights = opinion_class_weights
        elif use_opinion_reweighting:
            # 默认权重 (备用)
            self.opinion_class_weights = torch.tensor([
                0.8, 1.0, 1.2, 2.0, 1.5, 1.8
            ], dtype=torch.float32)
        else:
            self.opinion_class_weights = None

        self.dialogue_act_class_weights = dialogue_act_class_weights

    def forward(
            self,
            dialogue_act_logits: torch.Tensor,
            opinion_logits: torch.Tensor,
            dialogue_act_labels: torch.Tensor,
            opinion_labels: torch.Tensor
    ):
        """计算联合损失"""
        # 自动获取设备 (确保权重在GPU上)
        device = dialogue_act_logits.device

        # 计算对话行为损失
        da_weight = None
        if self.dialogue_act_class_weights is not None:
            da_weight = self.dialogue_act_class_weights.to(device)

        dialogue_act_loss = F.cross_entropy(
            dialogue_act_logits,
            dialogue_act_labels,
            weight=da_weight,
            label_smoothing=self.label_smoothing,
            ignore_index=-1
        )

        # 计算观点演化损失
        op_weight = None
        if self.opinion_class_weights is not None:
            op_weight = self.opinion_class_weights.to(device)

        opinion_loss = F.cross_entropy(
            opinion_logits,
            opinion_labels,
            weight=op_weight,
            label_smoothing=self.label_smoothing
        )

        # 加权组合
        total_loss = (
                self.dialogue_act_weight * dialogue_act_loss +
                self.opinion_weight * opinion_loss
        )

        return {
            'total_loss': total_loss,
            'dialogue_act_loss': dialogue_act_loss,
            'opinion_loss': opinion_loss
        }


def test_model():
    """测试多任务模型"""
    print("测试多任务模型...")
    
    # 创建模型
    model = MultiTaskDialogueModel(
        model_path='microsoft/deberta-v3-base',
        num_dialogue_acts=9,
        num_opinion_classes=6
    )
    
    # 创建假数据
    batch_size = 4
    seq_length = 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    
    # 前向传播
    outputs = model(input_ids, attention_mask, return_confidence=True, return_embedding=True)
    
    print(f"✓ 对话行为logits shape: {outputs['dialogue_act_logits'].shape}")
    print(f"✓ 观点演化logits shape: {outputs['opinion_logits'].shape}")
    print(f"✓ 共享embedding shape: {outputs['shared_embedding'].shape}")
    
    # 测试损失函数
    loss_fn = MultiTaskLoss(dialogue_act_weight=0.3, opinion_weight=0.7)
    dialogue_act_labels = torch.randint(0, 9, (batch_size,))
    opinion_labels = torch.randint(0, 6, (batch_size,))
    
    loss_dict = loss_fn(
        outputs['dialogue_act_logits'],
        outputs['opinion_logits'],
        dialogue_act_labels,
        opinion_labels
    )
    
    print(f"✓ Total loss: {loss_dict['total_loss'].item():.4f}")
    print(f"✓ Dialogue act loss: {loss_dict['dialogue_act_loss'].item():.4f}")
    print(f"✓ Opinion loss: {loss_dict['opinion_loss'].item():.4f}")
    
    print("\n模型参数统计:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")


if __name__ == '__main__':
    test_model()
