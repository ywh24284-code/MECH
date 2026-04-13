"""
多任务对话模型: 同时预测对话行为(Dialogue Act)和观点演化(Opinion Evolution)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # 类别权重
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)  
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
        
        # 任务1: 对话行为分类头
        if use_task_specific_layers:
            self.dialogue_act_head = nn.Sequential(
                nn.Linear(hidden_size, 512), 
                nn.GELU(),
                nn.LayerNorm(512),
                nn.Dropout(dropout / 3),
                nn.Linear(512, 256), 
                nn.GELU(),
                nn.LayerNorm(256),
                nn.Dropout(dropout / 3),
                nn.Linear(256, num_dialogue_acts)
            )
        else:
            self.dialogue_act_head = nn.Linear(hidden_size, num_dialogue_acts)
        
        # 任务2: 观点演化分类头
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

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :] 
        

        shared_features = self.shared_layer(cls_embedding) 
        

        dialogue_act_logits = self.dialogue_act_head(shared_features)

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
        
  
        if return_embedding:
            result['shared_embedding'] = shared_features
            result['cls_embedding'] = cls_embedding
        
    
        if return_confidence:
  
            act_probs = F.softmax(dialogue_act_logits, dim=-1)
            act_confidence, act_predicted = torch.max(act_probs, dim=-1)
            act_top_k_probs, act_top_k_indices = torch.topk(act_probs, k=min(3, self.num_dialogue_acts), dim=-1)
   
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
    """

    def __init__(
            self,
            dialogue_act_weight: float = 0.5,
            opinion_weight: float = 0.5,
            dialogue_act_class_weights: torch.Tensor = None,
            opinion_class_weights: torch.Tensor = None, 
            label_smoothing: float = 0.1,
            use_opinion_reweighting: bool = True 
    ):
        super().__init__()
        self.dialogue_act_weight = dialogue_act_weight
        self.opinion_weight = opinion_weight
        self.label_smoothing = label_smoothing

   
        if opinion_class_weights is not None:
            self.opinion_class_weights = opinion_class_weights
        elif use_opinion_reweighting:
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
        device = dialogue_act_logits.device

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

        op_weight = None
        if self.opinion_class_weights is not None:
            op_weight = self.opinion_class_weights.to(device)

        opinion_loss = F.cross_entropy(
            opinion_logits,
            opinion_labels,
            weight=op_weight,
            label_smoothing=self.label_smoothing
        )

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
    print("测试多任务模型...")
    model = MultiTaskDialogueModel(
        model_path='microsoft/deberta-v3-base',
        num_dialogue_acts=9,
        num_opinion_classes=6
    )
    
    batch_size = 4
    seq_length = 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    
    outputs = model(input_ids, attention_mask, return_confidence=True, return_embedding=True)
    
    print(f"✓ 对话行为logits shape: {outputs['dialogue_act_logits'].shape}")
    print(f"✓ 观点演化logits shape: {outputs['opinion_logits'].shape}")
    print(f"✓ 共享embedding shape: {outputs['shared_embedding'].shape}")
    
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
