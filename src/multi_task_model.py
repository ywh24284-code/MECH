"""
Multi-task Dialogue Model: Jointly predict Dialogue Act (DA) and Opinion Evolution (OE).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig


class MultiTaskDialogueModel(nn.Module):
    """
    Multi-task learning model.

    Architecture:
      Shared Encoder (DeBERTa) -> Shared Layer -> Branch 1: Dialogue Act Classification
                                                -> Branch 2: Opinion Evolution Classification
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
        
        # Load pretrained encoder
        self.encoder = AutoModel.from_pretrained(model_path)
        encoder_hidden_size = self.encoder.config.hidden_size
        
        self.num_dialogue_acts = num_dialogue_acts
        self.num_opinion_classes = num_opinion_classes
        self.use_task_specific_layers = use_task_specific_layers
        self.use_act_logits_in_opinion = use_act_logits_in_opinion
        
        # Shared feature extraction layer
        self.shared_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(encoder_hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout / 2)
        )
        
        # Task 1: Dialogue Act classification head
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
        
        # Task 2: Opinion Evolution classification head
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
    Multi-task joint loss function.
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
        """Compute joint loss."""
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


