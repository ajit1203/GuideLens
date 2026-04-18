import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights
from transformers import AutoModel


class TrustworthyVQAModel(nn.Module):
    def __init__(
        self,
        num_answers: int,
        text_model_name: str = "distilbert-base-uncased",
        hidden_dim: int = 256,
        dropout: float = 0.2,
        freeze_vision: bool = True,
        freeze_text: bool = True,
        use_pretrained_vision: bool = True,
        local_files_only: bool = False,
    ):
        super().__init__()

        if use_pretrained_vision:
            self.vision_encoder = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            self.vision_encoder = models.resnet18(weights=None)

        vision_dim = self.vision_encoder.fc.in_features
        self.vision_encoder.fc = nn.Identity()

        self.text_encoder = AutoModel.from_pretrained(
            text_model_name,
            local_files_only=local_files_only,
        )
        text_dim = self.text_encoder.config.hidden_size

        if freeze_vision:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

        if freeze_text:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        fusion_dim = vision_dim + text_dim

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.answer_head = nn.Linear(hidden_dim, num_answers)
        self.answerability_head = nn.Linear(hidden_dim, 2)

    def forward(self, image, input_ids, attention_mask):
        image_features = self.vision_encoder(image)

        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        text_features = text_outputs.last_hidden_state[:, 0, :]

        fused = torch.cat([image_features, text_features], dim=1)
        fused = self.fusion(fused)

        answer_logits = self.answer_head(fused)
        answerability_logits = self.answerability_head(fused)

        return {
            "answer_logits": answer_logits,
            "answerability_logits": answerability_logits,
        }