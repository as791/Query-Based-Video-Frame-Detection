import torch
import torch.nn as nn
import loralib as lora
from transformers import BertModel
from torchvision import models


class LoRAImageTextRetrievalModel(nn.Module):
    """
    Image-text retrieval model with LoRA adapters on BERT.
    Image branch: ResNet-50 visual encoder.
    Text branch:  BERT with LoRA on query/value projection.
    Both branches project to a shared 256-d embedding space.
    """

    EMBED_DIM = 256

    def __init__(self):
        super().__init__()

        # Visual encoder
        resnet = models.resnet50(weights=None)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.image_proj = nn.Linear(2048, self.EMBED_DIM)

        # Text encoder with LoRA
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        for name, module in self.bert.named_modules():
            if isinstance(module, nn.Linear) and ("query" in name or "value" in name):
                lora.mark_only_lora_as_trainable(self.bert)
                break

        self.text_proj = nn.Linear(768, self.EMBED_DIM)

    def forward(self, images, input_ids, attention_mask):
        # Image features
        img_feats = self.image_encoder(images).squeeze(-1).squeeze(-1)
        img_feats = self.image_proj(img_feats)
        img_feats = nn.functional.normalize(img_feats, dim=-1)

        # Text features
        text_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_feats = self.text_proj(text_out.pooler_output)
        text_feats = nn.functional.normalize(text_feats, dim=-1)

        return img_feats, text_feats
