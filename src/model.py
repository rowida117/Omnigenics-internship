"""
Model Architecture
SkinLesionModel: dual-purpose EfficientNet-B0 based model.
    forward(x)        -> 3 class logits   (used during training)
    get_embedding(x)  -> 1280-D vector    (used for retrieval)
"""

from torch import nn
from torchvision import models  # type: ignore[import-not-found]


class SkinLesionModel(nn.Module):
    def __init__(self, backbone_name="efficientnet_b0", num_classes=3, pretrained=True):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.embedding_dim = 1280  # EfficientNet-B0's known output size

        if pretrained:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            base = models.efficientnet_b0(weights=weights)
        else:
            base = models.efficientnet_b0(weights=None)

        self.features = base.features
        self.avgpool = base.avgpool

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2), nn.Linear(self.embedding_dim, num_classes)
        )

    def get_embedding(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return x

    def forward(self, x):
        embedding = self.get_embedding(x)
        logits = self.classifier(embedding)
        return logits
