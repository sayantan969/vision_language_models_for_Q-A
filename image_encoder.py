
import torch
import torch.nn as nn
import torchvision.models as models

class ImageEncoder(nn.Module):
    """ResNet-based image encoder that outputs a 512-D vector."""
    def __init__(self, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        resnet = models.resnet34(pretrained=pretrained)
        modules = list(resnet.children())[:-1]
        self.backbone = nn.Sequential(*modules)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: (B, 3, H, W)
        returns: (B, 512)
        """
        x = self.backbone(images)  # (B, 512, 1, 1)
        x = x.view(x.size(0), -1)
        return x

