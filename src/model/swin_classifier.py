

import timm
import torch.nn as nn


class SwinDriverClassifier(nn.Module):
    """Pretrained Swin-Base backbone + binary classification head."""

    def __init__(
        self,
        model_name: str = "swin_base_patch4_window7_224",
        pretrained: bool = True,
        num_classes: int = 2,
    ):
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes
        )

        if not hasattr(self.backbone, "head"):
            raise AttributeError(
                f"'{model_name}' modelinde 'head' attribute'u bulunamadı. "
                f"Checkpoint ayrımı (backbone/head) bu attribute'a dayanıyor, "
                f"farklı bir model mimarisi kullanılıyorsa get_backbone_state_dict/"
                f"get_head_state_dict fonksiyonları güncellenmeli."
            )

    def forward(self, x):
        return self.backbone(x)

    def get_backbone_state_dict(self) -> dict:
        """Sadece backbone ağırlıklarını döner (head hariç) — checkpoint ayrımı için."""
        return {
            k: v for k, v in self.backbone.state_dict().items()
            if not k.startswith("head.")
        }

    def get_head_state_dict(self) -> dict:
        """Sadece classification head ağırlıklarını döner."""
        return {
            k: v for k, v in self.backbone.state_dict().items()
            if k.startswith("head.")
        }

    def get_backbone_parameters(self):
        """Differential learning rate için backbone parametrelerini döner (head hariç)."""
        return [
            p for n, p in self.backbone.named_parameters()
            if not n.startswith("head.")
        ]

    def get_head_parameters(self):
        """Differential learning rate için head parametrelerini döner."""
        return [
            p for n, p in self.backbone.named_parameters()
            if n.startswith("head.")
        ]


if __name__ == "__main__":
    # Hızlı manuel test / sanity check için (indirme olmadan, pretrained=False)
    import torch

    model = SwinDriverClassifier(pretrained=False, num_classes=2)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"Çıktı şekli: {tuple(output.shape)}")

    backbone_sd = model.get_backbone_state_dict()
    head_sd = model.get_head_state_dict()
    print(f"Backbone parametre sayısı (tensor): {len(backbone_sd)}")
    print(f"Head parametre sayısı (tensor): {len(head_sd)}")
    print(f"Head anahtarları: {list(head_sd.keys())}")
