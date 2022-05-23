# coding: utf-8

"""
FeatureExtractorモジュールの定義
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchinfo import summary


class FeatureExtractor(nn.Module):
    """入力画像から特徴量を抽出しABNにわたすモジュール
    想定される入力サイズ：(224, 224, 3)
    """

    def __init__(self, pretrained=True):
        super().__init__()
        self.resnet18 = models.resnet50(pretrained=pretrained)
        self.feauture_extractor = nn.ModuleList(list(self.resnet18.children())[:-3])

    def forward(self, x):
        """ResNet18の前半部分を切り出して(14,14,256)の特徴量を抽出する
        Args:
            x (tensor): size(224, 224, 3)
        Return:
            x (tensor): size(14, 14, 256)
        """
        for model in self.feauture_extractor:
            x = model(x)

        return x


if __name__ == "__main__":
    # torch summayでモデルを確認
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = FeatureExtractor().to(device)
    print(feature_extractor)
    BATCH_SIZE = 64
    TEST_INPUT = (BATCH_SIZE, 3, 224, 224)
    summary(feature_extractor, [TEST_INPUT])