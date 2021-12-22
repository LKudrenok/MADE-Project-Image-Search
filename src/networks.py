import torch
from torch import nn
from pretrainedmodels import inceptionresnetv2


class InceptionResNetV2(nn.Module):
    def __init__(self, device: str):
        super().__init__()
        self.net = inceptionresnetv2(pretrained='imagenet', num_classes=1000)
        self.net.avgpool_1a = nn.AvgPool2d(5, count_include_pad=False)
        self.embedding_size = 1536
        self.device = device
        self.net.to(device)
        self.net.eval()

    def forward(self, x):
        with torch.no_grad():
            x = self.net.features(x)        # [batch, 1536, 5, 5]
            x = self.net.avgpool_1a(x)      # [batch, 1536, 1, 1]
            x = x.view(x.size(0), -1)       # [batch, 1536]
        return x


class Model(nn.Module):
    def __init__(self, device: str):
        super().__init__()
        self.encoder = InceptionResNetV2(device)
        self.embedding_size = self.encoder.embedding_size
        self.device = device

    def forward(self, x):
        x = self.encoder(x)
        return x
