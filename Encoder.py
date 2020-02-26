import torchvision
from torch import nn as nn


class Encoder(nn.Module):
    def __init__(self, linear_layer_size=1000):
        """
        CNN for the images

        :param linear_layer_size:
        """
        super(Encoder, self).__init__()
        self.mod = torchvision.models.resnet50(pretrained=True)
        for param in self.mod.parameters():
            param.requires_grad = False

        self.mod.fc = nn.Linear(2048, linear_layer_size)

        self.batch_norm = nn.BatchNorm1d(linear_layer_size, momentum=0.01)

    def forward(self, images):
        """
        Forward pass into the encoder

        :param images:
        :return: feature vector
        """
        features = self.mod(images)
        features = self.batch_norm(features.reshape(features.size(0), -1))
        return features