import torch.nn as nn
import torchvision


class RNN50(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # don't freeze params cause we want to fine-tune
        mod = torchvision.models.resnet50(pretrained=True)
        # take only the feature portions of the model (no avg pool or classification)
        self.vs = vocab_size
        #overwrite the fc layer
        self.mod.fc = nn.Linear(2048, self.vs)

    def forward(self, x):
        out_encoder = self.mod
        out_decoder = nn.Sequential(

        )

        encoded = out_encoder(x)
        decoded = out_decoder(encoded)

        score = self.classifier(decoded)
        # size=(N, n_class, x.H/1, x.W/1)
        return score