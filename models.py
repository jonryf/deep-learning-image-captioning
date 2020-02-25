# PyTorch
import torch
import torch.nn as nn
import torchvision
from torch.nn.utils.rnn import pack_padded_sequence


class Encoder(nn.Module):
    def __init__(self, linear_layer_size=1000):
        """
        CNN for the images

        :param linear_layer_size:
        """
        super(Encoder, self).__init__()
        self.mod = torchvision.models.resnet50(pretrained=True)
        self.mod.fc = nn.Linear(2048, linear_layer_size)
        for param in self.parameters():
            if not isinstance(param, nn.Linear):
                param.requires_grad = False
            print(param)
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


class Decoder(nn.Module):

    def __init__(self, input_size, hidden_size, vocabulary_size, lstm=True):
        """
        Initialize decoder

        Decode the image feature from the encoder to captions

        :param input_size:
        :param hidden_size:
        :param lstm: Vanilla RNN or LSTM
        """
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.LSTM = lstm
        self.embedding = nn.Embedding(vocabulary_size, input_size)

        if lstm:
            self.rnn_cell = nn.LSTM(input_size, hidden_size)
        else:
            self.rnn_cell = nn.RNN(input_size, hidden_size)

        # linear decoding of the captions
        self.linear = nn.Linear(hidden_size, vocabulary_size)

        # states
        self.hidden = None
        self.cell = None

    def forward(self, features, captions, lengths):
        """
        Forward pass data into the decoder

        :param features: feature vectors
        :param captions: captions
        :param lengths: size of captions
        :return: caption prediction
        """
        # One-hot encode and encode to a smaller vector space using a linear layer
        embedded = self.embedding(captions)
        embedded = torch.cat((features.unsqueeze(1), embedded), 1)

        # capsule the already padded sequences, improve performance when passing in to RNN
        combined = pack_padded_sequence(embedded, lengths, batch_first=True)

        # Run the LSTM,
        self.hidden, self.cell = self.rnn_cell(combined)

        # Decode the encoded captions
        return self.linear(self.hidden[0])
