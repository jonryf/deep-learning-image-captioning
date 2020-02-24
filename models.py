# PyTorch
import torch
import torch.nn as nn
import torchvision
from torch.nn.utils.rnn import pack_padded_sequence

from utils import get_device


class Encoder(nn.Module):
    def __init__(self, linear_layer_size=1000):
        """

        :param linear_layer_size:
        """
        super(Encoder, self).__init__()
        self.mod = torchvision.models.resnet50(pretrained=True)
        self.mod.fc = nn.Linear(2048, linear_layer_size)
        for param in self.parameters():
            if not isinstance(param, nn.Linear):
                param.requires_grad = False
        self.batch_norm = nn.BatchNorm1d(linear_layer_size, momentum=0.01)

    def forward(self, images):
        features = self.mod(images)
        features = self.batch_norm(features)
        return features


class Decoder(nn.Module):

    def __init__(self, encoder, input_size, hidden_size, embedding_size, LSTM=True):
        """
        Initialize decoder

        :param input_size:
        :param hidden_size:
        :param LSTM:
        """
        super(Decoder, self).__init__()
        self.encoder = encoder
        self.inputs_size = input_size
        self.hidden_size = hidden_size
        self.LSTM = LSTM
        self.embedding = nn.Embedding(input_size, embedding_size)

        # Model layout
        if LSTM:
            self.rnn_cell = nn.LSTM(input_size, hidden_size)
        else:
            self.rnn_cell = nn.RNN(input_size, hidden_size)

        self.softmax_function = nn.Softmax()
        self.hidden = None
        self.cell = None

        self.training_losses = []
        self.validation_losses = []

    def init_state(self, encoder_output):
        """
        Reset the state of the decoder
        """
        computing_device = get_device()
        self.hidden = encoder_output.to(computing_device)
        if self.LSTM:
            self.cell = torch.zeros(1, 1, self.hidden_size).to(computing_device)

    def forward(self, features, captions, lengths):
        """

        :param image:
        :param caption:
        :return:
        """
        embedded = self.embedding(captions)
        embedded = torch.cat((features, embedded), 1)
        combined = pack_padded_sequence(embedded, lengths, batch_first=True)
        self.hidden, self.cell = self.rnn_cell(combined)
        return self.linear(self.hidden)


        # image, caption = data
        #
        # features = self.encoder.mod(image)
        # if self.LSTM:
        #     self.hidden, self.cell = self.rnn_cell(features, (self.hidden, self.cell))
        # else:
        #     self.hidden = self.rnn_cell(features, self.hidden)
        #
        # outputs = [self.softmax_function(self.hidden)]
        # for feature in caption:
        #     if self.LSTM:
        #         self.hidden, self.cell = self.rnn_cell(feature, (self.hidden, self.cell))
        #     else:
        #         self.hidden = self.rnn_cell(feature, self.hidden)
        #     outputs.append(self.softmax_function(self.hidden))
        #
        # return outputs
