import torch
from torch import nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from utils import sampleFromDistribution, softmax


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
            self.rnn_cell = nn.LSTM(input_size, hidden_size, batch_first=True)
        else:
            self.rnn_cell = nn.RNN(input_size, hidden_size, batch_first=True)

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
        return self.linear(self.hidden.data)

    def predict(self, features, max_length):
        captions = []
        cell_state = None
        for i in range(max_length):
            features, cell_state = self.rnn_cell(features,  cell_state)

            # decode embedding
            features = self.linear(features.squeeze(1))  # batch vs vocab

            # select value from distribution with temperature
            features = features.max(1)[1]  # sampleFromDistribution(softmax(features), True)

            # save caption words
            captions.append(features)

            # embed word
            features = self.embedding(features)

        return torch.cat(captions).squeeze(1)
