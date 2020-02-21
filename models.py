# PyTorch
import torch
import torch.nn as nn

from utils import get_device


class Encoder(nn.Module):
    def __init__(self, linear_layer_size=1000):
        super(Encoder).__init__()
        # don't freeze params cause we want to fine-tune
        mod = torchvision.models.resnet50(pretrained=True)
        # take only the feature portions of the model (no avg pool or classification)
        #overwrite the fc layer
        self.mod.fc = nn.Linear(2048, linear_layer_size)


class Decoder(nn.Module):

    def __init__(self, encoder, input_size, hidden_size, LSTM=True):
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

        # Model layout
        if LSTM:
            self.rnn_cell = nn.LSTMCell(input_size, hidden_size)
        else:
            self.rnn_cell = nn.RNNCell(input_size, hidden_size)

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

    def forward(self, image, caption):
        """

        :param image:
        :param caption:
        :return:
        """
        feature = self.encoder.mod.forward(image)
        self.cell = torch.zeros(1, 1, self.hidden_size).to(get_device())
        self.hidden = feature.to(get_device())

        outputs = []
        for feature in caption:
            if self.LSTM:
                self.hidden, self.cell = self.rnn_cell(feature, (self.hidden, self.cell))
            else:
                self.hidden = self.rnn_cell(feature, self.hidden)
            outputs.append(self.softmax_function(self.hidden))

        return outputs
