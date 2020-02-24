import torch

from torch.nn import NLLLoss
from torch.optim import Adam

from settings import EPOCHS
from torch.nn.utils.rnn import pack_padded_sequence


class Runner:

    def __init__(self, encoder, decoder, train_dataset, val_dataset, test_dataset):
        self.encoder = encoder
        self.decoder = decoder
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.criterion = NLLLoss()

        params = list(self.decoder.parameters()) + list(self.encoder.parameters())
        self.optimizer = Adam(params)

    def train(self):
        for epoch in range(EPOCHS):
            train_loss = self.pass_data(self.train_dataset, True)
            val_loss = self.pass_data(self.val_dataset, False)

            print("Epoch: {}  -  training loss: {}, validation loss: {}".format(epoch, train_loss, val_loss))

    def val(self):
        with torch.no_grad():
            return self.pass_data(self.val_dataset, False)

    def test(self):
        with torch.no_grad():
            return self.pass_data(self.test_dataset, False)

    def pass_data(self, dataset, backward):
        if backward:
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()
        loss = 0

        for minibatch, (images, captions, lengths) in enumerate(dataset):
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            # forward
            encoded = self.encoder(images)
            predicted = self.decoder(encoded)

            batch_loss = self.criterion(predicted, targets.long())

            # backward
            if backward:
                self.encoder.zero_grad()
                self.decoder.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

            loss += batch_loss.item()
        loss /= minibatch
        return loss
