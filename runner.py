from torch.nn import NLLLoss
from torch.optim import Adam

from settings import EPOCHS
from torch.nn.utils.rnn import pack_padded_sequence


class Runner:

    def __init__(self, encoder, decoder, train_dataset):
        self.encoder = encoder
        self.decoder = decoder
        self.train_dataset = train_dataset

    def train(self):
        self.encoder.train()
        self.decoder.train()

        criterion = NLLLoss()

        params = list(self.decoder.parameters()) + list(self.encoder.parameters())
        optimizer = Adam(params)

        train_loss = 0

        for epoch in range(EPOCHS):
            for minibatch, (images, captions, lengths) in enumerate(self.train_dataset):
                targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

                # forward
                encoded = self.encoder(images)
                predicted = self.decoder(encoded)

                # backward
                batch_loss = criterion(predicted, targets.long())
                self.encoder.zero_grad()
                self.decoder.zero_grad()
                batch_loss.backward()
                optimizer.step()

    def val(self):
        pass

    def test(self):
        pass
