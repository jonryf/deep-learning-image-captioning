from torch.nn import NLLLoss
from torch.optim import Adam

from settings import EPOCHS


class Runner:

    def __init__(self, model, train_dataset):
        self.model = model
        self.train_dataset = train_dataset

    def train(self):
        self.model.train()

        criterion = NLLLoss()
        optimizer = Adam()

        for epoch in range(EPOCHS):
            # Reset state

            self.model.init_state()

            for minibatch, (image, caption) in enumerate(self.train_dataset):

                # encoder

                # decoder

                pass






    def val(self):
        pass

    def test(self):
        pass


