import json

import torch

from torch.nn import NLLLoss, CrossEntropyLoss
from torch.optim import Adam

from settings import EPOCHS
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm

from utils import plot_training_data, get_device


class Runner:

    def __init__(self, encoder, decoder, train_dataset, val_dataset, test_dataset):
        """
        Runner for training, validation and testing a the network

        :param encoder: CNN encoder
        :param decoder: LSTM decoder
        :param train_dataset: the training dataset
        :param val_dataset: validation dataset
        :param test_dataset: test dataset
        """
        self.encoder = encoder
        self.decoder = decoder
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.criterion = CrossEntropyLoss()
        self.training_data = []
        self.caption_results = []
        params = list(self.decoder.parameters()) + list(self.encoder.parameters())
        self.optimizer = Adam(params)

    def train(self):
        """
        Train the network
        """
        for epoch in tqdm(range(EPOCHS)):
            train_loss = self.pass_data(self.train_dataset, True)
            val_loss = self.pass_data(self.val_dataset, False)

            self.training_data.append([train_loss, val_loss])

            print("Epoch: {}  -  training loss: {}, validation loss: {}".format(epoch, train_loss, val_loss))

        plot_training_data(self.training_data)

    def val(self):
        """
        Run validation

        :return: loss
        """
        with torch.no_grad():
            return self.pass_data(self.val_dataset, False)

    def test(self):
        """
        Run test set

        :return: loss
        """
        with torch.no_grad():
            return self.pass_data(self.test_dataset, False)

    def store_captions(self, captions, image_ids):
        for caption, image_id in zip(captions, image_ids):
            image_prediction = {"image_id": image_id, "caption": caption}
            self.caption_results.append(image_prediction)

    def save_captions(self, file_name="basline_lstm_captions.json"):
        with open(file_name, 'w') as file:
            json.dump(self.caption_results, file)

    def pass_data(self, dataset, backward, bleu):
        """
        Combined loop for training and data prediction, returns loss.

        :param dataset: dataset to do data passing from
        :param backward: if backward propagation should be used
        :return: loss
        """
        if backward:
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()
        loss = 0

        captions = []
        for minibatch, (images, captions, img_ids, lengths) in tqdm(enumerate(dataset)):
            computing_device = get_device()
            images = images.to(computing_device)
            captions = captions.to(computing_device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True).data
            # forward
            encoded = self.encoder(images)
            predicted = self.decoder(encoded, captions, lengths)

            # predicted = self.decoder.predict(encoded)
            batch_loss = self.criterion(predicted, targets)
            # backward
            if backward:
                self.encoder.zero_grad()
                self.decoder.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
            if bleu:
                self.store_captions(self.decoder.create_captions(encoded, 25), img_ids)

            loss += batch_loss.item()
        loss /= minibatch

        if bleu:
            self.save_captions()

        return loss
