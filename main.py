from data_loader import get_loader
from models import Encoder, Decoder
from runner import Runner
from settings import BATCH_SIZE, SHUFFLE_DATA, NUM_WORKERS, IMAGES_DIR, CAPTIONS_DIR, VALIDATION_SIZE, LSTM_HIDDEN_SIZE, \
    EMBEDDED_SIZE
from evaluate_captions import evaluate_captions
import pandas
import numpy as np

from utils import select_ann_ids, load_image_ids, load_datasets


def run():
    train_dataset, val_dataset, test_dataset = load_datasets()
    vocabulary_size = len(train_dataset.dataset.vocab.word2idx)

    encoder = Encoder(EMBEDDED_SIZE)
    decoder = Decoder(EMBEDDED_SIZE, LSTM_HIDDEN_SIZE, vocabulary_size)

    runner = Runner(encoder, decoder, train_dataset, val_dataset, test_dataset)
    runner.train()


def show_options():
    print("(1): training and validation loss for LSTM and Vanilla RNN")
    print("(2): Cross Entropy and Perplexity score on test set")
    print("(3): BLEU-1 and BLEU-4 scores on deterministic LSTM and Vanilla RNN")
    print("(4): Experiment with Temperatures")
    print("(5): Pre=trained word embeddings")
    print("(q): quit program")


def task_4_1():  # training and validation loss for LSTM and Vanilla RNN
    EMBEDDED_SIZE = 5
    pass


def task_4_2():  # Cross Entropy and Perplexity score on test set
    pass


def task_4_3():  # BLEU-1 and BLEU-4 scores on deterministic LSTM and Vanilla RNN
    true_captions_path = './'

    print("Scoring Deterministic LSTM")
    deterministic_LSTM_captions_path = './'  # deterministic generation
    b1, b4 = evaluate_captions(true_captions_path, deterministic_LSTM_captions_path)
    print("BLEU-1 Score:", b1)
    print("BLEU-4 Score:", b4)

    print("Scoring Deterministic Vanilla RNN")
    deterministic_vanilla_captions_path = './'  # deterministic generation
    b1, b4 = evaluate_captions(true_captions_path, deterministic_vanilla_captions_path)
    print("BLEU-1 Score:", b1)
    print("BLEU-4 Score:", b4)


def task_4_4():  # Experiment with temperatures
    pass


def task_4_5():  # Pre-trained word embeddings
    pass


if __name__ == "__main__":

    show_options()
    i = ""
    while i != 'q':
        i = input("Please select your task: ")
        i = i.lower()

        if i == "1":
            task_4_1()

        elif i == "2":
            task_4_2()

        elif i == "3":
            task_4_3()

        elif i == "4":
            task_4_4()

        elif i == "5":
            task_4_5()
        run()
