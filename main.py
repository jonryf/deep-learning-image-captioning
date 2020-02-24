from data_loader import get_loader
from models import Encoder, Decoder
from runner import Runner
from settings import BATCH_SIZE, SHUFFLE_DATA, NUM_WORKERS, IMAGES_DIR, CAPTIONS_DIR
from evaluate_captions import evaluate_captions
import pandas
import numpy as np


def process_csv(train=True):
    if train:
        path = './TrainImageIds.csv'
    else:
        path = './TestImageIds.csv'
    df = pandas.read_csv(path, header=None)
    df = np.array(df)
    print("Training set = ", train)
    print("# Images: ", df.size)
    return np.array(df)


def show_options():
    print("(1): training and validation loss for LSTM and Vanilla RNN")
    print("(2): Cross Entropy and Perplexity score on test set")
    print("(3): BLEU-1 and BLEU-4 scores on deterministic LSTM and Vanilla RNN")
    print("(4): Experiment with Temperatures")
    print("(5): Pre=trained word embeddings")
    print("(q): quit program")


def task_4_1(): # training and validation loss for LSTM and Vanilla RNN

    training_ids = process_csv(True)
    testing_ids = process_csv(False)

    # load data and transform images
    train_dataset = get_loader(IMAGES_DIR + "/train/", CAPTIONS_DIR + "/train/", BATCH_SIZE, SHUFFLE_DATA, NUM_WORKERS)
    val_dataset = get_loader(IMAGES_DIR + "/val/", CAPTIONS_DIR + "/val/", BATCH_SIZE, SHUFFLE_DATA, NUM_WORKERS)
    test_dataset = get_loader(IMAGES_DIR + "/test/", CAPTIONS_DIR + "/test/", BATCH_SIZE, SHUFFLE_DATA, NUM_WORKERS)

    encoder = Encoder(2048)
    decoder = Decoder(2048, 1024, len(train_dataset.dataset.vocab.word2idx))

    runner = Runner(encoder, decoder, train_dataset, val_dataset, test_dataset)

    runner.train()
def task_4_2(): # Cross Entropy and Perplexity score on test set
    pass


def task_4_3(): # BLEU-1 and BLEU-4 scores on deterministic LSTM and Vanilla RNN
    true_captions_path = './'

    print("Scoring Deterministic LSTM")
    deterministic_LSTM_captions_path = './' # deterministic generation
    b1, b4 = evaluate_captions(true_captions_path, deterministic_LSTM_captions_path)
    print("BLEU-1 Score:", b1)
    print("BLEU-4 Score:", b4)

    print("Scoring Deterministic Vanilla RNN")
    deterministic_vanilla_captions_path = './' # deterministic generation
    b1, b4 = evaluate_captions(true_captions_path, deterministic_vanilla_captions_path)
    print("BLEU-1 Score:", b1)
    print("BLEU-4 Score:", b4)


def task_4_4(): # Experiment with temperatures
    pass


def task_4_5(): # Pre-trained word embeddings
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
