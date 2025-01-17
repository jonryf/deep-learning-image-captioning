from decoder import Decoder
from encoder import Encoder
from Embedding import getPreTrainedEmbeddingRunner
from evaluate_captions import evaluate_captions
from runner import Runner
from settings import LSTM_HIDDEN_SIZE, EMBEDDED_SIZE, TASK_NAME
from utils import load_datasets, get_device
from glove import loadGlove


def run_network():
    train_dataset, val_dataset, test_dataset = load_datasets()
    vocabulary_size = len(train_dataset.dataset.vocab.wordToIndex)

    computing_device = get_device()

    encoder = Encoder(EMBEDDED_SIZE).to(computing_device)
    decoder = Decoder(EMBEDDED_SIZE, LSTM_HIDDEN_SIZE, vocabulary_size).to(computing_device)

    runner = Runner(encoder, decoder, train_dataset, val_dataset, test_dataset)
    runner.train()


def show_options():
    print("(1): training and validation loss for LSTM and Vanilla RNN")
    print("(2): Cross Entropy and Perplexity score on test set")
    print("(3): BLEU-1 and BLEU-4 scores on deterministic LSTM and Vanilla RNN")
    print("(4): Experiment with Temperatures")
    print("(5): Pre=trained word embeddings")
    print("(q): quit program")


def task_4_1_lstm():
    """
    Training and validation loss for LSTM
    """
    global TASK_NAME; TASK_NAME = "Task 4-1 lstm"
    global LSTM; LSTM = True

    run_network()


def task_4_1_rnn():
    """
    Training and validation loss for Vanilla RNN
    """
    global TASK_NAME; TASK_NAME = "Task 4-1 rnn"
    global LSTM; LSTM = False

    run_network()


def task_4_2():
    """
    Cross Entropy and Perplexity score on test set
    """
    pass


def task_4_3():
    """
    BLEU-1 and BLEU-4 scores on deterministic LSTM and Vanilla RNN
    """
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


def task_4_4():
    """
    Experiment with temperatures
    """
    global TEMPERATURE
    TEMPERATURE = 1
    pass


def task_4_5():  # Pre-trained word embeddings
    runner = getPreTrainedEmbeddingRunner()
    runner.train()


if __name__ == "__main__":

    show_options()
    i = ""
    while i != 'q':
        i = input("Please select your task: ")
        i = i.lower()

        if i == "1i":
            task_4_1_lstm()
        if i == "1ii":
            task_4_1_rnn()

        elif i == "2":
            task_4_2()

        elif i == "3":
            task_4_3()

        elif i == "4":
            task_4_4()

        elif i == "5":
            task_4_5()
