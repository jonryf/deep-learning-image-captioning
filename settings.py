EPOCHS = 50  # Number of epochs
BATCH_SIZE = 1
SHUFFLE_DATA = True
NUM_WORKERS = 1

LSTM = True  # RNN or LSTM
TEMPERATURE = 1
LSTM_HIDDEN_SIZE = 1024
EMBEDDED_SIZE = 2048

IMAGES_DIR = "./data/images/"  # Location for images
CAPTIONS_DIR = "./data/annotations/"  # Location for annotation files
VALIDATION_SIZE = 0.2


def task_4_1():
    """
    Training and validation loss for LSTM and Vanilla RNN
    """
    pass


def task_4_2():
    """
    Cross Entropy and Perplexity score on test set
    """
    global EMBEDDED_SIZE
    EMBEDDED_SIZE = 5



task_4_1()
