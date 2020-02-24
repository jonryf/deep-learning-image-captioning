import pickle

from data_loader import get_loader
from models import Encoder, Decoder
from runner import Runner
from settings import BATCH_SIZE, SHUFFLE_DATA, NUM_WORKERS, IMAGES_DIR, CAPTIONS_DIR

if __name__ == "__main__":
    # load data and transform images
    train_dataset = get_loader(IMAGES_DIR + "/train/", CAPTIONS_DIR + "/train/", BATCH_SIZE, SHUFFLE_DATA, NUM_WORKERS)
    val_dataset = get_loader(IMAGES_DIR + "/val/", CAPTIONS_DIR + "/val/", BATCH_SIZE, SHUFFLE_DATA, NUM_WORKERS)
    test_dataset = get_loader(IMAGES_DIR + "/test/", CAPTIONS_DIR + "/test/", BATCH_SIZE, SHUFFLE_DATA, NUM_WORKERS)

    encoder = Encoder(2048)
    decoder = Decoder(2048, 1024, len(train_dataset.dataset.vocab.word2idx))

    runner = Runner(encoder, decoder, train_dataset, val_dataset, test_dataset)

    runner.train()

    # # main UI loop
    # i = ""
    # while i != 'q':
    #     print("(s): print size of vocabulary")
    #     i = input("Please select your task: ")
    #     i = i.lower()
    #     if i == "s":
    #         vocab_size()
