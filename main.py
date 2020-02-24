from data_loader import get_loader
from models import Encoder, Decoder
from runner import Runner
from settings import BATCH_SIZE, SHUFFLE_DATA, NUM_WORKERS, ROOT

if __name__ == "__main__":
    # load data and transform images

    train_dataset = get_loader(ROOT, "train2014", ids, vocab, BATCH_SIZE, SHUFFLE_DATA, NUM_WORKERS)
    val_dataset = get_loader(ROOT, "train2014", ids, vocab, BATCH_SIZE, SHUFFLE_DATA, NUM_WORKERS)
    test_dataset = get_loader(ROOT, "train2014", ids, vocab, BATCH_SIZE, SHUFFLE_DATA, NUM_WORKERS)

    encoder = Encoder(2048)
    decoder = Decoder(2048, 1024, -1)

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
