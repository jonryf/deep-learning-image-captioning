import torch
from utils import load_datasets
from runner import Runner
import numpy as np

train_dataset, val_dataset, test_dataset = load_datasets()

encoder_file = input("Encoder file: ")
decoder_file = input("Decoder file: ")


encoder = torch.load("./{}".format(encoder_file))
decoder = torch.load("./{}".format(decoder_file))

runner = Runner(encoder, decoder, train_dataset, val_dataset, test_dataset)

bleu1, bleu4, test_loss = runner.test()
print("\n --- Test scores ---  \nBleu 1:     {} \nBleu4:      {} \nLoss:       {} \nPerplexity: {}"
              .format(bleu1, bleu4, test_loss, np.exp(test_loss)))