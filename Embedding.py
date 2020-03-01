
from decoder import Decoder
from encoder import Encoder
from runner import Runner
from settings import LSTM_HIDDEN_SIZE, EMBEDDED_SIZE
from utils import load_datasets, get_device
from glove import loadGlove
import numpy as np
from torch import nn
import torch

def getPreTrainedEmbeddingRunner():
    embed_size = 50

        

    glove = loadGlove()
    print("Loaded glove")
    train_dataset, val_dataset, test_dataset = load_datasets()

    vocabulary_size = len(train_dataset.dataset.vocab.wordToIndex)

    glove_weights = np.zeros((vocabulary_size, embed_size))
    missed_words = 0
    found_words = 0

    for i, word in enumerate(train_dataset.dataset.vocab.wordToIndex.keys()):
        try: 
            glove_weights[i] = glove[word]
            found_words += 1
        except KeyError:
            glove_weights[i] = np.random.normal(scale=0.6, size=(embed_size, ))
            missed_words += 1

    print("{} words not in glove".format(missed_words))
    print("{} words found".format(found_words))

    computing_device = get_device()

    encoder = Encoder(embed_size).to(computing_device)
    decoder = Decoder(embed_size, LSTM_HIDDEN_SIZE, vocabulary_size).to(computing_device)

    (num_embed, embed_dim) = glove_weights.shape

    embed_layer = nn.Embedding(num_embed, embed_dim).to(computing_device)
    #glove_weights = torch.tensor(glove_weights).to(computing_device)

    embed_layer.load_state_dict({'weight': torch.tensor(glove_weights)})
    embed_layer.weight.requires_grad = False
    decoder.embedding = embed_layer

    runner = Runner(encoder, decoder, train_dataset, val_dataset, test_dataset)
    return runner