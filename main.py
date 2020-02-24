import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary
from pycocotools.coco import COCO
from data_loader import get_loader
from evaluate_captions import evaluate_captions
def show_options():
    print("(1): print size of vocabulary")
    print("(2): print size of vocabulary")
    print("(3): print size of vocabulary")
    print("(4): print size of vocabulary")
    print("(5): print size of vocabulary")
    print("(q): quit program")

def task_4_1(): # training and validation loss for LSTM and Vanilla RNN
    pass
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

def task_4_4():
    pass
def task_4_5():
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
