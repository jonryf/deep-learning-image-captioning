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
from data_loader import CocoDataset

def vocab_size():
    data = CocoDataset
    print(data.vocab)

def train_baseline():
    pass

def train_vanilla():
    # initialize model
    # get data
    # training loop
    # generate captions and save to file
    # plot training/validation error
    # save best model
    #
    pass


if __name__ == "__main__":
    i = ""
    while i != 'q':
        print("(s): print size of vocabulary")
        i = input("Please select your task: ")
        i = i.lower()
        if i == "s":
            vocab_size()
