import pickle
import nltk

from os import path
from pycocotools.coco import COCO

from settings import CAPTIONS_DIR


class Vocabulary:

    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}


def get_vocabulary(file_name):
    path_to_file = file_name + "-tokenized.p"

    if path.exists(path_to_file):
        return pickle.load(open(path_to_file, "rb"))

    coco = COCO(CAPTIONS_DIR + file_name)
    items = set()
    for id, key in coco.anns.keys():
        caption = str(coco.anns[id]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        items.update(tokens)

    vocab = Vocabulary()

    for index, item in enumerate(items):
        .vocab.word_to_id[item] = index