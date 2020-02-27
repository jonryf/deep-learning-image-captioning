import numpy as np
import pandas
import torch
import matplotlib.pyplot as plt

from pycocotools.coco import COCO

from data_loader import get_loader
from settings import CAPTIONS_DIR, IMAGES_DIR, NUM_WORKERS, SHUFFLE_DATA, BATCH_SIZE, VALIDATION_SIZE, TEMPERATURE


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# gets softmax of each row with temperature
# subtracts max from every element in each row, keps ordering but plays with ratios of solutions
def softmax(x, temp=TEMPERATURE):
    values = (((x.T - torch.max(x, axis=1)[0]).T) / temp).float()
    exp = torch.exp(values).float()
    sM = (exp.T / torch.sum(exp, axis=1)).T
    sM[torch.where(sM == 0)] = 0.0000001
    sM[torch.where(sM == 1)] = 0.9999999
    return sM

# draw one sample from distribution defined by rows in x
# if deterministic then returns max prob idx
def sample_from_distribution(x, deterministic=True):
    # if deterministic then return max for each row
    if deterministic:
        return torch.argmax(x, axis=1)

    sampler = torch.distributions.categorical.Categorical(probs=x)
    return sampler.sample()

def plot_graph(data, labels, legends, title):
    """
    Plot multiple graphs in same plot

    :param data: data of the graphs to be plotted
    :param labels: x- and y-label
    :param legends: legends for the graphs
    :param title: Title of the graph
    """
    x = np.arange(1, len(data[0]) + 1)
    for to_plot in data:
        plt.plot(x, to_plot)
    plt.title(title)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.legend(legends)
    plt.show()
    plt.savefig('{}.png'.format(title))


def plot_training_data(data):
    """
    Plot training data, loss over epochs.
    """
    training_data = np.array(data).T
    plot_graph(training_data, ["Epoch", "Cross-entropy-loss"], ["Training loss", "Validation loss"],
               "Loss over epochs")


def load_image_ids(train=True):
    """
    Load select images

    :param train: train or test image ids
    :return: ids
    """
    if train:
        path = './TrainImageIds.csv'
    else:
        path = './TestImageIds.csv'
    df = pandas.read_csv(path, header=None)
    df = np.array(df)
    return np.array(df).reshape(-1).tolist()


def select_ann_ids(ids, path):
    """
    Select the annotation ids based on list of selected image ids

    :param ids: image ids
    :param path: annotation file
    :return: annotation ids
    """
    coco = COCO(CAPTIONS_DIR + path)
    filtered = []
    for ann_id in coco.anns:
        ann = coco.anns[ann_id]
        if ann['image_id'] in ids:
            filtered.append(ann_id)
    return filtered


def load_datasets():
    """
    Load the dataset based on the selection of image ids

    :return: train, val and test dataset
    """
    # Select ids
    training_ids = select_ann_ids(load_image_ids(True), "captions_train2014.json")

    validation_ids = training_ids[:int(len(training_ids) * VALIDATION_SIZE)]
    training_ids = training_ids[int(len(training_ids) * VALIDATION_SIZE):]
    testing_ids = select_ann_ids(load_image_ids(False), "captions_val2014.json")

    # load data and transform images
    train_dataset = get_loader(IMAGES_DIR + "/train/", CAPTIONS_DIR + "/captions_train2014.json", training_ids,
                               BATCH_SIZE, SHUFFLE_DATA,
                               NUM_WORKERS)
    val_dataset = get_loader(IMAGES_DIR + "/train/", CAPTIONS_DIR + "/captions_train2014.json", validation_ids,
                             BATCH_SIZE, SHUFFLE_DATA,
                             NUM_WORKERS)
    test_dataset = get_loader(IMAGES_DIR + "/test/", CAPTIONS_DIR + "/captions_val2014.json", testing_ids, BATCH_SIZE,
                              SHUFFLE_DATA,
                              NUM_WORKERS)
    return train_dataset, val_dataset, test_dataset
