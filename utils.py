import numpy as np
import torch
import matplotlib.pyplot as plt


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# gets softmax of each row with temperature
# subtracts max from every element in each row, keps ordering but plays with ratios of solutions
def softmax(x, temp=1.0):
    values = (((x.T - torch.max(x, axis=1)[0]).T) / temp).float()
    exp = torch.exp(values).float()
    sM = (exp.T / torch.sum(exp, axis=1)).T
    sM[torch.where(sM == 0)] = 0.0000001
    sM[torch.where(sM == 1)] = 0.9999999
    return sM

# draw one sample from distribution defined by rows in x
# if deterministic then returns max prob idx
def sampleFromDistribution(x, deterministic=True):
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
