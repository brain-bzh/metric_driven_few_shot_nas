import torch
import torch.nn as nn
import numpy as np

# Taken from TENAS (Credits W. Chen)

class RandChannel(object):
    # randomly pick channels from input
    def __init__(self, num_channel):
        self.num_channel = num_channel

    def __repr__(self):
        return ('{name}(num_channel={num_channel})'.format(name=self.__class__.__name__, **self.__dict__))

    def __call__(self, img):
        n_channels = img.size(0)
        channel_choice = sorted(np.random.choice(list(range(n_channels)), size=self.num_channel, replace=False))
        return torch.index_select(img, 0, torch.Tensor(channel_choice).long())