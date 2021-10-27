
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from scipy.spatial.distance import cdist


class GraphNet(nn.Module):
    def __init__(self, image_size = 28, pred_edge = False):
        super(GraphNet, self).__init__()
        self.pred_edge = pred_edge
        N = image_size ** 2 # Number of pixels in the image
        self.fc = nn.Linear(N, 10, bias = False)
        # Create the adjacency matrix of size (N X N)
        if pred_edge:
            # Learn the adjacency matrix (learn to predict the edge between any pair of pixels)
            col, row = np.meshgrid(np.arange(image_size), np.arange(image_size)) # (28 x 28) Explanation: https://www.geeksforgeeks.org/numpy-meshgrid-function/
            coord = np.stack((col, row), axis = 2).reshape(-1, 2)  # (784 x 2)