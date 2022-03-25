"""
Image Search Engine for Historical Research: A Prototype
This file contains classes/functions related to feature extractors
"""

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from torch.utils.data import Dataset

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def feature_normlization(X, N_books):
    '''
    Input:
        X = [[x_11,...x_1M],...[x_N1,...,x_NM]], x_ij are row vectors
        N_books = M
        Apply l2 normalization to each x_ij
    Output: Normalized X
    '''
    L_word = int(X.size()[1] / N_books)
    x = list(T.split(X, L_word, dim=1))
    for i in range(N_books):
        x[i] = F.normalize(x[i], dim=1)
    x = tuple(x)
    X = T.cat(x, dim=1)
    return X


def Soft_Quantization(X, C, N_books, tau_q):
    '''
    Generates soft-quantized feature vectors and the index matrix
    Input:
        X: feature vectors of images in the database  N_images * (N_books * L_word)
        C: a matrix contains all the codewords  N_words * (N_books * L_word)
        N_books: number of codebooks N_words: number of codewords per codebook L_word: length of codewords
        tau_q: a constant. larger -> soft quantization is closer with hard quantization
    Output:
        Z: soft-quantized feature vectors
        idx: A matrix contains the indexes of the corresponding sub-codewords of the sub-vectors  N_images * N_books
    '''
    L_word = int(C.size()[1]/N_books)
    # x = list(T.split(X, L_word, dim=1))
    x = T.split(X, L_word, dim=1)
    c = T.split(C, L_word, dim=1)
    for i in range(N_books):
        # normalization
        # x[i] = F.normalize(x[i], dim=1)
        # c[i] = F.normalize(c[i], dim=1)
        dist = squared_distances(x[i], c[i])
        soft_c = F.softmax(dist * (-tau_q), dim=-1)
        if i==0:
            Z = soft_c @ c[i]
            idx = T.argmin(dist, dim=1, keepdim=True)
        else:
            Z = T.cat((Z, soft_c @ c[i]), dim=1)
            idx = T.cat((idx, T.argmin(dist, dim=1, keepdim=True)), dim=1)
    return Z, idx


def squared_distances(x, y):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    diff = x.unsqueeze(1) - y.unsqueeze(0)
    return T.sum(diff * diff, -1)


def resnet18(pretrained=True, **kwargs):
    """
    Construct a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = torchvision.models.resnet.ResNet(
        torchvision.models.resnet.BasicBlock, [2, 2, 2, 2])
    if pretrained:
        model.load_state_dict(T.utils.model_zoo.load_url(
            model_urls['resnet18'], model_dir='../models'))
    return EmbeddingNet(model)


def resnet101(pretrained=True, **kwargs):
    """
    Construct a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = torchvision.models.resnet.ResNet(
        torchvision.models.resnet.BasicBlock, [3, 4, 23, 3])
    if pretrained:
        model.load_state_dict(T.utils.model_zoo.load_url(
            model_urls['resnet101'], model_dir='../models'))
    return EmbeddingNet(model)


class TripletNet(nn.Module):
    """Triplet Network."""

    def __init__(self, embeddingnet):
        """Triplet Network Builder."""
        super(TripletNet, self).__init__()
        self.embeddingnet = embeddingnet

    def forward(self, a, p, n):
        """Forward pass."""
        # anchor
        embedded_a = self.embeddingnet(a)

        # positive examples
        embedded_p = self.embeddingnet(p)

        # negative examples
        embedded_n = self.embeddingnet(n)

        return embedded_a, embedded_p, embedded_n


class EmbeddingNet(nn.Module):
    """EmbeddingNet using ResNet."""

    def __init__(self, resnet):
        """Initialize EmbeddingNet model."""
        super(EmbeddingNet, self).__init__()

        # Everything except the last linear layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        num_ftrs = resnet.fc.in_features
        self.fc1 = nn.Linear(num_ftrs, 4096)

    def forward(self, x):
        """Forward pass of EmbeddingNet."""
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)

        return out


class Soft_PQ(nn.Module):
    """Product Quantization Layer"""
    def __init__(self, N_words=64, N_books=128, L_word=32, tau_q=10):
        super(Soft_PQ, self).__init__()
        # self.fc = nn.Linear(4096, N_books * L_word, bias=False)
        # nn.init.xavier_normal_(self.fc.weight, gain=0.1)

        # Codebooks
        self.C = T.nn.Parameter(Variable((T.randn(N_words, N_books * L_word)).type(T.float32), requires_grad=True))
        nn.init.xavier_normal_(self.C, gain=0.1)

        self.N_books = N_books
        self.L_word = L_word
        self.tau_q = tau_q

    def forward(self, input):
        # X = self.fc(input)
        # Z = Soft_Quantization(X, self.C, self.N_books, self.tau_q)
        self.C = T.nn.Parameter(feature_normlization(self.C, self.N_books))
        Z, idx = Soft_Quantization(input, self.C, self.N_books, self.tau_q)
        return Z, idx


class TripletNet_PQ(nn.Module):
    """Triplet Network with Product Quantization Layer"""

    def __init__(self, embeddingnet, Soft_PQ):
        """Triplet Network Builder."""
        super(TripletNet_PQ, self).__init__()
        self.embeddingnet = embeddingnet
        self.Soft_PQ = Soft_PQ
        self.C = Soft_PQ.C
        self.N_books = Soft_PQ.N_books

    def forward(self, a, p, n):
        """Forward pass."""
        # anchor
        embedded_a = self.embeddingnet(a)
        embedded_a = feature_normlization(embedded_a, self.N_books)

        # positive examples
        embedded_p = self.embeddingnet(p)
        embedded_p = feature_normlization(embedded_p, self.N_books)
        quantized_p, idx_p = self.Soft_PQ(embedded_p)

        # negative examples
        embedded_n = self.embeddingnet(n)
        embedded_n = feature_normlization(embedded_n, self.N_books)
        quantized_n, idx_n = self.Soft_PQ(embedded_n)

        return embedded_a, quantized_p, quantized_n, idx_p, idx_n
