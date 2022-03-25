import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from src.layers.functional import Soft_Quantization, feature_normlization


class Soft_PQ(nn.Module):
    """Product Quantization Layer"""
    def __init__(self, N_words=256, N_books=8, L_word=32, tau_q=5):
        super(Soft_PQ, self).__init__()
        # self.fc = nn.Linear(4096, N_books * L_word, bias=False)
        # nn.init.xavier_normal_(self.fc.weight, gain=0.1)

        # Codebooks
        self.C = Parameter(Variable((torch.randn(N_words, N_books * L_word)).type(torch.float32), requires_grad=True))
        nn.init.xavier_normal_(self.C, gain=0.1)

        self.N_books = N_books
        self.L_word = L_word
        self.tau_q = tau_q

    def forward(self, input):
        # X = self.fc(input)
        # self.C = T.nn.Parameter(feature_normlization(self.C, self.N_books))
        # Z, idx = Soft_Quantization(X, self.C, self.N_books, self.tau_q)
        # self.C = Parameter(feature_normlization(self.C, self.N_books))
        Z, idx, C= Soft_Quantization(input, self.C, self.N_books, self.tau_q)
        return Z, idx, C