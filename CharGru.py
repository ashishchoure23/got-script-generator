import torch
from torch import nn
import torch.nn.functional as functional
import numpy as np


class CharGruNet(nn.Module):
    def __init__(self, tokens, hidden_size = 256, num_layers = 2, drop_prob = 0.5):
        super().__init__()

        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}

        self.n_hidden = hidden_size
        self.n_layers = num_layers
        self.drop_prob = drop_prob

        # ==============GRU===================
        self.gru = nn.GRU(input_size = len(self.chars),
                            hidden_size = hidden_size,
                            num_layers = num_layers,
                            dropout = drop_prob,
                            batch_first = True)
        # ==============GRU===================

        self.dropout = nn.Dropout(p = self.drop_prob)

        self.fc = nn.Linear(hidden_size, len(self.chars))


    def forward(self, x, hidden):

        gru_out, hidden = self.gru(x, hidden)

        #gru_out = self.dropout(gru_out)

        gru_out = gru_out.reshape(-1, self.n_hidden)

        fc_out = self.fc(gru_out)

        return fc_out, hidden


    def init_hidden(self, batch_size):

        '''
        h_0 of shape (num_layers * num_directions, batch, hidden_size): tensor containing the initial hidden state for each element in the batch.
        If the LSTM is bidirectional, num_directions should be 2, else it should be 1.

        If (h_0) is not provided, both h_0 and c_0 default to zero.
        '''
        train_on_gpu = torch.cuda.is_available()

        if train_on_gpu:
            hidden = torch.zeros((self.n_layers, batch_size, self.n_hidden)).cuda()                     
        else:
            hidden = torch.zeros((self.n_layers, batch_size, self.n_hidden))

        return hidden


    def encode_text(self, text):
        encoded = np.array([self.char2int[ch] for ch in text])

        return encoded
#=======================
