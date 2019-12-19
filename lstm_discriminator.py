import torch
import torch.nn as nn

from neuroir.inputters import PAD
from neuroir.modules.embeddings import Embeddings
from neuroir.encoders import RNNEncoder
import numpy as np


class Encoder(nn.Module):
    def __init__(self, args, input_size):
        super(Encoder, self).__init__()
        self.encoder = RNNEncoder(args.rnn_type,
                                  input_size,
                                  args.bidirection,
                                  args.nlayers,
                                  args.nhid,
                                  args.dropout_rnn)

    def forward(self, input, input_len):
        hidden, M = self.encoder(input, input_len)  # B x Seq-len x h
        return hidden, M


class LSTM_Discriminator(nn.Module):
    """Class that classifies question pair as duplicate or not."""

    def __init__(self, num_classes, vocab_size, emb_dim,dropout_rnn,rnn_type,bidirection,nlayers,nhid):
        """"Constructor of the class."""
        super(LSTM_Discriminator, self).__init__()

        self.emb = nn.Embedding(vocab_size, emb_dim)

        self.encoder = RNNEncoder(rnn_type,
                                        emb_dim,
                                        bidirection,
                                        nlayers,
                                        nhid,
                                        dropout_rnn)

        self.output = nn.Linear(nhid, num_classes)
        self.softmax = nn.LogSoftmax()


    def forward(self, x):
        emb = self.emb(x)
        a, encoded_seq = self.encoder(emb)
        scores = self.softmax(self.output(encoded_seq[:,-1]))

        return scores

