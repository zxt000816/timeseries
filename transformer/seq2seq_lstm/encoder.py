import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim):
        super(Encoder, self).__init__()

        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim,  embedding_dim
        self.num_layers = 3
        self.rnn1 = nn.LSTM(
          input_size=n_features,
          hidden_size=self.hidden_dim,
          num_layers=3,
          batch_first=True,
          dropout = 0.35
        )
   
    def forward(self, x):
       
        # x = x.reshape((1, self.seq_len, self.n_features))
        x = x.reshape((1, self.seq_len, self.n_features)).float()
        
        # h_1 = Variable(torch.zeros(
        #     self.num_layers, x.size(0), self.hidden_dim).to(x.device))
        h_1 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device).float()
        
        # c_1 = Variable(torch.zeros(
        #     self.num_layers, x.size(0), self.hidden_dim).to(x.device))
        c_1 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device).float()
        
        x, (hidden, cell) = self.rnn1(x,(h_1, c_1))
        
        #return hidden_n.reshape((self.n_features, self.embedding_dim))
        return x, hidden , cell 