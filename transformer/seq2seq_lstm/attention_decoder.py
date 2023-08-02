import torch
import torch.nn as nn

class AttentionDecoder(nn.Module):
    def __init__(self, seq_len, attention, input_dim, n_features, encoder_hidden_state):
        super(AttentionDecoder, self).__init__()

        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features =  input_dim, n_features
        self.attention = attention 
        
        self.rnn1 = nn.LSTM(
          #input_size=1,
          input_size= encoder_hidden_state + 1,  # Encoder Hidden State + One Previous input
          hidden_size=input_dim,
          num_layers=3,
          batch_first=True,
          dropout = 0.35
        )
        self.output_layer = nn.Linear(self.hidden_dim * 2 , n_features)

    def forward(self, x, input_hidden, input_cell, encoder_outputs): 

      input_hidden = input_hidden.float()
      input_cell = input_cell.float()
      
      a = self.attention(input_hidden, encoder_outputs)
      
      a = a.unsqueeze(1)
      
      #a = [batch size, 1, src len]
      
      #encoder_outputs = encoder_outputs.permute(1, 0, 2)
      
      #encoder_outputs = [batch size, src len, enc hid dim * 2]
      
      weighted = torch.bmm(a, encoder_outputs)
      x = x.reshape((1,1,1))
      
      rnn_input = torch.cat((x, weighted), dim = 2).float()
      self.rnn1.to(rnn_input.device)

      #x, (hidden_n, cell_n) = self.rnn1(x,(input_hidden,input_cell))

      self.rnn1.to(rnn_input.device).float()

      x, (hidden_n, cell_n) = self.rnn1(rnn_input, (input_hidden, input_cell))
      
      output = x.squeeze(0)
      weighted = weighted.squeeze(0)
      
      x = self.output_layer(torch.cat((output, weighted), dim = 1))
      return x, hidden_n, cell_n