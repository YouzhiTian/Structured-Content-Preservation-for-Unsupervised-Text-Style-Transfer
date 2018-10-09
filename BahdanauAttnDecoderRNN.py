import torch
from torch import nn
from Attn import Attn
from AttnDecoderRNN import AttnDecoderRNN
import torch.nn.functional as F
from torch.autograd import Variable
eps=1e-20
class BahdanauAttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, embed_size, output_size, n_layers=1, dropout_p=0.1):
        super(BahdanauAttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(dropout_p)
        self.attn = Attn(hidden_size)
        self.gru = nn.GRU(hidden_size + embed_size, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self,embedding, word_input, initial_state, encoder_outputs,gumbel = False,gamma = 1):
        word_embedded = embedding(word_input).view(1, word_input.size(0), -1) 
        word_embedded = self.dropout(word_embedded)
        attn_weights = self.attn(initial_state, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs)  
        context = context.transpose(0, 1)  
        rnn_input = torch.cat((word_embedded, context), 2)
        output, hidden = self.gru(rnn_input, initial_state)
        output = output.squeeze(0)  
        if gumbel:
            output = F.gumbel_softmax(self.out(output),tau = gamma)
        else:
            output = F.log_softmax(self.out(output))
        return output, hidden