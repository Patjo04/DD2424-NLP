import torch 
from torch import nn
import numpy as np

"""
    Author: Erik Lidbjörk
    Date: 2024.
    Overall structure (RNN, GRU, LSTM implements from RNNBase) and arguments to methods
    is based on the offical source code: https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/rnn.py,
    due to it being easy to debug and compare our implementation with theirs.
    The actual code is however written from scratch and based on the theory.

    Architectures and algorithms are based on 
    The course lectures,
    D. Jurafsky & J.H. Martin, Speech and Language Processing, 3rd ed, 
    https://en.wikipedia.org/wiki/Long_short-term_memory, 
    https://en.wikipedia.org/wiki/Recurrent_neural_network, and
    https://en.wikipedia.org/wiki/Gated_recurrent_unit

    Pytorch seems to have an additional bias vector, which is different from the course litterature.

    TODO: Implement bidirectional networks. Test correctness.
"""

"""
    RNN, GRU and LSTM should all inherit from this class.
    Currently assumes batch_first = False and unidirecional (see pytorch docs).
"""
class RNNBase(nn.Module):
    _input_size: int
    _hidden_size: int 
    _num_layers: int 

    # Weights and biases.
    _W: dict[str, torch.Tensor] = {}
    _U: dict[str, torch.Tensor] = {}
    _b: dict[str, torch.Tensor] = {}

    # Activation functions.
    _sigma: dict[str, nn.Module] = {}

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1) -> None:
        super().__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._num_layers = num_layers

    # Set internal weights.
    def _init_weights(self, key: str) -> None:
        self._W[key] = torch.empty(self._num_layers, self._hidden_size, self._input_size)
        self._U[key] = torch.empty(self._num_layers, self._hidden_size, self._hidden_size)
        self._b[key] = torch.empty(self._num_layers, self._hidden_size)

        k = 1 / self._hidden_size
        # Initialize from U(-sqrt(k), sqrt(k)) like in https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/rnn.py.
        # Other ideas are to sample from a normal distribution or use HE or Xavier initalizaiton. 
        for layer in range(self._num_layers):
            for weight in self._W[key][layer], self._U[key][layer], self._b[key][layer]:
                weight = nn.init.uniform_(weight, -np.sqrt(k), np.sqrt(k))
        
    # Children should implement this.
    def forward(self, x: torch.Tensor, h0: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        pass

# Elman network
class RNN(RNNBase):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1) -> None:
        super().__init__(input_size, hidden_size, num_layers)
        for key in 'h', 'y':
            self._init_weights(key)
        self._W['y'] = torch.empty(self._num_layers, self._input_size, self._hidden_size)
        self._b['y'] = torch.empty(self._num_layers, self._input_size)
        k = 1 / self._hidden_size
        self._W['y'] = nn.init.uniform_(self._W['y'], -np.sqrt(k), np.sqrt(k))
        self._b['y'] = nn.init.uniform_(self._b['y'], -np.sqrt(k), np.sqrt(k))

        self._sigma['h'] = nn.Sigmoid()
        self._sigma['y'] = nn.Softmax(dim = -1)

    # Took inspiration from: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html. 
    def forward(self, x: torch.Tensor, h_0: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        seq_length = x.shape[0]
        batch_size = x.shape[1]

        if h_0 is None:
            h_0 = torch.zeros(self._num_layers, batch_size, self._hidden_size)

        output = torch.empty(seq_length, batch_size, self._hidden_size)
        #output = torch.empty(seq_length, batch_size, self._input_size)
        h = h_0
        for t in range(seq_length):
            for layer in range(self._num_layers):
                # Update equations.
                h[layer] = self._sigma['h'](x[t] @ self._W['h'][layer].T + h[layer] @ self._U['h'][layer].T + self._b['h'][layer])
            output[t] = h[-1]
            #output[t] = self._sigma['y'](h[-1] @ self._W['y'][layer].T + self._b['y'][layer])
        return output, h

    # Compare class with pytorch implementation.
    # Source: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html.
    @staticmethod
    def main() -> None:
        #rnn = nn.RNN(10, 20, 2)
        rnn = RNN(10, 20, 2)
        input = torch.randn(5, 3, 10)
        h0 = torch.randn(2, 3, 20)
        output, hn = rnn(input, h0) 

# LSTM with a forget gate
class LSTM(RNNBase):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1) -> None:
        super().__init__(input_size, hidden_size, num_layers)
        for key in 'f', 'i', 'o', 'c':
            self._init_weights(key)
        self._sigma['g'] = nn.Sigmoid()
        self._sigma['c'] = nn.Tanh()
        self._sigma['h'] = nn.Tanh()  
        #self._sigma['h'] = nn.Identity()  # Suggestion from the peephole LSTM paper, see: https://en.wikipedia.org/wiki/Long_short-term_memory. 

    # Took inspiration from: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html. 
    def forward(self, x: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor] = None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        seq_length = x.shape[0]
        batch_size = x.shape[1]

        if states is None:
            h_0 = torch.zeros(self._num_layers, batch_size, self._hidden_size)
            c_0 = torch.zeros(self._num_layers, batch_size, self._hidden_size)
        else:
            h_0, c_0 = states

        output = torch.empty(seq_length, batch_size, self._hidden_size)
        h = h_0
        c = c_0
        for t in range(seq_length):
            for layer in range(self._num_layers):
                # Update equations.
                f = self._sigma['g'](x[t] @ self._W['f'][layer].T + h[layer] @ self._U['f'][layer].T + self._b['f'][layer])
                i = self._sigma['g'](x[t] @ self._W['i'][layer].T + h[layer] @ self._U['i'][layer].T + self._b['i'][layer])
                o = self._sigma['g'](x[t] @ self._W['o'][layer].T + h[layer] @ self._U['o'][layer].T + self._b['o'][layer])
                ct = self._sigma['c'](x[t] @ self._W['c'][layer].T + h[layer] @ self._U['c'][layer].T + self._b['c'][layer])
                c[layer] = f * c[layer] + i * ct
                h[layer] = o * self._sigma['h'](c[layer])
            output[t] = h[-1]
        return output, (h, c)

    # Compare class with pytorch implementation.
    # Source: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
    @staticmethod
    def main() -> None:
        #rnn = nn.LSTM(10, 20, 2)
        rnn = LSTM(10, 20, 2) 
        input = torch.randn(5, 3, 10)
        h0 = torch.randn(2, 3, 20)
        c0 = torch.randn(2, 3, 20)
        output, (hn, cn) = rnn(input, (h0, c0))

# Fully gated unit
class GRU(RNNBase):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1) -> None:
        super().__init__(input_size, hidden_size, num_layers)
        for key in 'z', 'r', 'h':
            self._init_weights(key)
        self._sigma['g'] = nn.Sigmoid() # Sigma in wikipedia article.
        self._sigma['h'] = nn.Tanh() # Phi in wikipedia article.

    # Took inspiration from: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html. 
    def forward(self, x: torch.Tensor, h_0: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        seq_length = x.shape[0]
        batch_size = x.shape[1]

        if h_0 is None:
            h_0 = torch.zeros(self._num_layers, batch_size, self._hidden_size)

        output = torch.empty(seq_length, batch_size, self._hidden_size)
        h = h_0
        for t in range(seq_length):
            for layer in range(self._num_layers):
                # Update equations.
                z = self._sigma['g'](x[t] @ self._W['z'][layer].T + h[layer] @ self._U['z'][layer].T + self._b['z'][layer])
                r = self._sigma['g'](x[t] @ self._W['r'][layer].T + h[layer] @ self._U['r'][layer].T + self._b['r'][layer])
                hh = self._sigma['h'](x[t] @ self._W['h'][layer].T + (r * h[layer]) @ self._U['h'][layer].T + self._b['h'][layer])
                h[layer] = (1 - z) * h[layer] + z * hh
            output[t] = h[-1]
        return output, h

    # Compare class with pytorch implementation.
    # Source: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
    @staticmethod
    def main() -> None:
        #rnn = nn.GRU(10, 20, 2)
        rnn = GRU(10, 20, 2)
        input = torch.randn(5, 3, 10)
        h0 = torch.randn(2, 3, 20)
        output, hn = rnn(input, h0)

# Test classes. 
# No run-time errors but correctness checks needed.
if __name__ == '__main__':
    RNN.main()
    LSTM.main()
    GRU.main()