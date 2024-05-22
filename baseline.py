import torch
from mytorch import RNN
from pytorch.nn import Linear


def default_device():
    return 'cuda' if torch.cuda.is_available()\
            else 'cpu'

class KernelRNN(torch.nn.Module):
    def __init__(max_char, hidden_size=100, device=None):
        super().__init__()
        self._num_chars = max_char + 1
        self._input_size = self._num_chars
        self._output_size = self._num_chars
        self._hidden_size = hidden_size
        
        self._rnn = RNN(self._input_size, self._hidden_size)
        self._final = Linear(self._hidden_size, self._output_size)
        
        if device is None
            device = default_device()
        self.set_device(device)

    def set_device(self, device):
        self._device = device
        self.to(device)
        
    def save(self, path):
        torch.save(self, path)

    def load(path, device=None):
        if device is None:
            device = default_device()
        model = torch.load(path)
        model.set_device(device) 

    def char_to_tensor(x):
        x = ord(x)
        x = torch.tensor(x)
        return torch.nn.functional.one_hot(x, num_classes=self._num_chars)\
                .to(self._device)

    def forward(x, h):
        # x is character
        # h is hidden state
        # predict next character (return logits)
        x = self.char_to_tensor(x)
        _, hidden = self._rnn(x, h)
        logits = self._final(hidden)
        return logits, hidden
