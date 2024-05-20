import torch
from torch import nn
import mytorch

""" 
    Authors: Erik Lidbjörk and Rasmus Söderström Nylander.
    Date: 2024.
"""

""" 
    TODO: Read from data source (fill word-index), 
    continue implement training function,
    continue building architecture. 
"""
class Network(nn.Module):
    _rnn: mytorch.RNNBase | nn.RNNBase
    _embeddings: nn.Embedding
    _final: nn.Module
    _w2i: dict[str, int] = {}
    _i2w: list[str] = []
    device: str = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    _NUM_SPECIAL_WORDS: int
    _vocab_size: int
    _output_size: int
    _embedding_dim: int

    def __init__(self, 
                 data_src: str = None, 
                 embedding_dim: int = 100,
                 num_layers: int = 1,
                 hidden_size: int = 50,
                 use_my_torch: bool = True,
                 network_type: str = 'lstm') -> None:
        super().__init__()

        self._NUM_SPECIAL_WORDS = Special.size()
        self.add_word(Special.PADDING)
        self.add_word(Special.UNKNOWN)
        self.add_word(Special.START)

        self._vocab_size = len(self._w2i)
        self._output_size = self._vocab_size
        self._embedding_dim = embedding_dim
        self._embeddings = nn.Embedding(self._vocab_size, embedding_dim)

        match network_type.strip().lower():
            case 'lstm':
                self._rnn = mytorch.LSTM(embedding_dim, hidden_size, num_layers) if use_my_torch \
                    else nn.LSTM(embedding_dim, hidden_size, num_layers)
            case 'gru':
                self._rnn = mytorch.GRU(embedding_dim, hidden_size, num_layers) if use_my_torch \
                    else nn.GRU(embedding_dim, hidden_size, num_layers)
            case 'rnn':
                self._rnn = mytorch.RNN(embedding_dim, hidden_size, num_layers) if use_my_torch \
                    else nn.RNN(embedding_dim, hidden_size, num_layers)
            case _:
                raise ValueError("Unknown model.")
        self._final = nn.Linear(num_layers, self._output_size)

    def train_model(self, 
                    data_src: str, 
                    epochs: int = 1,
                    batch_size: int = 16,
                    optimizer_type: str = 'adam',
                    loss_type: str = 'cross entropy') -> None:

        match optimizer_type.strip().lower():
            case 'sgd':
                optimizer = torch.optim.SGD(self.parameters())
            case 'adam':
                optimizer = torch.optim.Adam(self.parameters())
            case 'adagrad':
                optimizer = torch.optim.Adagrad(self.parameters())
            case _:
                raise ValueError("Unknown optimizer.")

        match loss_type.strip().lower():
            case 'cross entropy':
                criterion = torch.nn.CrossEntropyLoss()
            case _:
                raise ValueError("Unknown loss.")

        print('Starting training')
        self.train() # Set to training mode.
        try: 
            for epoch in range(epochs):
                print('epoch', epoch+1)
                epoch_loss = 0.0
                running_loss = 0.0
                # TODO: Implement.

        except KeyboardInterrupt:
            self.eval()
            print('Finished Training early')
            

    def add_word(self, word: str) -> None:
        if word not in self._w2i:
            idx = len(self._i2w)
            self._w2i[word] = idx
            self._i2w.append(word)

    @staticmethod
    def main() -> None:
        net = Network('data')
        net.train_model('data')
        
# TODO: Should probably be moved to another file.
class Special:
    PADDING = '<P>'
    UNKNOWN = '<U>'
    START = '<S>'
    
    def all():
        return [Special.PADDING, Special.UNKNOWN, Special.START]
    
    def size():
        return len(Special.all())



if __name__ == '__main__':
    Network.main()