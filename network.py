import torch
from torch import nn
import mytorch
from data import DataSource
import numpy as np

""" 
    Authors: Erik Lidbjörk and Rasmus Söderström Nylander.
    Date: 2024.
"""


class DataSourceInterface:
    def vocab(self) -> list[str]:
        pass 

    # Return a list of data (context, label):
    def labeled_samples_batch(batch_size: int) -> any:
        pass 

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
    _device: str = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    _vocab_size: int
    _output_size: int
    _embedding_dim: int
    _padding = 'PAD'

    def __init__(self, 
                 data_src: DataSource = None, 
                 embedding_dim: int = 100,
                 num_layers: int = 1,
                 hidden_size: int = 50,
                 use_my_torch: bool = True,
                 dropout_rate: float = 0.5,
                 network_type: str = 'lstm') -> None:
        super().__init__()

        self.add_word(self._padding)
        self.learn_vocab(data_src)
        self._vocab_size = len(self._w2i)
        self._output_size = self._vocab_size
        self._embedding_dim = embedding_dim
        self._embeddings = nn.Embedding(self._vocab_size, embedding_dim)

        self._network_type = network_type
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
        self._dropout = nn.Dropout(p = dropout_rate)
        #self._final = nn.Linear(num_layers, self._output_size)
        layer_count = 0 
        layers = []
        for i in range(layer_count):
            layers.append(torch.nn.Linear(hidden_size, hidden_size))
            layers.append(torch.nn.Sigmoid())

        self._final = torch.nn.Sequential(*layers,\
                torch.nn.Linear(\
                num_layers * hidden_size,\
                self._output_size))

    def train_model(self, 
                    data_src: DataSource, 
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
                for i, data in enumerate(data_src.labeled_samples_batch(batch_size)):
                    # get the inputs; data is a list of [inputs, labels]
                    contexts, labels = data

                    #labels = list(map(lambda l: Special.UNKNOWN if l not in self._w2i else l, labels))
                    labels = list(map(lambda l: self._w2i[l], labels))
                    labels = torch.tensor(labels).to(self._device)
                
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = self(contexts)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # print statistics
                    running_loss += loss.item()
                    epoch_loss += loss.item()
                    if i % 200 == 199:    # print every 200 mini-batches
                        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                        running_loss = 0.0
                print(f'[epoch {epoch + 1}] loss: {epoch_loss / (i+1):.3f}')
            self.eval()
            print('Finished Training')

        except KeyboardInterrupt:
            self.eval()
            print('Finished Training early')

    def forward(self, x):
        if type(x) == type(''):
            x = [x]
        max_len = max(map(len, x))
        x = map(\
                lambda ctx: [self._padding] * (max_len - len(ctx)) + ctx,\
                x)
        x = list(x)
        x = list(map(\
                lambda ctx:\
                list(map(lambda w: self._w2i[w], ctx)),\
                x))
        ctx = torch.tensor(x, dtype=torch.long).to(self._device)
        ctx = ctx.permute((1, 0))
        ctx_emb = self._embeddings(ctx).float()
        
        _, sentence_state = self._rnn(ctx_emb)
        if self._network_type == 'lstm':
            sentence_state = sentence_state[0] # Retrieve h's
        batch_size = sentence_state.shape[1]
        sentence_state = sentence_state.reshape((batch_size, -1))
        sentence_state = self._dropout(sentence_state)
        logits = self._final(sentence_state)
        return logits

    def learn_vocab(self, data_src: DataSource) -> None:
        for word in data_src.vocab():
            self.add_word(word)     

    def add_word(self, word: str) -> None:
        if word not in self._w2i:
            idx = len(self._i2w)
            self._w2i[word] = idx
            self._i2w.append(word)
    
    def evaluate_model(self, batch_size, data_src: DataSource):
        odds = 0
        batch = 0
        for features, label in data_src.labeled_samples_batch(batch_size):
            batch += 1
            #self.eval()

            # ge till systemet
            # softmaxa det hela
            # jämför med resultat 
            logits = self.forward(features)
            #torch.Size([1, 83])
            #print(logits.shape)
            probs = torch.softmax(logits, dim=-1)
            index = torch.argmax(probs)
            result = self._i2w[index]
            if result == label[0]:
                odds += 1
                #print(odds)
            #print("Expected: " + str(label[0]))
            #print("Actual: " + str(result))


        bet = odds/batch
        print("Odds = " + str(bet))
        # Odds = 0.84075 #about that value
        return bet

        
            
        # argmaxa


        # jämför med labels
        #with open data_src as file:
        #    for line in data_src:
        #        intake, label = data_src.split(",")
        #        result = predict(intake)
        #        print("Expected: " + str(label))
        #        print("Actual: " + str(result))




    @staticmethod
    def main() -> None:
        data_src = DataSource("./data/train.txt")
        net = Network(data_src, use_my_torch=True, network_type='gru')
        net.train_model(data_src)
        data_test = DataSource("./data/test.txt")
        odds = net.evaluate_model(1, data_test)
        
if __name__ == '__main__':
    Network.main()