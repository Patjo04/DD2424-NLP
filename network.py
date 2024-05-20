import torch
from torch import nn
import mytorch

class Network(nn.Module):
    def __init__(self, data_src=None) -> None:
        self.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self._NUM_SPECIAL_WORDS = Special.size()
        self.add_word(Special.PADDING)
        self.add_word(Special.UNKNOWN)
        self.add_word(Special.START)
        self._w2i = {}
        self._i2w = []
        
        

# Should probably be moved to another file.
class Special:
    PADDING = '<P>'
    UNKNOWN = '<U>'
    START = '<S>'
    
    def all():
        return [Special.PADDING, Special.UNKNOWN, Special.START]
    
    def size():
        return len(Special.all())