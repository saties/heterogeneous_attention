
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset

from datasets import load_dataset

"""
1, MLM utils, find in transformers Huggingface
2, Dataset: (x) -> (x, s)
"""

class MultiSourceDataset(Dataset):
    
    def __init__(self, *sources):
        
        super().__init__()
        
        self.mapper, self.datasets = {}, {}
        
        for idt, d in enumerate(sources):
            
            args = d['args']
            name = args[0]
            kwargs = d['kwargs']
            
            self.mapper.update({name: idt})
            self.datasets.update({name: load_dataset(*args, **kwargs)})
        
        cumsum = {}
        for n, d in self.datasets.items():
            current_max = max(cumsum.values()) if len(cumsum) > 0 else 0
            cumsum.update({n: current_max + len(d)})
        self.cumsum = cumsum

    def __getitem__(self, index):
        
        lasts = 0
        
        for n, s in sorted(self.cumsum.items(), key=lambda t: t[1]):
            if index < s:
                source = n
                ids = index - lasts
                break
            else:
                lasts = s
            
        return self.datasets[n][ids], self.mapper[source]
    
    def __len__(self):
        return sum([len(d) for d in self.datasets.values()])
    
    
if __name__ == '__main__':
    
    data = MultiSourceDataset({'args': ('wikitext', 'wikitext-103-v1',), 'kwargs': {'split': 'train'}},
                              {'args': ('rotten_tomatoes',), 'kwargs': {'split': 'train'}})
    
    print(data[0])
    print(data[-1])