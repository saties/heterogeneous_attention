from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.modeling_bert import BertEncoder
from transformers.models.bert.modeling_bert import BertLayer
from transformers.models.bert.modeling_bert import BertAttention
from transformers.models.bert.modeling_bert import BertSelfOutput

from .layer import NewSelfAttention

from torch import nn


class NewBertAttention(BertAttention):
    
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config, position_embedding_type)
        self.self = NewSelfAttention(config, position_embedding_type=position_embedding_type)
        
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        else:
            raise NotImplementedError("Not implemented.")
    

class NewBertLayer(BertLayer):
    
    def __init__(self, config):
        super().__init__(config)
        self.attention = NewBertAttention(config)
        
    
class NewBertEncoder(BertEncoder):
    
    def __init__(self, config):        
        super().__init__(config)
        self.layer = nn.ModuleList([NewBertLayer(config) for _ in range(config.num_hidden_layers)])


class NewBertModel(BertModel):
    
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.encoder = NewBertEncoder(config)
    
    