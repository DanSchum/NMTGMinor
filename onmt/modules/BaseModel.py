import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import onmt, math


#~ from onmt.modules.Transformer.Layers import XavierLinear

class Generator(nn.Module):

    def __init__(self, hidden_size, output_size):
        
        super(Generator, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        #~ self.linear = onmt.modules.Transformer.Layers.XavierLinear(hidden_size, output_size)
        self.linear = nn.Linear(hidden_size, output_size)
        
        stdv = 1. / math.sqrt(self.linear.weight.size(1))
        
        torch.nn.init.uniform_(self.linear.weight, -stdv, stdv)
        
        self.linear.bias.data.zero_()
        
        
        
    def forward(self, input, log_softmax=True):
        #D.S: Input has dim: (batch_size_sentences x embedding_size)
        # added float to the end 
        # print(input.size())
        logits = self.linear(input).float() 

        #D.S: output has dim: (batch_size_sentences x Target_vocab)

        if log_softmax:
            output = F.log_softmax(logits, dim=-1)
        else:
            output = logits
        return output
        

class NMTModel(nn.Module):

    def __init__(self, encoder, decoder, generator=None):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator        
        
    def tie_weights(self):
        assert self.generator is not None, "The generator needs to be created before sharing weights"
        self.generator.linear.weight = self.decoder.word_lut.weight
        
    
    def share_enc_dec_embedding(self):
        self.encoder.word_lut.weight = self.decoder.word_lut.weight
        
    def mark_pretrained(self):
        
        self.encoder.mark_pretrained()
        self.decoder.mark_pretrained()
        
    def load_state_dict(self, state_dict, strict=True):
        
        def condition(param_name):
            
            # we don't need to load the position encoding (not weight)
            if 'positional_encoder' in param_name:
                return False
            if 'time_transformer' in param_name and self.encoder.time == 'positional_encoding':
                return False
            # we don't need to load the decoder mask (not weight)
            if param_name == 'decoder.mask':
                return False
            
            return True
        
        filtered = {k: v for k, v in state_dict.items() if condition(k)}
        
        model_dict = self.state_dict()
        
        # add missing keys to the filtered state dict
        for k,v in model_dict.items():
            if k not in filtered:
                filtered[k] = v
        super().load_state_dict(filtered)   

        
        
    


class Reconstructor(nn.Module):
    
    def __init__(self, decoder, generator=None):
        super(Reconstructor, self).__init__()
        self.decoder = decoder
        self.generator = generator        
    

class DecoderState(object):
    """Interface for grouping together the current state of a recurrent
    decoder. In the simplest case just represents the hidden state of
    the model.  But can also be used for implementing various forms of
    input_feeding and non-recurrent models.
    Modules need to implement this to utilize beam search decoding.
    """
    
    
