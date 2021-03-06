import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init
import torch.nn.utils.weight_norm as WeightNorm
import onmt 

from onmt.modules.Transformer.Layers import XavierLinear, MultiHeadAttention, FeedForward, PrePostProcessing, EncoderLayer, DecoderLayer
from onmt.modules.Bottle import Bottle

    
    
    
class ReinforcedStochasticDecoderLayer(DecoderLayer):
    """Wraps multi-head attentions and position-wise feed forward into one layer of decoder
    
    Args:
        h:       number of heads
        d_model: dimension of model
        p:       dropout probabolity 
        d_ff:    dimension of feed forward
        
    Params:
        multihead_tgt:  multi-head self attentions layer
        multihead_src:  multi-head encoder-decoder attentions layer        
        feedforward:    feed forward layer
    
    Input Shapes:
        query:    batch_size x len_query x d_model 
        key:      batch_size x len_key x d_model   
        value:    batch_size x len_key x d_model
        context:  batch_size x len_src x d_model
        mask_tgt: batch_size x len_query x len_key or broadcastable 
        mask_src: batch_size x len_query x len_src or broadcastable 
    
    Output Shapes:
        out:      batch_size x len_query x d_model
        coverage: batch_size x len_query x len_key
        
    """    
    
    def __init__(self, h, d_model, p, d_ff, attn_p=0.1):
        super().__init__(h, d_model, p, d_ff, attn_p)
        
    
    def forward(self, input, context, mask_tgt, mask_src, layer_mask=None, pad_mask_tgt=None, pad_mask_src=None, residual_dropout=0.0):
        
        """ Self attention layer 
            layernorm > attn > dropout > residual
        """
        
        """
            input is 'unnormalized' so the first preprocess layer is to normalize it before attention
            
            output (input after stacked with other outputs) is also unnormalized (to be normalized in the next layer)
            
            so if we skip the layer and propagate input forward:

        """
        coverage = None
        
        last_layer = input
        
        batch_size, input_length = input.size(1), input.size(0)
        
    
        query = self.preprocess_attn(input)
        
        self_context = query
        
        out, _ = self.multihead_tgt(query, self_context, self_context, mask_tgt, 
                                    query_mask=pad_mask_tgt, value_mask=pad_mask_tgt)
        
        # ~ if self.training:
            # ~ out = out / ( 1 - self.death_rate)
        
        input = self.postprocess_attn(out, input)
        

        """ Context Attention layer 
            layernorm > attn > dropout > residual
        """
        query = self.preprocess_src_attn(input, mask=pad_mask_tgt)
        out, coverage = self.multihead_src(query, context, context, mask_src, 
                                           query_mask=pad_mask_tgt, value_mask=pad_mask_src)
        
        if self.training:
            out = out / ( 1 - self.death_rate)
        
        input = self.postprocess_src_attn(out, input)
        
        """ Feed forward layer 
            layernorm > ffn > dropout > residual
        """
        out = self.feedforward(self.preprocess_ffn(input, mask=pad_mask_tgt), 
                                           mask=pad_mask_tgt)
        # During testing we scale the output to match its EXPECTATION of participation during training                                   
        # ~ if self.training:
            # ~ out = out / ( 1 - self.death_rate)
            
        input = self.postprocess_ffn(out, input)
        
        
        # if training: residual connection to the next layer
        if layer_mask is not None:
            input = layer_mask * input + ( 1 - layer_mask ) * last_layer
    
        return input, coverage
        
    # The step function should be identical to normal Transformer
        
