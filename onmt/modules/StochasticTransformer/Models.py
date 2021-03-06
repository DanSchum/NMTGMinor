import numpy as np
import torch, math
import torch.nn as nn
from onmt.modules.Transformer.Layers import PositionalEncoding
from onmt.modules.Transformer.Layers import EncoderLayer, DecoderLayer
from onmt.modules.StochasticTransformer.Layers import StochasticEncoderLayer, StochasticDecoderLayer
from onmt.modules.Transformer.Models import TransformerEncoder, TransformerDecoder
from onmt.modules.BaseModel import NMTModel, Reconstructor
import onmt
from onmt.modules.WordDrop import embedded_dropout
from onmt.modules.Checkpoint import checkpoint

from onmt.modules.Transformer.Layers import XavierLinear, MultiHeadAttention, FeedForward, PrePostProcessing


def custom_layer(module):
    def custom_forward(*args):
        output = module(*args)
        return output
    return custom_forward
    
def expected_length(length, death_rate, death_type):
    
    e_length = 0
    
    for l in range(length):
        
        if death_type == 'linear_decay':
            survival_rate = 1.0 - (l+1)/length*death_rate
        elif death_type == 'uniform':
            survival_rate = 1.0 - death_rate
        else:
            raise NotImplementedError
        
        e_length += survival_rate
        
    return e_length

class StochasticTransformerEncoder(TransformerEncoder):
    """Encoder in 'Attention is all you need'
    
    Args:
        opt: list of options ( see train.py )
        dicts : dictionary (for source language)
        
    """
    
    def __init__(self, opt, dicts, positional_encoder):
    
        self.death_rate = opt.death_rate
        self.death_type = opt.death_type
        
        # build_modules will be called from the inherited constructor
        super(StochasticTransformerEncoder, self).__init__(opt, dicts, positional_encoder)
            
        e_length = expected_length(self.layers, self.death_rate, self.death_type)    
        
        print("Stochastic Encoder with %.2f expected layers" % e_length) 
       
    def build_modules(self):
        
        self.layer_modules = nn.ModuleList()

        for l in range(self.layers):
            
            # linearly decay the death rate
            if self.death_type == 'linear_decay':
                death_r = ( l + 1 ) / self.layers * self.death_rate
            elif self.death_type == 'uniform':
                death_r = self.death_rate
            else:
                raise NotImplementedError("Death type incorrectly specified")
            
            block = StochasticEncoderLayer(self.n_heads, self.model_size, self.dropout, self.inner_size, self.attn_dropout, death_rate=death_r)
            
            self.layer_modules.append(block)
            
    def sample(self, input, **kwargs):
        """
        Inputs Shapes: 
            input: batch_size x len_src (wanna tranpose)
        
        Outputs Shapes:
            out: batch_size x len_src x d_model
            mask_src 
            
        """

        """ Embedding: batch_size x len_src x d_model """
        emb = self.word_lut(input)
        
        """ Scale the emb by sqrt(d_model) """
        
        emb = emb * math.sqrt(self.model_size)
            
        """ Adding positional encoding """
        emb = self.time_transformer(emb)
        
        emb = self.preprocess_layer(emb)
        
        mask_src = input.eq(onmt.Constants.PAD).unsqueeze(1) # batch_size x len_src x 1 for broadcasting
                
        context = emb.transpose(0, 1).contiguous()
        
        for i, layer in enumerate(self.layer_modules):
            
            context = layer.sample(context, mask_src)      # batch_size x len_src x d_model
            
        
        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.    
        context = self.postprocess_layer(context)
             
        return context, mask_src   

class StochasticTransformerDecoder(TransformerDecoder):
    """Encoder in 'Attention is all you need'
    
    Args:
        opt
        dicts 
        
        
    """
    
    def __init__(self, opt, dicts, positional_encoder):
    
        self.death_rate = opt.death_rate
        self.death_type = opt.death_type
        
        # build_modules will be called from the inherited constructor
        super(StochasticTransformerDecoder, self).__init__(opt, dicts, positional_encoder)
        
        e_length = expected_length(self.layers, self.death_rate, self.death_type)    
        
        print("Stochastic Decoder with %.2f expected layers" % e_length) 
        
    
    def build_modules(self):
        
        self.layer_modules = nn.ModuleList()
        
        for l in range(self.layers):
            
            # linearly decay the death rate
            if self.death_type == 'linear_decay':
                death_r = ( l + 1 ) / self.layers * self.death_rate
            elif self.death_type == 'uniform':
                death_r = self.death_rate
            else:
                raise NotImplementedError("Death type incorrectly specified")
            
            block = StochasticDecoderLayer(self.n_heads, self.model_size, self.dropout, self.inner_size, self.attn_dropout, death_rate=death_r)
            
            self.layer_modules.append(block)
            
    def step_sample(self, input, decoder_state):
        """
        Inputs Shapes: 
            input: (Variable) batch_size x len_tgt (wanna tranpose)
            context: (Variable) batch_size x len_src x d_model
            mask_src (Tensor) batch_size x len_src
            buffer (List of tensors) List of batch_size * len_tgt-1 * d_model for self-attention recomputing
        Outputs Shapes:
            out: batch_size x len_tgt x d_model
            coverage: batch_size x len_tgt x len_src
            
        """
        context = decoder_state.context
        buffers = decoder_state.attention_buffers
        src = decoder_state.src.transpose(0, 1)
        
        if decoder_state.input_seq is None:
            decoder_state.input_seq = input
        else:
            # concatenate the last input to the previous input sequence
            decoder_state.input_seq = torch.cat([decoder_state.input_seq, input], 0)
        input = decoder_state.input_seq.transpose(0, 1)
        input_ = input[:,-1].unsqueeze(1)
        
        
        output_buffer = list()
            
        batch_size = input_.size(0)
        
        """ Embedding: batch_size x 1 x d_model """
        emb = self.word_lut(input_)
       
        
        if self.time == 'positional_encoding':
            emb = emb * math.sqrt(self.model_size)
        """ Adding positional encoding """
        if self.time == 'positional_encoding':
            emb = self.time_transformer(emb, t=input.size(1))
        else:
            prev_h = buffer[0] if buffer is None else None
            emb = self.time_transformer(emb, prev_h)
            buffer[0] = emb[1]
            
        if isinstance(emb, tuple):
            emb = emb[0]
        # emb should be batch_size x 1 x dim
            
        # Preprocess layer: adding dropout
        emb = self.preprocess_layer(emb)
        
        emb = emb.transpose(0, 1)
        
        # batch_size x 1 x len_src
        mask_src = src.data.eq(onmt.Constants.PAD).unsqueeze(1)
        
        
        len_tgt = input.size(1)
        mask_tgt = input.data.eq(onmt.Constants.PAD).unsqueeze(1) + self.mask[:len_tgt, :len_tgt]
        mask_tgt = torch.gt(mask_tgt, 0)
        mask_tgt = mask_tgt[:, -1, :].unsqueeze(1)
                
        output = emb.contiguous()
        
        setup = list()
    
        for i, layer in enumerate(self.layer_modules):
            
            buffer = buffers[i] if i in buffers else None
            assert(output.size(0) == 1)
            output, coverage, buffer, coin = layer.step_sample(output, context, mask_tgt, mask_src, buffer=buffer) # batch_size x len_src x d_model
            
            decoder_state._update_attention_buffer(buffer, i)
            
            if coin == True:
                setup.append(i)
        
        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.    
        output = self.postprocess_layer(output)

        return output, coverage, setup
            
        
        
class StochasticTransformer(NMTModel):
    """Main model in 'Attention is all you need' """
    
        
    def forward(self, input):
        """
        Inputs Shapes: 
            src: len_src x batch_size
            tgt: len_tgt x batch_size
        
        Outputs Shapes:
            out:      batch_size*len_tgt x model_size
            
            
        """
        src = input[0]
        tgt = input[1][:-1]  # exclude last target from inputs
        
        src = src.transpose(0, 1) # transpose to have batch first
        tgt = tgt.transpose(0, 1)
        
        context, src_mask = self.encoder(src)
        
        output, coverage = self.decoder(tgt, context, src)
        
        output = output.transpose(0, 1) # transpose to have time first, like RNN models
        
        return output


    def create_decoder_state(self, src, context, beamSize=1):
        
        from onmt.modules.Transformer.Models import TransformerDecodingState
        
        decoder_state = TransformerDecodingState(src, context, beamSize=beamSize)
        return decoder_state
