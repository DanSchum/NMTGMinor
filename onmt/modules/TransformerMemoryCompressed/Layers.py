import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init
import torch.nn.utils.weight_norm as WeightNorm
import onmt 
import torch.nn.functional as F
from onmt.modules.Bottle import Bottle
from onmt.modules.StaticDropout import StaticDropout
from onmt.modules.Linear import XavierLinear, group_linear, FeedForward
from onmt.modules.MaxOut import MaxOut
from onmt.modules.GlobalAttention import MultiHeadAttention, MultiHeadAttentionMemoryCompressed, LocalAttention
from onmt.modules.PrePostProcessing import PrePostProcessing

        
Linear = XavierLinear

def variational_dropout(input, p, training=False):
    """Applies Variational Dropout (query, key, value)
    Inputs:
        input: Variable - batch_size * time_steps * hidden
    """
    if training:
        bsize = input.size(0)
        hsize = input.size(2)
        
        # create a mask for one time step
        mask = Variable(input.data.new(bsize, hsize).bernoulli_(1 - p).div_(1 - p), requires_grad=False)
        
        # then expand it to all time steps 
        mask = mask.unsqueeze(1).expand_as(input)
        output = input * mask
        return output
    # if eval then return the input
    return input

    

    
class EncoderLayer(nn.Module):
    """Wraps multi-head attentions and position-wise feed forward into one encoder layer
    
    Args:
        h:       number of heads
        d_model: dimension of model
        p:       dropout probabolity 
        d_ff:    dimension of feed forward
        
    Params:
        multihead:    multi-head attentions layer
        feedforward:  feed forward layer
    
    Input Shapes:
        query: batch_size x len_query x d_model 
        key:   batch_size x len_key x d_model   
        value: batch_size x len_key x d_model
        mask:  batch_size x len_query x len_key or broadcastable 
    
    Output Shapes:
        out: batch_size x len_query x d_model
    """
    
    def __init__(self, h, d_model, p, d_ff, attn_p=0.1, residual_p=0.1, version=1.0):
        super(EncoderLayer, self).__init__()
        self.version = version
        
        self.preprocess_attn = PrePostProcessing(d_model, 0.0, sequence='n')
        self.postprocess_attn = PrePostProcessing(d_model, residual_p, sequence='da', static=onmt.Constants.static)
        self.preprocess_ffn = PrePostProcessing(d_model, 0.0, sequence='n')
        self.postprocess_ffn = PrePostProcessing(d_model, residual_p, sequence='da', static=onmt.Constants.static)
        self.multihead = MultiHeadAttention(h, d_model, attn_p=attn_p, static=onmt.Constants.static, share=1)
        
        if onmt.Constants.activation_layer == 'linear_relu_linear':
            ff_p = p
            feedforward = FeedForward(d_model, d_ff, ff_p,static=onmt.Constants.static)
        elif onmt.Constants.activation_layer == 'maxout':
            k = int(math.ceil(d_ff / d_model))
            feedforward = MaxOut(d_model, d_model, k)
        self.feedforward = Bottle(feedforward)
            
    def forward(self, input, attn_mask, pad_mask=None):
        pad_mask = None
        query = self.preprocess_attn(input)
        out, _ = self.multihead(query, query, query, attn_mask)
        input = self.postprocess_attn(out, input)
        
        """ Feed forward layer 
            layernorm > ffn > dropout > residual
        """
        out = self.feedforward(self.preprocess_ffn(input))
        input = self.postprocess_ffn(out, input)
        
        return input
    
    
    
class DecoderLayer(nn.Module):
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
    
    def __init__(self, h, d_model, p, d_ff, attn_p=0.1, residual_p=0.1, version=1.0):
        super(DecoderLayer, self).__init__()
        self.version = version
        
        self.preprocess_attn = PrePostProcessing(d_model, p, sequence='n')
        self.postprocess_attn = PrePostProcessing(d_model, residual_p, sequence='da', static=onmt.Constants.static)
        
        self.preprocess_src_attn = PrePostProcessing(d_model, p, sequence='n')
        self.postprocess_src_attn = PrePostProcessing(d_model, residual_p, sequence='da', static=onmt.Constants.static)
        
        self.preprocess_ffn = PrePostProcessing(d_model, p, sequence='n')
        self.postprocess_ffn = PrePostProcessing(d_model, residual_p, sequence='da', static=onmt.Constants.static)
        
        
        self.multihead_tgt = MultiHeadAttention(h, d_model, attn_p=attn_p, static=onmt.Constants.static, share=1)
        self.multihead_src = MultiHeadAttention(h, d_model, attn_p=attn_p, static=onmt.Constants.static, share=2)
        
        if onmt.Constants.activation_layer == 'linear_relu_linear':
            ff_p = p
            feedforward = FeedForward(d_model, d_ff, ff_p, static=onmt.Constants.static)
        elif onmt.Constants.activation_layer == 'maxout':
            k = int(math.ceil(d_ff / d_model))
            feedforward = MaxOut(d_model, d_model, k)
        self.feedforward = Bottle(feedforward)
    
    def forward(self, input, context, mask_tgt, mask_src, pad_mask_tgt=None, pad_mask_src=None, residual_dropout=0.0):
        
        """ Self attention layer 
            layernorm > attn > dropout > residual
        """
        
        # input and context should be time first ?
        
        query = self.preprocess_attn(input)
        
        self_context = query

        out, _ = self.multihead_tgt(query, self_context, self_context, mask_tgt)
        
        if residual_dropout > 0:
            input_ = F.dropout(input, residual_dropout, self.training, False)
            input = self.postprocess_attn(out, input_)
        else:
            input = self.postprocess_attn(out, input)

        """ Context Attention layer 
            layernorm > attn > dropout > residual
        """
        
        query = self.preprocess_src_attn(input)
        out, coverage = self.multihead_src(query, context, context, mask_src)
        input = self.postprocess_src_attn(out, input)
        
        """ Feed forward layer 
            layernorm > ffn > dropout > residual
        """
        out = self.feedforward(self.preprocess_ffn(input))
        input = self.postprocess_ffn(out, input)
    
        return input, coverage
        
    def step(self, input, context, mask_tgt, mask_src, pad_mask_tgt=None, pad_mask_src=None, buffer=None):
        """ Self attention layer 
            layernorm > attn > dropout > residual
        """
        
        query = self.preprocess_attn(input)
        
        out, _, buffer = self.multihead_tgt.step(query, query, query, mask_tgt, buffer=buffer)
                                   
        

        input = self.postprocess_attn(out, input)
        
        """ Context Attention layer 
            layernorm > attn > dropout > residual
        """
        
        query = self.preprocess_src_attn(input)
        out, coverage, buffer = self.multihead_src.step(query, context, context, mask_src, buffer=buffer)
                                           
        input = self.postprocess_src_attn(out, input)
        
        """ Feed forward layer 
            layernorm > ffn > dropout > residual
        """
        out = self.feedforward(self.preprocess_ffn(input))
                                           
        input = self.postprocess_ffn(out, input)
        
        return input, coverage, buffer


class PositionalEncoding(nn.Module):
    """Adds positional embeddings to standard word embeddings 
    This matches the original TensorFlow implementation at https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py.
    
    Args:
        d_model: dimension of model
        p:       dropout probability  
        len_max: max seq length for pre-calculated positional embeddings
        
    Inputs Shapes: 
        word_emb: batch_size x len_seq x d_model 
        
    Outputs Shapes:
        out:   batch_size x len_seq x d_model
        
    """
    def __init__(self, d_model, cuda, p=0, len_max=512):
        # save a fixed positional embedding matrix up to len_max,
        # so that no need to recreate it everytime
        super(PositionalEncoding, self).__init__()
        self.len_max=len_max
        self.d_model = d_model
        self.data_type = None
        self.cuda = cuda
        
        self.renew(len_max)
        
        self.p = p
        
    
    def renew(self, new_max_len):
        ## detele the old variable to avoid Pytorch's error when register new buffer
        if hasattr(self, 'pos_emb'):
            del self.pos_emb
        position = torch.arange(0,new_max_len).float()  
        
                            
        num_timescales = self.d_model // 2
        log_timescale_increment = math.log(10000) / (num_timescales-1)
        inv_timescales = torch.exp(torch.arange(0, num_timescales).float() * -log_timescale_increment)
        scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
        pos_emb = torch.cat((torch.sin(scaled_time), torch.cos(scaled_time)), 1)
        
        if self.data_type is not None:
            pos_emb.type(self.data_type)
        # wrap in a buffer so that model can be moved to GPU
        self.register_buffer('pos_emb', pos_emb)        
        self.data_type = self.pos_emb.type()
        self.len_max = new_max_len

        
    def forward(self, word_emb, t=None):
    
        len_seq = t if t else word_emb.size(1)
        
        if len_seq > self.len_max:
            self.renew(len_seq)

        if self.cuda and not onmt.Constants.memoryCompressionActivated:
            print('Transfered word_emb to cuda')
            word_emb = word_emb.cuda()
            self.pos_emb = self.pos_emb.cuda()

        if word_emb.size(1) == len_seq:
            out = word_emb + Variable(self.pos_emb[:len_seq, :], requires_grad=False)
        else:
            # out = word_emb + Variable(self.pos_emb[:len_seq, :][-1, :], requires_grad=False)
            time_emb = Variable(self.pos_emb[len_seq-1, :], requires_grad=False) # 1 x dim
            # out should have size bs x 1 x dim
            out = word_emb + time_emb.unsqueeze(0).repeat(word_emb.size(0), 1, 1)
        return out


class EncoderLayerLocalAttention(nn.Module):
    """Wraps multi-head attentions and position-wise feed forward into one encoder layer

    Args:
        h:       number of heads
        d_model: dimension of model
        p:       dropout probabolity
        d_ff:    dimension of feed forward

    Params:
        multihead:    multi-head attentions layer
        feedforward:  feed forward layer

    Input Shapes:
        query: batch_size x len_query x d_model
        key:   batch_size x len_key x d_model
        value: batch_size x len_key x d_model
        mask:  batch_size x len_query x len_key or broadcastable

    Output Shapes:
        out: batch_size x len_query x d_model
    """

    def __init__(self, h, d_model, p, d_ff, block_size, cuda, attn_p=0.1, residual_p=0.1, version=1.0):
        super(EncoderLayerLocalAttention, self).__init__()
        self.version = version
        self.cuda = cuda

        self.preprocess_attn = PrePostProcessing(d_model, 0.0, sequence='n')
        self.postprocess_attn = PrePostProcessing(d_model, residual_p, sequence='da', static=onmt.Constants.static)
        self.preprocess_ffn = PrePostProcessing(d_model, 0.0, sequence='n')
        self.postprocess_ffn = PrePostProcessing(d_model, residual_p, sequence='da', static=onmt.Constants.static)
        self.multihead = LocalAttention(h, d_model, block_size, self.cuda, attn_p=attn_p, static=onmt.Constants.static, share=1)

        if onmt.Constants.activation_layer == 'linear_relu_linear':
            ff_p = p
            feedforward = FeedForward(d_model, d_ff, ff_p, static=onmt.Constants.static)
        elif onmt.Constants.activation_layer == 'maxout':
            k = int(math.ceil(d_ff / d_model))
            feedforward = MaxOut(d_model, d_model, k)
        self.feedforward = Bottle(feedforward)

    def forward(self, query, split, attn_mask, step_num, prev_k, prev_v, pad_mask=None):
        '''

        :param query: Is the context value from Models implementation (Dim: Batch_Size_Words x Batch_Size_Sentence x Embedding_Size)
        :param split: Contains the split values, which iterate through the src embedding (Dim: Block_Size x Batch_Size_Sentence x Embedding_Size)
        :param attn_mask:
        :param step_num:
        :param prev_k:
        :param prev_v:
        :param pad_mask:
        :return:
        '''
        pad_mask = None
        query = self.preprocess_attn(query)
        split = self.preprocess_attn(split)
        out, k, v = self.multihead(query, split, split, attn_mask, step_num, prev_k, prev_v)
        #Out: Contains Context (Dim: Batch_Size_Words x Batch_Size_Sentence x Embedding_Size)
        #k & v: Contains k & v values including all last splits (Dim: heads x Batch_Size_Words x (Embedding_Size / heads))
        #k & v: At Dim (1) - Batch_Size_Words: The Values are added from split, concatenated with the previous values from previous splits
        input = self.postprocess_attn(out, query)

        """ Feed forward layer 
            layernorm > ffn > dropout > residual
        """
        out = self.feedforward(self.preprocess_ffn(input))
        input = self.postprocess_ffn(out, input)

        return input, k, v

class EncoderLayerMemoryCompressed(nn.Module):
    """Wraps multi-head attentions and position-wise feed forward into one encoder layer

    Args:
        h:       number of heads
        d_model: dimension of model
        p:       dropout probabolity
        d_ff:    dimension of feed forward

    Params:
        multihead:    multi-head attentions layer
        feedforward:  feed forward layer

    Input Shapes:
        query: batch_size x len_query x d_model
        key:   batch_size x len_key x d_model
        value: batch_size x len_key x d_model
        mask:  batch_size x len_query x len_key or broadcastable

    Output Shapes:
        out: batch_size x len_query x d_model
    """

    def __init__(self, h, d_model, p, d_ff, attn_p=0.1, residual_p=0.1, version=1.0):
        super(EncoderLayerMemoryCompressed, self).__init__()
        self.version = version

        self.preprocess_attn = PrePostProcessing(d_model, 0.0, sequence='n')
        self.postprocess_attn = PrePostProcessing(d_model, residual_p, sequence='da', static=onmt.Constants.static)
        self.preprocess_ffn = PrePostProcessing(d_model, 0.0, sequence='n')
        self.postprocess_ffn = PrePostProcessing(d_model, residual_p, sequence='da', static=onmt.Constants.static)
        self.multihead = MultiHeadAttentionMemoryCompressed(
            h, d_model, attn_p=attn_p, static=onmt.Constants.static, share=1, compression_factor=2, compression_function=1)

        if onmt.Constants.activation_layer == 'linear_relu_linear':
            ff_p = p
            feedforward = FeedForward(d_model, d_ff, ff_p, static=onmt.Constants.static)
        elif onmt.Constants.activation_layer == 'maxout':
            k = int(math.ceil(d_ff / d_model))
            feedforward = MaxOut(d_model, d_model, k)
        self.feedforward = Bottle(feedforward)

    def forward(self, input, attn_mask, pad_mask=None):
        pad_mask = None
        query = self.preprocess_attn(input)
        out, _ = self.multihead(query, query, query, attn_mask)
        input = self.postprocess_attn(out, input)

        """ Feed forward layer 
            layernorm > ffn > dropout > residual
        """
        out = self.feedforward(self.preprocess_ffn(input))
        input = self.postprocess_ffn(out, input)

        return input


class DecoderLayerLocalAttention(nn.Module):
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

    def __init__(self, h, d_model, p, d_ff, block_size, cuda, attn_p=0.1, residual_p=0.1, version=1.0):
        super(DecoderLayerLocalAttention, self).__init__()
        self.version = version
        self.cuda = cuda

        self.preprocess_attn = PrePostProcessing(d_model, p, sequence='n')
        self.postprocess_attn = PrePostProcessing(d_model, residual_p, sequence='da', static=onmt.Constants.static)

        self.preprocess_src_attn = PrePostProcessing(d_model, p, sequence='n')
        self.postprocess_src_attn = PrePostProcessing(d_model, residual_p, sequence='da', static=onmt.Constants.static)

        self.preprocess_ffn = PrePostProcessing(d_model, p, sequence='n')
        self.postprocess_ffn = PrePostProcessing(d_model, residual_p, sequence='da', static=onmt.Constants.static)

        self.multihead_tgt = MultiHeadAttention(h, d_model, attn_p=attn_p, static=onmt.Constants.static, share=1)
        self.multihead_src = LocalAttention(h, d_model, block_size, self.cuda, attn_p=attn_p, static=onmt.Constants.static, share=2)

        if onmt.Constants.activation_layer == 'linear_relu_linear':
            ff_p = p
            feedforward = FeedForward(d_model, d_ff, ff_p, static=onmt.Constants.static)
        elif onmt.Constants.activation_layer == 'maxout':
            k = int(math.ceil(d_ff / d_model))
            feedforward = MaxOut(d_model, d_model, k)
        self.feedforward = Bottle(feedforward)

    def forward(self, input, context, mask_tgt, mask_src, step_num, prev_k, prev_v, pad_mask_tgt=None, pad_mask_src=None, residual_dropout=0.0):

        """ Self attention layer
            layernorm > attn > dropout > residual
        """

        # input and context should be time first ?

        query = self.preprocess_attn(input)

        self_context = query

        #D.S. Set to local attention layer
        out, coverage = self.multihead_tgt(query, self_context, self_context, mask_tgt)

        if residual_dropout > 0:
            input_ = F.dropout(input, residual_dropout, self.training, False)
            input = self.postprocess_attn(out, input_)
        else:
            input = self.postprocess_attn(out, input)

        """ Context Attention layer 
            layernorm > attn > dropout > residual
        """

        query = self.preprocess_src_attn(input)
        out, k, v = self.multihead_src(query, context, context, mask_src, step_num, prev_k, prev_v)
        input = self.postprocess_src_attn(out, input)

        """ Feed forward layer 
            layernorm > ffn > dropout > residual
        """
        out = self.feedforward(self.preprocess_ffn(input))
        input = self.postprocess_ffn(out, input)

        return input, k, v

    def step(self, input, context, mask_tgt, mask_src, pad_mask_tgt=None, pad_mask_src=None, buffer=None):
        """ Self attention layer
            layernorm > attn > dropout > residual
        """

        #TODO: D.S: What is this doing here?

        query = self.preprocess_attn(input)

        out, _, buffer = self.multihead_tgt.step(query, query, query, mask_tgt, buffer=buffer)

        input = self.postprocess_attn(out, input)

        """ Context Attention layer 
            layernorm > attn > dropout > residual
        """

        query = self.preprocess_src_attn(input)
        out, coverage, buffer = self.multihead_src.step(query, context, context, mask_src, buffer=buffer)

        input = self.postprocess_src_attn(out, input)

        """ Feed forward layer 
            layernorm > ffn > dropout > residual
        """
        out = self.feedforward(self.preprocess_ffn(input))

        input = self.postprocess_ffn(out, input)

        return input, coverage, buffer
