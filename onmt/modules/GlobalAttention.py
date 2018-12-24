#~ """
#~ Global attention takes a matrix and a query vector. It
#~ then computes a parameterized convex combination of the matrix
#~ based on the input query.
#~ 
#~ 
        #~ H_1 H_2 H_3 ... H_n
          #~ q   q   q       q
            #~ |  |   |       |
              #~ \ |   |      /
                      #~ .....
                  #~ \   |  /
                          #~ a
#~ 
#~ Constructs a unit mapping.
    #~ $$(H_1 + H_n, q) => (a)$$
    #~ Where H is of `batch x n x dim` and q is of `batch x dim`.
#~ 
    #~ The full def is  $$\tanh(W_2 [(softmax((W_1 q + b_1) H) H), q] + b_2)$$.:
#~ 
#~ """
#~ 
import math
import torch
import torch.nn as nn
from torch._C import dtype
from torch.autograd import Variable
import torch.nn.init as init
import torch.nn.utils.weight_norm as WeightNorm
import onmt 
import torch.nn.functional as F
from onmt.modules.Bottle import Bottle
from onmt.modules.StaticDropout import StaticDropout
from onmt.modules.Linear import XavierLinear as Linear
from onmt.modules.Linear import group_linear
from onmt.modules.MaxOut import MaxOut


#~ 
#~ 
class GlobalAttention(nn.Module):
    def __init__(self, dim):
        super(GlobalAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.linear_context = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax(dim=1)
        self.linear_out = nn.Linear(dim*2, dim, bias=False)
        self.linear_to_one = nn.Linear(dim, 1, bias=True)
        self.tanh = nn.Tanh()
        self.mlp_tanh = nn.Tanh()
        self.mask = None
        
        # For context gate
        self.linear_cg = nn.Linear(dim*2, dim, bias=True)
        self.sigmoid_cg = nn.Sigmoid()

    def applyMask(self, mask):
        self.mask = mask

    def forward(self, input, context, attn_mask=None):
        """
        input: batch x dim
        context: batch x sourceL x dim
        """
        bsize = context.size(0)
        seq_length = context.size(1)
        dim = context.size(2)
        
        # project the hidden state (query)
        targetT = self.linear_in(input).unsqueeze(1)  # batch x 1 x dim
        
        # project the context (keys and values)
        reshaped_ctx = context.contiguous().view(bsize * seq_length, dim)
        
        projected_ctx = self.linear_context(reshaped_ctx)
        
        projected_ctx = projected_ctx.view(bsize, seq_length, dim)
        
        # MLP attention model
        repeat = targetT.expand_as(projected_ctx)
        sum_query_ctx = repeat + projected_ctx 
        sum_query_ctx = sum_query_ctx.view(bsize * seq_length, dim)
        
        mlp_input = self.mlp_tanh(sum_query_ctx)
        mlp_output = self.linear_to_one(mlp_input)
        
        mlp_output = mlp_output.view(bsize, seq_length, 1)
        attn = mlp_output.squeeze(2) # bsize x seq_length

        # Get attention
        if attn_mask is not None:
            # if not reshape then a bug may happen at decoding 
            reshaped_attn = attn_mask.view(*attn.size())
            attn = attn.masked_fill(reshaped_attn, -float('inf'))
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL

        weightedContext = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        contextCombined = torch.cat((weightedContext, input), 1)
        
        #ContextGate
        contextGate = self.sigmoid_cg(self.linear_cg(contextCombined))
        inputGate = 1 - contextGate
        
        gatedContext = weightedContext * contextGate
        gatedInput = input * inputGate
        gatedContextCombined = torch.cat((gatedContext, gatedInput), 1)
        

        contextOutput = self.tanh(self.linear_out(gatedContextCombined))

        return contextOutput, attn





class MultiHeadAttention(nn.Module):
    """Applies multi-head attentions to inputs (query, key, value)
    Args:
        h:       number of heads
        d_model: dimension of model
        p:       dropout probabolity  
        
    Params:
        fc_query:  FC layer to project query, d_model x (h x d_head)
        fc_key:    FC layer to project key,   d_model x (h x d_head)
        fc_value:  FC layer to project value, d_model x (h x d_head)
        fc_concat: FC layer to concat and project multiheads, d_model x (h x d_head)
        
    Inputs Shapes: 
        query: batch_size x len_query x d_model 
        key:   batch_size x len_key x d_model   
        value: batch_size x len_key x d_model
        mask:  batch_size x len_query x len_key or broadcastable 
        
    Outputs Shapes:
        out:      batch_size x len_query x d_model
        coverage: batch_size x len_query x len_key
        
    """
    
    def __init__(self, h, d_model, attn_p=0.1, static=True, share=3):
        super(MultiHeadAttention, self).__init__()      
        self.h = h
        self.d = d_model
        self.share = share
        
        assert d_model % h == 0
        
        self.d_head = d_model//h
        self.fc_query = Bottle(Linear(d_model, h*self.d_head, bias=False))
        self.fc_key = Bottle(Linear(d_model, h*self.d_head, bias=False))
        self.fc_value = Bottle(Linear(d_model, h*self.d_head, bias=False))
        
        self.attention_out = onmt.Constants.attention_out
        self.fc_concat = Bottle(Linear(h*self.d_head, d_model, bias=False))

        self.sm = nn.Softmax(dim=-1)
        
        if static:
            self.attn_dropout = StaticDropout(attn_p)
        else:
            self.attn_dropout = nn.Dropout(attn_p)
    
    
        
    def forward(self, query, key, value, mask, query_mask=None, value_mask=None):

        len_query, b = query.size(0), query.size(1)
        len_key,  b_ = key.size(0), key.size(1)
        
        key_mask = value_mask
        
         # batch_size*h x len_query x d_head
        # project inputs to multi-heads
        if self.share == 1:
            shared_qkv = group_linear([self.fc_query.function.linear, self.fc_key.function.linear, self.fc_value.function.linear], query)
            proj_query, proj_key, proj_value = shared_qkv.chunk(3, dim=-1)
        elif self.share == 2:
            proj_query = self.fc_query(query) # batch_size x len_query x h*d_head
            shared_kv = group_linear([self.fc_key.function.linear, self.fc_value.function.linear], key)
            proj_key, proj_value = shared_kv.chunk(2, dim=-1)
        else:
            proj_query = self.fc_query(query, mask=query_mask)  
            proj_key   = self.fc_key(key, mask=key_mask)             # batch_size x len_key x h*d_head
            proj_value = self.fc_value(value, mask=value_mask)       # batch_size x len_key x h*d_head
        
        q, k, v = proj_query, proj_key, proj_value
        # prepare the shape for applying softmax
        q = q.contiguous().view(len_query, b*self.h, self.d_head).transpose(0, 1)
        k = k.contiguous().view(len_key,   b*self.h, self.d_head).transpose(0, 1)
        v = v.contiguous().view(len_key,   b*self.h, self.d_head).transpose(0, 1)
        
        q = q * (self.d_head**-0.5)
        
        # get dotproduct softmax attns for each head
        attns = torch.bmm(q, k.transpose(1,2))  # batch_size*h x len_query x len_key
        
        attns = attns.view(b, self.h, len_query, len_key) 
        mask_ = mask.unsqueeze(-3)
        # FP16 support: cast to float and back
        attns = attns.float().masked_fill_(mask_, -float('inf')).type_as(attns)
        attns = F.softmax(attns.float(), dim=-1).type_as(attns)
        # return mean attention from all heads as coverage 
        coverage = torch.mean(attns, dim=1) 
        attns = self.attn_dropout(attns)
        attns = attns.view(b*self.h, len_query, len_key)
        
        # apply attns on value
        out = torch.bmm(attns, v)      # batch_size*h x len_query x d_head
        out = out.transpose(0, 1).contiguous().view(len_query, b, self.d)
            
        out = self.fc_concat(out)
               
        return out, coverage
        
    def step(self, query, key, value, mask, query_mask=None, value_mask=None, buffer=None):
    
        len_query, b = query.size(0), query.size(1)
        len_key,  b_ = key.size(0), key.size(1)
        
        key_mask = value_mask
        
        # project inputs to multi-heads
        
        if self.share == 1:
            # proj_query = self.fc_query(query, mask=query_mask)   # batch_size*h x len_query x d_head
            # proj_key   = self.fc_key(key, mask=key_mask)             # batch_size x len_key x h*d_head
            # proj_value = self.fc_value(value, mask=value_mask)       # batch_size x len_key x h*d_head
            shared_qkv = group_linear([self.fc_query.function.linear, self.fc_key.function.linear, self.fc_value.function.linear], query)
            proj_query, proj_key, proj_value = shared_qkv.chunk(3, dim=-1)
            if buffer is not None and 'k' in buffer and 'v' in buffer:
                proj_key = torch.cat([buffer['k'], proj_key], dim=0) # time first
                buffer['k'] = proj_key
                proj_value = torch.cat([buffer['v'], proj_value], dim=0) # time first
                buffer['v'] = proj_value
                len_key,  b_ = proj_key.size(0), proj_key.size(1)
            else:
                if buffer is None:
                    buffer = dict()
                buffer['k'] = proj_key
                buffer['v'] = proj_value
        elif self.share == 2:
            proj_query = self.fc_query(query) # batch_size x len_query x h*d_head
            if buffer is not None and 'c_k' in buffer and 'c_v' in buffer:
                proj_key = buffer['c_k']
                proj_value = buffer['c_v']
            else:
                if buffer is None:
                    buffer = dict()
                shared_kv = group_linear([self.fc_key.function.linear, self.fc_value.function.linear], key)
                proj_key, proj_value = shared_kv.chunk(2, dim=-1)
                buffer['c_k'] = proj_key
                buffer['c_v'] = proj_value
        else:
            raise NotImplementedError
        
        q, k, v = proj_query, proj_key, proj_value
        
        # prepare the shape for applying softmax
        q = q.contiguous().view(len_query, b*self.h, self.d_head).transpose(0, 1)
        k = k.contiguous().view(len_key,   b*self.h, self.d_head).transpose(0, 1)
        v = v.contiguous().view(len_key,   b*self.h, self.d_head).transpose(0, 1)
        
        q = q * (self.d_head**-0.5)
        
        # get dotproduct softmax attns for each head
        attns = torch.bmm(q, k.transpose(1,2))  # batch_size*h x len_query x len_key
        
        attns = attns.view(b, self.h, len_query, len_key) 
        mask_ = mask.unsqueeze(-3)
        # FP16 support: cast to float and back
        attns = attns.float().masked_fill_(mask_, -float('inf')).type_as(attns)
        attns = F.softmax(attns.float(), dim=-1).type_as(attns)
        # return mean attention from all heads as coverage 
        coverage = torch.mean(attns, dim=1) 
        attns = self.attn_dropout(attns)
        attns = attns.view(b*self.h, len_query, len_key)
        
        # apply attns on value
        out = torch.bmm(attns, v)      # batch_size*h x len_query x d_head
        out = out.transpose(0, 1).contiguous().view(len_query, b, self.d)
            
        out = self.fc_concat(out)
       
        return out, coverage, buffer


class MultiHeadAttentionMemoryCompressed(nn.Module):
    """Applies multi-head attentions to inputs (query, key, value)
    Args:
        h:       number of heads
        d_model: dimension of model
        p:       dropout probabolity

    Params:
        fc_query:  FC layer to project query, d_model x (h x d_head)
        fc_key:    FC layer to project key,   d_model x (h x d_head)
        fc_value:  FC layer to project value, d_model x (h x d_head)
        fc_concat: FC layer to concat and project multiheads, d_model x (h x d_head)

    Inputs Shapes:
        query: batch_size x len_query x d_model
        key:   batch_size x len_key x d_model
        value: batch_size x len_key x d_model
        mask:  batch_size x len_query x len_key or broadcastable

    Outputs Shapes:
        out:      batch_size x len_query x d_model
        coverage: batch_size x len_query x len_key

    """

    def __init__(self, h, d_model, attn_p=0.1, static=True, share=3, compression_factor=2, compression_function=1):
        """

        :param h: number of heads
        :param d_model: dimension of embeddings
        :param attn_p:
        :param static:
        :param share:
        :param compression_factor: compression factor to reduce dimension of key and values (dmodel // compression_factor)
        """
        super(MultiHeadAttentionMemoryCompressed, self).__init__()
        self.h = h
        self.d = d_model
        self.share = share

        assert d_model % h == 0

        self.d_head = d_model // h
        #self.fc_query = Bottle(Linear(d_model, h * self.d_head, bias=False))
        self.fc_query = Bottle(Linear(d_model, h * self.d_head, bias=False))
        #self.fc_key = Bottle(Linear(d_model, h * self.d_head, bias=False))
        self.fc_key = Bottle(Linear((d_model // compression_factor), h * self.d_head, bias=False))
        #D.S: Compression of Inputs
        #self.fc_value = Bottle(Linear(d_model, h * self.d_head, bias=False))
        self.fc_value = Bottle(Linear((d_model // compression_factor), h * self.d_head, bias=False))
        # D.S: Compression of Inputs

        self.attention_out = onmt.Constants.attention_out
        self.fc_concat = Bottle(Linear(h * self.d_head, d_model, bias=False))

        #D.S: Added global compression factor
        self.compression_factor = compression_factor
        self.compression_function = compression_function

        self.sm = nn.Softmax(dim=-1)

        if static:
            self.attn_dropout = StaticDropout(attn_p)
        else:
            self.attn_dropout = nn.Dropout(attn_p)

    def forward(self, query, key, value, mask, query_mask=None, value_mask=None):


        len_query, b = query.size(0), query.size(1)
        len_key, b_ = key.size(0), key.size(1)

        compressed_key_length = len_key // self.compression_factor

        key_mask = value_mask


        #D.S: Do compression via different compression functions
        if self.compression_function == 1:
            #Use mean as compression function
            compressed_key = torch.mean(key, -1)
        elif self.compression_function == 2:
            #Use max as compression function
            compressed_key = torch.max(key, -1)

        #Linear transformation on original data for queries
        proj_query = self.fc_query(query)  # batch_size x len_query x h*d_head
        #Linear transformation on compressed data for key and values
        shared_kv = group_linear([self.fc_key.function.linear, self.fc_value.function.linear], key)
        proj_key, proj_value = shared_kv.chunk(2, dim=-1)



        # # batch_size*h x len_query x d_head
        # # project inputs to multi-heads
        # if self.share == 1:
        #     shared_qkv = group_linear(
        #         [self.fc_query.function.linear, self.fc_key.function.linear, self.fc_value.function.linear], query)
        #     proj_query, proj_key, proj_value = shared_qkv.chunk(3, dim=-1)
        # elif self.share == 2:
        #     proj_query = self.fc_query(query)  # batch_size x len_query x h*d_head
        #     shared_kv = group_linear([self.fc_key.function.linear, self.fc_value.function.linear], key)
        #     proj_key, proj_value = shared_kv.chunk(2, dim=-1)
        # else:
        #     #D.S: No shared weights used
        #     proj_query = self.fc_query(query, mask=query_mask)
        #     proj_key = self.fc_key(key, mask=key_mask)  # batch_size x len_key x h*d_head
        #     proj_value = self.fc_value(value, mask=value_mask)  # batch_size x len_key x h*d_head

        q, k, v = proj_query, proj_key, proj_value #D.S: Dimensions of q = (batch_size_words, batch_size_sents, dmodel  e)
        # prepare the shape for applying softmax
        q = q.contiguous().view(len_query, b * self.h, self.d_head).transpose(0, 1)
        k = k.contiguous().view(compressed_key_length, b * self.h, self.d_head).transpose(0, 1)
        v = v.contiguous().view(compressed_key_length, b * self.h, self.d_head).transpose(0, 1)

        q = q * (self.d_head ** -0.5)

        # get dotproduct softmax attns for each head
        attns = torch.bmm(q, k.transpose(1, 2))  # batch_size*h x len_query x len_key

        attns = attns.view(b, self.h, len_query, len_key)
        mask_ = mask.unsqueeze(-3)
        # FP16 support: cast to float and back
        attns = attns.float().masked_fill_(mask_, -float('inf')).type_as(attns)
        attns = F.softmax(attns.float(), dim=-1).type_as(attns)
        # return mean attention from all heads as coverage
        coverage = torch.mean(attns, dim=1)
        attns = self.attn_dropout(attns)
        attns = attns.view(b * self.h, len_query, len_key)

        # apply attns on value
        out = torch.bmm(attns, v)  # batch_size*h x len_query x d_head
        out = out.transpose(0, 1).contiguous().view(len_query, b, self.d)

        out = self.fc_concat(out)

        return out, coverage

    def step(self, query, key, value, mask, query_mask=None, value_mask=None, buffer=None):

        len_query, b = query.size(0), query.size(1)
        len_key, b_ = key.size(0), key.size(1)

        key_mask = value_mask

        # project inputs to multi-heads

        if self.share == 1:
            # proj_query = self.fc_query(query, mask=query_mask)   # batch_size*h x len_query x d_head
            # proj_key   = self.fc_key(key, mask=key_mask)             # batch_size x len_key x h*d_head
            # proj_value = self.fc_value(value, mask=value_mask)       # batch_size x len_key x h*d_head
            shared_qkv = group_linear(
                [self.fc_query.function.linear, self.fc_key.function.linear, self.fc_value.function.linear], query)
            proj_query, proj_key, proj_value = shared_qkv.chunk(3, dim=-1)
            if buffer is not None and 'k' in buffer and 'v' in buffer:
                proj_key = torch.cat([buffer['k'], proj_key], dim=0)  # time first
                buffer['k'] = proj_key
                proj_value = torch.cat([buffer['v'], proj_value], dim=0)  # time first
                buffer['v'] = proj_value
                len_key, b_ = proj_key.size(0), proj_key.size(1)
            else:
                if buffer is None:
                    buffer = dict()
                buffer['k'] = proj_key
                buffer['v'] = proj_value
        elif self.share == 2:
            proj_query = self.fc_query(query)  # batch_size x len_query x h*d_head
            if buffer is not None and 'c_k' in buffer and 'c_v' in buffer:
                proj_key = buffer['c_k']
                proj_value = buffer['c_v']
            else:
                if buffer is None:
                    buffer = dict()
                shared_kv = group_linear([self.fc_key.function.linear, self.fc_value.function.linear], key)
                proj_key, proj_value = shared_kv.chunk(2, dim=-1)
                buffer['c_k'] = proj_key
                buffer['c_v'] = proj_value
        else:
            raise NotImplementedError

        q, k, v = proj_query, proj_key, proj_value

        # prepare the shape for applying softmax
        q = q.contiguous().view(len_query, b * self.h, self.d_head).transpose(0, 1)
        k = k.contiguous().view(len_key, b * self.h, self.d_head).transpose(0, 1)
        v = v.contiguous().view(len_key, b * self.h, self.d_head).transpose(0, 1)

        q = q * (self.d_head ** -0.5)

        # get dotproduct softmax attns for each head
        attns = torch.bmm(q, k.transpose(1, 2))  # batch_size*h x len_query x len_key

        attns = attns.view(b, self.h, len_query, len_key)
        mask_ = mask.unsqueeze(-3)
        # FP16 support: cast to float and back
        attns = attns.float().masked_fill_(mask_, -float('inf')).type_as(attns)
        attns = F.softmax(attns.float(), dim=-1).type_as(attns)
        # return mean attention from all heads as coverage
        coverage = torch.mean(attns, dim=1)
        attns = self.attn_dropout(attns)
        attns = attns.view(b * self.h, len_query, len_key)

        # apply attns on value
        out = torch.bmm(attns, v)  # batch_size*h x len_query x d_head
        out = out.transpose(0, 1).contiguous().view(len_query, b, self.d)

        out = self.fc_concat(out)

        return out, coverage, buffer


class LocalAttention(nn.Module):
    """Applies multi-head attentions to inputs (query, key, value)
    Args:
        h:       number of heads
        d_model: dimension of model
        p:       dropout probabolity

    Params:
        fc_query:  FC layer to project query, d_model x (h x d_head)
        fc_key:    FC layer to project key,   d_model x (h x d_head)
        fc_value:  FC layer to project value, d_model x (h x d_head)
        fc_concat: FC layer to concat and project multiheads, d_model x (h x d_head)

    Inputs Shapes:
        query: batch_size x len_query x d_model
        key:   batch_size x len_key x d_model
        value: batch_size x len_key x d_model
        mask:  batch_size x len_query x len_key or broadcastable

    Outputs Shapes:
        out:      batch_size x len_query x d_model
        coverage: batch_size x len_query x len_key

    """

    #TODO: D.S: Remove parameter share or add possibility to share weights in linear transformation
    def __init__(self, h, d_model, block_size, attn_p=0.1, static=True, share=3):
        super(LocalAttention, self).__init__()
        self.h = h
        self.d = d_model
        self.share = share

        assert d_model % h == 0

        self.d_head = d_model // h
        self.fc_query = Bottle(Linear(d_model, h * self.d_head, bias=False))
        self.fc_key = Bottle(Linear(d_model, h * self.d_head, bias=False))
        self.fc_value = Bottle(Linear(d_model, h * self.d_head, bias=False))

        self.attention_out = onmt.Constants.attention_out
        self.fc_concat = Bottle(Linear(h * self.d_head, d_model, bias=False))

        #D.S: Add block length value, to set the size of blocks in which the input sequence is divided to.
        self.block_size = block_size

        self.sm = nn.Softmax(dim=-1)

        if static:
            self.attn_dropout = StaticDropout(attn_p)
        else:
            self.attn_dropout = nn.Dropout(attn_p)

    def forward(self, query, key, value, mask, step_num, prev_k, prev_v, query_mask=None, value_mask=None):

        len_query, b = query.size(0), query.size(1) #batch_size_words x batch_size_sentence
        len_key, b_ = key.size(0), key.size(1)

        key_mask = value_mask

        # batch_size*h x len_query x d_head
        # project inputs to multi-heads
        proj_query = self.fc_query(query)  # batch_size x len_query x h*d_head
        shared_kv = group_linear([self.fc_key.function.linear, self.fc_value.function.linear], key)
        proj_key, proj_value = shared_kv.chunk(2, dim=-1)

        q, k, v = proj_query, proj_key, proj_value
        # prepare the shape for applying softmax
        q = q.contiguous().view(len_query, b * self.h, self.d_head).transpose(0, 1)
        k = k.contiguous().view(len_key, b * self.h, self.d_head).transpose(0, 1)
        v = v.contiguous().view(len_key, b * self.h, self.d_head).transpose(0, 1)

        #D.S: Local attention starts here
        print('Step  Number: '+ str(step_num))
        print('block_size: '+ str(self.block_size))
        current_position = torch.cat((torch.zeros((1, step_num * self.block_size, 1)).byte(),
                       torch.ones((1, self.block_size, 1)).byte(),
                       torch.zeros((1,
                                    ((len_query - ((step_num + 1) * self.block_size)) if
                                     (len_query - ((step_num + 1) * self.block_size)) >= 0 else 0), 1)).byte()), dim=1)
        k = torch.cat([k, torch.zeros(k.shape[0],(prev_k.shape[1]-k.shape[1]),k.shape[2])], dim=1)
        v = torch.cat([v, torch.zeros(v.shape[0],(prev_v.shape[1]-v.shape[1]),v.shape[2])], dim=1)

        k = torch.where(current_position, k, prev_k)
        v = torch.where(current_position, v, prev_v)

        q = torch.cat([q, torch.zeros(q.shape[0], (prev_v.shape[1] - q.shape[1]), q.shape[2])], dim=1)
        q = q * (self.d_head ** -0.5)


        # get dotproduct softmax attns for each head
        attns = torch.bmm(q, k.transpose(1, 2))  # batch_size*h x len_query x len_key

        attns = attns.view(b, self.h, len_query, len_query)
        mask_ = mask.unsqueeze(-3)
        # FP16 support: cast to float and back
        attns = attns.float().masked_fill_(mask_, -float('inf')).type_as(attns)
        attns = F.softmax(attns.float(), dim=-1).type_as(attns)
        # return mean attention from all heads as coverage
        #coverage = torch.mean(attns, dim=1)
        attns = self.attn_dropout(attns)
        attns = attns.view(b * self.h, len_query, len_query)

        # apply attns on value
        out = torch.bmm(attns, v)  # batch_size*h x len_query x d_head
        out = out.transpose(0, 1).contiguous().view(len_query, b, self.d)

        out = self.fc_concat(out)

        return out, k, v

    def step(self, query, key, value, mask, query_mask=None, value_mask=None, buffer=None):

        len_query, b = query.size(0), query.size(1)
        len_key, b_ = key.size(0), key.size(1)

        key_mask = value_mask

        # project inputs to multi-heads

        if self.share == 1:
            # proj_query = self.fc_query(query, mask=query_mask)   # batch_size*h x len_query x d_head
            # proj_key   = self.fc_key(key, mask=key_mask)             # batch_size x len_key x h*d_head
            # proj_value = self.fc_value(value, mask=value_mask)       # batch_size x len_key x h*d_head
            shared_qkv = group_linear(
                [self.fc_query.function.linear, self.fc_key.function.linear, self.fc_value.function.linear], query)
            proj_query, proj_key, proj_value = shared_qkv.chunk(3, dim=-1)
            if buffer is not None and 'k' in buffer and 'v' in buffer:
                proj_key = torch.cat([buffer['k'], proj_key], dim=0)  # time first
                buffer['k'] = proj_key
                proj_value = torch.cat([buffer['v'], proj_value], dim=0)  # time first
                buffer['v'] = proj_value
                len_key, b_ = proj_key.size(0), proj_key.size(1)
            else:
                if buffer is None:
                    buffer = dict()
                buffer['k'] = proj_key
                buffer['v'] = proj_value
        elif self.share == 2:
            proj_query = self.fc_query(query)  # batch_size x len_query x h*d_head
            if buffer is not None and 'c_k' in buffer and 'c_v' in buffer:
                proj_key = buffer['c_k']
                proj_value = buffer['c_v']
            else:
                if buffer is None:
                    buffer = dict()
                shared_kv = group_linear([self.fc_key.function.linear, self.fc_value.function.linear], key)
                proj_key, proj_value = shared_kv.chunk(2, dim=-1)
                buffer['c_k'] = proj_key
                buffer['c_v'] = proj_value
        else:
            raise NotImplementedError

        q, k, v = proj_query, proj_key, proj_value

        # prepare the shape for applying softmax
        q = q.contiguous().view(len_query, b * self.h, self.d_head).transpose(0, 1)
        k = k.contiguous().view(len_key, b * self.h, self.d_head).transpose(0, 1)
        v = v.contiguous().view(len_key, b * self.h, self.d_head).transpose(0, 1)

        q = q * (self.d_head ** -0.5)

        # get dotproduct softmax attns for each head
        attns = torch.bmm(q, k.transpose(1, 2))  # batch_size*h x len_query x len_key

        attns = attns.view(b, self.h, len_query, len_key)
        mask_ = mask.unsqueeze(-3)
        # FP16 support: cast to float and back
        attns = attns.float().masked_fill_(mask_, -float('inf')).type_as(attns)
        attns = F.softmax(attns.float(), dim=-1).type_as(attns)
        # return mean attention from all heads as coverage
        coverage = torch.mean(attns, dim=1)
        attns = self.attn_dropout(attns)
        attns = attns.view(b * self.h, len_query, len_key)

        # apply attns on value
        out = torch.bmm(attns, v)  # batch_size*h x len_query x d_head
        out = out.transpose(0, 1).contiguous().view(len_query, b, self.d)

        out = self.fc_concat(out)

        return out, coverage, buffer

