import numpy as np
import torch, math
import torch.nn as nn
from onmt.modules.TransformerMemoryCompressed.Layers import EncoderLayer, DecoderLayer, PositionalEncoding, \
    variational_dropout, PrePostProcessing, EncoderLayerLocalAttention, EncoderLayerMemoryCompressed, DecoderLayerLocalAttention
from onmt.modules.BaseModel import NMTModel, Reconstructor, DecoderState
import onmt
from onmt.modules.WordDrop import embedded_dropout
from torch.utils.checkpoint import checkpoint
from torch.autograd import Variable
from onmt.utils import padToBlockSizeDimOne
import onmt.Constants



def custom_layer(module):
    def custom_forward(*args):
        return module(*args)
        #return output
    return custom_forward


class TransformerEncoderMemoryCompressed(nn.Module):
    """Encoder in 'Attention is all you need'

    Args:
        opt: list of options ( see train.py )        self.cuda = opt
        dicts : dictionary (for source language)

    """

    def __init__(self, opt, dicts, positional_encoder):

        super(TransformerEncoderMemoryCompressed, self).__init__()

        self.model_size = opt.model_size
        self.n_heads = opt.n_heads
        self.inner_size = opt.inner_size
        self.layers = opt.layers
        self.dropout = opt.dropout
        self.word_dropout = opt.word_dropout
        self.attn_dropout = opt.attn_dropout
        self.emb_dropout = opt.emb_dropout
        self.time = opt.time
        self.residual_dropout = opt.residual_dropout
        self.compression_factor = 2
        self.compression_function = 1
        self.block_size = opt.block_size
        self.cudaBool = (len(opt.gpus) >= 1)

        self.encoderLayers = opt.encoderLayers


        self.word_lut = nn.Embedding(dicts.size(),
                                     self.model_size,
                                     padding_idx=onmt.Constants.PAD)

        if opt.time == 'positional_encoding':
            self.time_transformer = positional_encoder
        elif opt.time == 'gru':
            self.time_transformer = nn.GRU(self.model_size, self.model_size, 1, batch_first=True)
        elif opt.time == 'lstm':
            self.time_transformer = nn.LSTM(self.model_size, self.model_size, 1, batch_first=True)

        self.preprocess_layer = PrePostProcessing(self.model_size, self.emb_dropout, sequence='d', static=False)

        self.postprocess_layer = PrePostProcessing(self.model_size, 0, sequence='n')

        self.positional_encoder = positional_encoder

        self.build_modules()

    def build_modules(self):

        self.layer_modules = nn.ModuleList([EncoderLayerLocalAttention(self.n_heads, self.model_size, self.dropout,
                                                                       self.inner_size, self.block_size, self.cudaBool,
                                                                       self.attn_dropout, self.residual_dropout) for _
                                            in
                                            range(self.encoderLayers)])

        # self.layer_modules = nn.ModuleList([EncoderLayerLocalAttention(self.n_heads, self.model_size,
        #                                                                self.dropout, self.inner_size,
        #                                                                self.block_size,
        #                                                                 self.attn_dropout, self.residual_dropout),
        #                                     EncoderLayerMemoryCompressed(self.n_heads, self.model_size,
        #                                                                  self.dropout, self.inner_size,
        #                                                                 self.attn_dropout, self.residual_dropout),
        #                                     EncoderLayerLocalAttention(self.n_heads, self.model_size, self.dropout,
        #                                                                self.inner_size, self.block_size,
        #                                                                self.attn_dropout, self.residual_dropout),
        #                                     EncoderLayerMemoryCompressed(self.n_heads, self.model_size, self.dropout,
        #                                                                  self.inner_size,
        #                                                                  self.attn_dropout, self.residual_dropout),
        #                                     EncoderLayerLocalAttention(self.n_heads, self.model_size, self.dropout,
        #                                                                self.inner_size, self.block_size,
        #                                                                self.attn_dropout, self.residual_dropout)
        #                                     ])

    def _slow_forward(self, *input, **kwargs):
        print('test slow forward')

    def forward(self, input, **kwargs):
        """
        Inputs Shapes:
            input: batch_sizUse half precision traininge x len_src (wanna tranpose)
            Input Dimension is (Batch_Size_Sentences x Batch_Size_words)

        Outputs Shapes:
            out: batch_size x len_src x d_model
            mask_src

        """

        if input.is_cuda:
            print('Input is cuda here. Bad')
        else:
            print('Input is not cuda here. Good')

        #D.S: Here padding to fit in blocks is made
        #input = padToBlockSizeDimOne(input, self.block_size, self.cudaBool)
        input = padToBlockSizeDimOne(input, self.block_size, False)


        """ Embedding: batch_size x len_src x d_model """
        emb = embedded_dropout(self.word_lut, input, dropout=self.word_dropout if self.training else 0)
        #D.S: emb dim (batch_size_sentence x batch_size_words x embedding_size)
        """ Scale the emb by sqrt(d_model) """

        if emb.is_cuda:
            print('emb is cuda here 1')

        emb = emb * math.sqrt(self.model_size)

        if emb.is_cuda:
            print('emb is cuda here 2')
        else:
            print('emb is not cuda here 2')

        """ Adding positional encoding """
        emb = self.time_transformer(emb)

        if emb.is_cuda:
            print('emb is cuda here 3')
        else:
            print('emb is not cuda here 3')

        emb = self.preprocess_layer(emb)

        if emb.is_cuda:
            print('emb is cuda here 4')
        else:
            print('emb is not cuda here 4')

        mask_src = input.eq(onmt.Constants.PAD).unsqueeze(1)  # batch_size x 1 x len_src for broadcasting

        # ~ pad_mask = input.ne(onmt.Constants.PAD)) # batch_size x len_src

        if emb.is_cuda:
            print('emb is cuda here 5')
        else:
            print('emb is not cuda here 5')


        context = emb.transpose(0, 1).contiguous()

        if context.is_cuda:
            print('Context is cuda here. Bad')

        # D.S: Context is splitted in seperate parts
        batch_sentences = context.shape[1]
        states_k = torch.zeros(
            (self.n_heads * batch_sentences, context.shape[0], (self.model_size // self.n_heads)))
        states_v = torch.zeros(
            (self.n_heads * batch_sentences, context.shape[0], (self.model_size // self.n_heads)))

        if onmt.Constants.cudaActivated:
            states_k = states_k.cuda()
            states_v = states_v.cuda()
            mask_src = mask_src.cuda()

        if onmt.Constants.cudaActivated and onmt.Constants.debug:
            print('Position 1')
            print('Real Memory allocated: ' + str(torch.cuda.memory_allocated()))

        first = True

        #original_batch_size = context.shape[0]
        splits = torch.split(context, self.block_size, dim=0)
        for step_num, split in enumerate(splits):

            if onmt.Constants.cudaActivated:
                #Check if split was on GPU before
                if split.is_cuda and first:
                    print('Split was already on GPU. Thats bad!!!')
                else:
                    first = False
                split = split.cuda()
                context = context.cuda()


            step_tensor = torch.tensor(step_num)
            for i, layer in enumerate(self.layer_modules):
                if onmt.Constants.cudaActivated and onmt.Constants.debug:
                    print('Position 5')
                    print('Real Memory allocated: ' + str(torch.cuda.memory_allocated()))

                if type(layer) is EncoderLayerLocalAttention:
                    #D.S: Handle Local Attention Layer
                    if len(self.layer_modules) - i <= onmt.Constants.checkpointing and self.training:
                        #def forward(self, query, key, value, mask, step_num, prev_k, prev_v, query_mask=None, value_mask=None):
                        #Context is used as v-Values, which are mulitplied with the attention (attn = q * k)
                        print('We are checkpointing for layer ' + str(i) + ' and split ' + str(step_num))
                        context, prev_k, prev_v = checkpoint(custom_layer(layer), context, split, mask_src,
                                                             step_tensor, states_k, states_v)
                        #Return values:
                        #Context contains
                        #prev_k & prev_v contains k anv v values from all previous splits, which are needed for the
                        # next split in where(...) operation, to concat the previous k & v values with the current one (From current split)
                    else:
                        context, prev_k, prev_v = layer(context, split, mask_src,
                                                        step_tensor, states_k, states_v)  # batch_size x len_src x d_model
                else:
                    raise NotImplementedError

                states_k = prev_k #Update states with k & v values from previous splits
                states_v = prev_v

                if onmt.Constants.cudaActivated and onmt.Constants.debug:
                    print('Position 6')
                    print('Real Memory allocated: ' + str(torch.cuda.memory_allocated()))

        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.
        context = self.postprocess_layer(context)

        return context, mask_src


class TransformerEncoder(nn.Module):
    """Encoder in 'Attention is all you need'
    
    Args:
        opt: list of options ( see train.py )
        dicts : dictionary (for source language)
        
    """
    
    def __init__(self, opt, dicts, positional_encoder):
    
        super(TransformerEncoder, self).__init__()
        
        self.model_size = opt.model_size
        self.n_heads = opt.n_heads
        self.inner_size = opt.inner_size
        self.layers = opt.layers
        self.dropout = opt.dropout
        self.word_dropout = opt.word_dropout
        self.attn_dropout = opt.attn_dropout
        self.emb_dropout = opt.emb_dropout
        self.time = opt.time
        self.residual_dropout = opt.residual_dropout
        
        self.word_lut = nn.Embedding(dicts.size(),
                                     self.model_size,
                                     padding_idx=onmt.Constants.PAD)
        
        if opt.time == 'positional_encoding':
            self.time_transformer = positional_encoder
        elif opt.time == 'gru':
            self.time_transformer = nn.GRU(self.model_size, self.model_size, 1, batch_first=True)
        elif opt.time == 'lstm':
            self.time_transformer = nn.LSTM(self.model_size, self.model_size, 1, batch_first=True)
        
        self.preprocess_layer = PrePostProcessing(self.model_size, self.emb_dropout, sequence='d', static=False)
        
        self.postprocess_layer = PrePostProcessing(self.model_size, 0, sequence='n')
        
        self.positional_encoder = positional_encoder
    
        self.build_modules()
        
    def build_modules(self):
        
        self.layer_modules = nn.ModuleList([EncoderLayer(self.n_heads, self.model_size, self.dropout, self.inner_size, self.attn_dropout, self.residual_dropout) for _ in range(self.layers)])

    def forward(self, input, **kwargs):
        """
        Inputs Shapes: 
            input: batch_size x len_src (wanna tranpose)
        
        Outputs Shapes:
            out: batch_size x len_src x d_model
            mask_src 
            
        """

        """ Embedding: batch_size x len_src x d_model """
        emb = embedded_dropout(self.word_lut, input, dropout=self.word_dropout if self.training else 0)
        
        """ Scale the emb by sqrt(d_model) """
        
        emb = emb * math.sqrt(self.model_size)
            
        """ Adding positional encoding """
        emb = self.time_transformer(emb)
        
        emb = self.preprocess_layer(emb)
        
        mask_src = input.eq(onmt.Constants.PAD).unsqueeze(1) # batch_size x 1 x len_src for broadcasting
        
        #~ pad_mask = input.ne(onmt.Constants.PAD)) # batch_size x len_src
        
        context = emb.transpose(0, 1).contiguous()
        
        for i, layer in enumerate(self.layer_modules):
            
            if len(self.layer_modules) - i <= onmt.Constants.checkpointing and self.training:        
                context = checkpoint(custom_layer(layer), context, mask_src)

            else:
                context = layer(context, mask_src)      # batch_size x len_src x d_model
            
        
        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.    
        context = self.postprocess_layer(context)
            
        
        return context, mask_src


class TransformerDecoderMemoryCompressed(nn.Module):
    """Encoder in 'Attention is all you need'

    Args:
        opt
        dicts


    """

    def __init__(self, opt, dicts, positional_encoder):

        super(TransformerDecoderMemoryCompressed, self).__init__()

        self.model_size = opt.model_size
        self.n_heads = opt.n_heads
        self.inner_size = opt.inner_size
        self.layers = opt.layers
        self.dropout = opt.dropout
        self.word_dropout = opt.word_dropout
        self.attn_dropout = opt.attn_dropout
        self.emb_dropout = opt.emb_dropout
        self.time = opt.time
        self.version = opt.version
        self.residual_dropout = opt.residual_dropout
        self.block_size = opt.block_size
        self.cuda = (len(opt.gpus) >= 1)


        if opt.time == 'positional_encoding':
            self.time_transformer = positional_encoder
        elif opt.time == 'gru':
            self.time_transformer = nn.GRU(self.model_size, self.model_size, 1, batch_first=True)
        elif opt.time == 'lstm':
            self.time_transformer = nn.LSTM(self.model_size, self.model_size, 1, batch_first=True)

        self.preprocess_layer = PrePostProcessing(self.model_size, self.emb_dropout, sequence='d', static=False)

        self.postprocess_layer = PrePostProcessing(self.model_size, 0, sequence='n')

        self.word_lut = nn.Embedding(dicts.size(),
                                     self.model_size,
                                     padding_idx=onmt.Constants.PAD)

        self.positional_encoder = positional_encoder

        len_max = self.positional_encoder.len_max
        mask = torch.ByteTensor(np.triu(np.ones((len_max, len_max)), k=1).astype('uint8'))
        self.register_buffer('mask', mask)

        self.build_modules()

    def build_modules(self):
        self.layer_modules = nn.ModuleList([DecoderLayerLocalAttention(self.n_heads, self.model_size, self.dropout,
                                                                       self.inner_size, self.block_size, self.cuda,
                                                                       self.attn_dropout, self.residual_dropout) for _ in
                                            range(self.layers)])

    def renew_buffer(self, new_len):

        self.positional_encoder.renew(new_len)

        if hasattr(self, 'mask'):
            del self.mask
        mask = torch.ByteTensor(np.triu(np.ones((new_len, new_len)), k=1).astype('uint8'))
        self.register_buffer('mask', mask)

    def mark_pretrained(self):

        self.pretrained_point = self.layers

    def add_layers(self, n_new_layer):

        #TODO: D.S: Check if this is still needed
        print('add_layers was executed. Change it to local attention layers')
        self.new_modules = list()
        self.layers += n_new_layer

        for i in range(n_new_layer):
            layer = EncoderLayer(self.n_heads, self.model_size, self.dropout, self.inner_size, self.attn_dropout)

            # the first layer will use the preprocessing which is the last postprocessing
            if i == 0:
                layer.preprocess_attn = self.postprocess_layer
                # replace the last postprocessing layer with a new one
                self.postprocess_layer = PrePostProcessing(d_model, 0, sequence='n')

            self.layer_modules.append(layer)

    def forward(self, input, context, src, **kwargs):
        """
        Inputs Shapes:
            input: (Variable) batch_size x len_tgt (wanna tranpose) (D.S: Batch_Size_Sentence x (Block_Size??)
            context: (Variable) batch_size x len_src x d_model
            mask_src (Tensor) batch_size x len_src
        Outputs Shapes:
            out: batch_size x len_tgt x d_model
            coverage: batch_size x len_tgt x len_src

        """


        #D.S: Here padding to fit in blocks is made
        if onmt.Constants.memoryCompressionActivated:
            input = padToBlockSizeDimOne(input, self.block_size, False)
            src = padToBlockSizeDimOne(src, self.block_size, False)
        else:
            input = padToBlockSizeDimOne(input, self.block_size, onmt.Constants.cudaActivated)
            src = padToBlockSizeDimOne(src, self.block_size, onmt.Constants.cudaActivated)

        """ Embedding: batch_size x len_tgt x d_model """
        emb = embedded_dropout(self.word_lut, input, dropout=self.word_dropout if self.training else 0)
        if self.time == 'positional_encoding':
            emb = emb * math.sqrt(self.model_size)
        """ Adding positional encoding """
        emb = self.time_transformer(emb)
        if isinstance(emb, tuple):
            emb = emb[0]
        emb = self.preprocess_layer(emb)

        mask_src = src.eq(onmt.Constants.PAD).unsqueeze(1)
        #mask_src = torch.split(src, self.block_size, dim=-1)[0].eq(onmt.Constants.PAD).unsqueeze(1)

        pad_mask_src = src.data.ne(onmt.Constants.PAD)

        if onmt.Constants.memoryCompressionActivated:
            self.mask = self.mask.cpu()

        #TODO: D.S: Remove this afterwards
        if self.mask.is_cuda:
            print('self.mask is cuda')

        # TODO: D.S: Remove this afterwards
        if input.is_cuda:
            print('input is cuda')

        len_tgt = input.size(1)
        mask_tgt = input.data.eq(onmt.Constants.PAD).unsqueeze(1) + self.mask[:len_tgt, :len_tgt]
        mask_tgt = torch.gt(mask_tgt, 0)

        output = emb.transpose(0, 1).contiguous()
        # D.S: Context is splitted in seperate parts
        batch_sentences = output.shape[1]

        states_k = torch.zeros((self.n_heads * batch_sentences, context.shape[0], (self.model_size // self.n_heads)))
        states_v = torch.zeros((self.n_heads * batch_sentences, context.shape[0], (self.model_size // self.n_heads)))

        if  onmt.Constants.cudaActivated:
            states_k = states_k.cuda()
            states_v = states_v.cuda()
            if onmt.Constants.memoryCompressionActivated:
                mask_tgt = mask_tgt.cuda()
                mask_src = mask_src.cuda()

        #TODO: D.S. Is the split on output correct? Should here the split on context/input be done?
        splits = torch.split(context, self.block_size, dim=0)
        for step_num, split in enumerate(splits):
            step_tensor = torch.tensor(step_num)
            for i, layer in enumerate(self.layer_modules):

                if type(layer) is DecoderLayerLocalAttention:
                    # D.S: Handle Local Attention Layer
                    if len(self.layer_modules) - i <= onmt.Constants.checkpointing and self.training:
                        # def forward(self, query, key, value, mask, step_num, prev_k, prev_v, query_mask=None, value_mask=None):
                        #output, prev_k, prev_v = checkpoint(custom_layer(layer), output,
                        #                                     split, mask_tgt, mask_src, step_tensor, states_k, states_v)
                        output, prev_k, prev_v = checkpoint(custom_layer(layer), output,
                                                            split, mask_tgt, mask_src, step_tensor, states_k, states_v)
                    else:
                        output, prev_k, prev_v = layer(output, split, mask_tgt, mask_src, step_tensor,
                                                        states_k, states_v)  # batch_size x len_src x d_model
                else:
                    raise NotImplementedError

                #Write states from last split to input of next split
                states_k = prev_k
                states_v = prev_v

        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.
        output = self.postprocess_layer(output)

        return output, None

    def step(self, input, decoder_state):
        """
        Inputs Shapes:
            input: (Variable) batch_size x len_tgt (wanna tranpose)
            context: (Tensor) len_src x batch_size * beam_size x d_model
            mask_src (Tensor) batch_size x len_src
            buffer (List of tensors) List of batch_size * len_tgt-1 * d_model for self-attention recomputing
        Outputs Shapes:
            out: batch_size x len_tgt x d_model
            coverage: batch_size x len_tgt x len_src

        """
        context = decoder_state.context
        buffers = decoder_state.attention_buffers
        mask_src = decoder_state.src_mask

        if decoder_state.concat_input_seq == True:
            if decoder_state.input_seq is None:
                decoder_state.input_seq = input
            else:
                # concatenate the last input to the previous input sequence
                decoder_state.input_seq = torch.cat([decoder_state.input_seq, input], 0)
            input = decoder_state.input_seq.transpose(0, 1)
            src = decoder_state.src.transpose(0, 1)

        input_ = input[:, -1].unsqueeze(1)
        # ~ print(input.size())
        # ~ print(mask_src.size())
        # ~ print(context.size())

        output_buffer = list()

        batch_size = input_.size(0)

        """ Embedding: batch_size x 1 x d_model """
        emb = self.word_lut(input_)

        emb = emb * math.sqrt(self.model_size)
        """ Adding positional encoding """
        emb = self.time_transformer(emb, t=input.size(1))

        if isinstance(emb, tuple):
            emb = emb[0]
        # emb should be batch_size x 1 x dim

        # Preprocess layer: adding dropout
        emb = self.preprocess_layer(emb)

        emb = emb.transpose(0, 1)

        # batch_size x 1 x len_src
        if mask_src is None:
            mask_src = src.eq(onmt.Constants.PAD).unsqueeze(1)

        len_tgt = input.size(1)
        mask_tgt = input.data.eq(onmt.Constants.PAD).unsqueeze(1) + self.mask[:len_tgt, :len_tgt]
        mask_tgt = torch.gt(mask_tgt, 0)
        mask_tgt = mask_tgt[:, -1, :].unsqueeze(1)

        output = emb.contiguous()

        # FOR DEBUGGING
        # ~ decoder_state._debug_attention_buffer(0)

        for i, layer in enumerate(self.layer_modules):
            buffer = buffers[i] if i in buffers else None
            assert (output.size(0) == 1)
            output, coverage, buffer = layer.step(output, context, mask_tgt, mask_src,
                                                  buffer=buffer)  # batch_size x len_src x d_model

            decoder_state._update_attention_buffer(buffer, i)

        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.
        output = self.postprocess_layer(output)

        return output, coverage


class TransformerDecoder(nn.Module):
    """Encoder in 'Attention is all you need'
    
    Args:
        opt
        dicts 
        
        
    """
    
    def __init__(self, opt, dicts, positional_encoder):
    
        super(TransformerDecoder, self).__init__()
        
        self.model_size = opt.model_size
        self.n_heads = opt.n_heads
        self.inner_size = opt.inner_size
        self.layers = opt.layers
        self.dropout = opt.dropout
        self.word_dropout = opt.word_dropout 
        self.attn_dropout = opt.attn_dropout
        self.emb_dropout = opt.emb_dropout
        self.time = opt.time
        self.version = opt.version
        self.residual_dropout = opt.residual_dropout
        
        if opt.time == 'positional_encoding':
            self.time_transformer = positional_encoder
        elif opt.time == 'gru':
            self.time_transformer = nn.GRU(self.model_size, self.model_size, 1, batch_first=True)
        elif opt.time == 'lstm':
            self.time_transformer = nn.LSTM(self.model_size, self.model_size, 1, batch_first=True)
        
        self.preprocess_layer = PrePostProcessing(self.model_size, self.emb_dropout, sequence='d', static=False)
        
        self.postprocess_layer = PrePostProcessing(self.model_size, 0, sequence='n')
        
        self.word_lut = nn.Embedding(dicts.size(),
                                     self.model_size,
                                     padding_idx=onmt.Constants.PAD)
        
        self.positional_encoder = positional_encoder
        
        
        len_max = self.positional_encoder.len_max
        mask = torch.ByteTensor(np.triu(np.ones((len_max,len_max)), k=1).astype('uint8'))
        self.register_buffer('mask', mask)
        
        self.build_modules()
        
    def build_modules(self):
        self.layer_modules = nn.ModuleList([DecoderLayer(self.n_heads, self.model_size, self.dropout, self.inner_size, self.attn_dropout, self.residual_dropout) for _ in range(self.layers)])
    
    def renew_buffer(self, new_len):
        
        self.positional_encoder.renew(new_len)

        if hasattr(self, 'mask'):
            del self.mask
        mask = torch.ByteTensor(np.triu(np.ones((new_len,new_len)), k=1).astype('uint8'))
        self.register_buffer('mask', mask)
    
    def mark_pretrained(self):
        
        self.pretrained_point = self.layers
        
    
    def add_layers(self, n_new_layer):
        
        self.new_modules = list()
        self.layers += n_new_layer
        
        for i in range(n_new_layer):
            layer = EncoderLayer(self.n_heads, self.model_size, self.dropout, self.inner_size, self.attn_dropout) 
            
            # the first layer will use the preprocessing which is the last postprocessing
            if i == 0:
                layer.preprocess_attn = self.postprocess_layer
                # replace the last postprocessing layer with a new one
                self.postprocess_layer = PrePostProcessing(d_model, 0, sequence='n')
            
            self.layer_modules.append(layer)
        
    def forward(self, input, context, src, **kwargs):
        """
        Inputs Shapes: 
            input: (Variable) batch_size x len_tgt (wanna tranpose)
            context: (Variable) batch_size x len_src x d_model
            mask_src (Tensor) batch_size x len_src
        Outputs Shapes:
            out: batch_size x len_tgt x d_model
            coverage: batch_size x len_tgt x len_src
            
        """
        
        """ Embedding: batch_size x len_tgt x d_model """
        emb = embedded_dropout(self.word_lut, input, dropout=self.word_dropout if self.training else 0)
        if self.time == 'positional_encoding':
            emb = emb * math.sqrt(self.model_size)
        """ Adding positional encoding """
        emb = self.time_transformer(emb)
        if isinstance(emb, tuple):
            emb = emb[0]
        emb = self.preprocess_layer(emb)
        
        mask_src = src.eq(onmt.Constants.PAD).unsqueeze(1)
        
        pad_mask_src = src.data.ne(onmt.Constants.PAD)
        
        len_tgt = input.size(1)
        mask_tgt = input.data.eq(onmt.Constants.PAD).unsqueeze(1) + self.mask[:len_tgt, :len_tgt]
        mask_tgt = torch.gt(mask_tgt, 0)
        
        output = emb.transpose(0, 1).contiguous()

        for i, layer in enumerate(self.layer_modules):
            
            if len(self.layer_modules) - i <= onmt.Constants.checkpointing and self.training:           
                
                output, coverage = checkpoint(custom_layer(layer), output, context, mask_tgt, mask_src) 
                                                                              # batch_size x len_src x d_model
                
            else:
                output, coverage = layer(output, context, mask_tgt, mask_src) # batch_size x len_src x d_model
            
            
        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.    
        output = self.postprocess_layer(output)
            
        
        return output, None
    

    def step(self, input, decoder_state):
        """
        Inputs Shapes: 
            input: (Variable) batch_size x len_tgt (wanna tranpose)
            context: (Tensor) len_src x batch_size * beam_size x d_model
            mask_src (Tensor) batch_size x len_src
            buffer (List of tensors) List of batch_size * len_tgt-1 * d_model for self-attention recomputing
        Outputs Shapes:
            out: batch_size x len_tgt x d_model
            coverage: batch_size x len_tgt x len_src
            
        """
        context = decoder_state.context
        buffers = decoder_state.attention_buffers
        mask_src = decoder_state.src_mask
        
        if decoder_state.concat_input_seq == True:
            if decoder_state.input_seq is None:
                decoder_state.input_seq = input
            else:
                # concatenate the last input to the previous input sequence
                decoder_state.input_seq = torch.cat([decoder_state.input_seq, input], 0)
            input = decoder_state.input_seq.transpose(0, 1)
            src = decoder_state.src.transpose(0, 1)
        
        input_ = input[:,-1].unsqueeze(1)
        # ~ print(input.size())
        # ~ print(mask_src.size())
        # ~ print(context.size())
        
        
        output_buffer = list()
            
        batch_size = input_.size(0)
        
        """ Embedding: batch_size x 1 x d_model """
        emb = self.word_lut(input_)
       
        
        emb = emb * math.sqrt(self.model_size)
        """ Adding positional encoding """
        emb = self.time_transformer(emb, t=input.size(1))
            
        if isinstance(emb, tuple):
            emb = emb[0]
        # emb should be batch_size x 1 x dim
            
        # Preprocess layer: adding dropout
        emb = self.preprocess_layer(emb)
        
        emb = emb.transpose(0, 1)
        
        # batch_size x 1 x len_src
        if mask_src is None:
            mask_src = src.eq(onmt.Constants.PAD).unsqueeze(1)
        
        len_tgt = input.size(1)
        mask_tgt = input.data.eq(onmt.Constants.PAD).unsqueeze(1) + self.mask[:len_tgt, :len_tgt]
        mask_tgt = torch.gt(mask_tgt, 0)
        mask_tgt = mask_tgt[:, -1, :].unsqueeze(1)
                
        output = emb.contiguous()
        
        # FOR DEBUGGING
        # ~ decoder_state._debug_attention_buffer(0)
    
        for i, layer in enumerate(self.layer_modules):
            
            buffer = buffers[i] if i in buffers else None
            assert(output.size(0) == 1)
            output, coverage, buffer = layer.step(output, context, mask_tgt, mask_src, buffer=buffer) # batch_size x len_src x d_model
            
            decoder_state._update_attention_buffer(buffer, i)

        
        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.    
        output = self.postprocess_layer(output)

        return output, coverage


class TransformerMemoryCompressed(NMTModel):
    """Main model in 'Attention is all you need' """

    def __init__(self, encoder, decoder, generator, cuda):
        super(TransformerMemoryCompressed, self).__init__(encoder, decoder, generator)
        self.cudaBool = cuda


    def forward(self, batch, grow=False):
        """
        Inputs Shapes:
            src: len_src x batch_size
            tgt: len_tgt x batch_size

        Outputs Shapes:
            out:      batch_size*len_tgt x model_size


        """
        src = batch.get('source')
        tgt = batch.get('target_input')

        src = src.transpose(0, 1)  # transpose to have batch first
        tgt = tgt.transpose(0, 1)

        if self.cudaBool and onmt.Constants.debug:
            print('Position 2')
            print('Max Memory allocated (Before encoder forward 1): ' + str(torch.cuda.max_memory_allocated()))
            print('Real Memory allocated (Before encoder forward 1): ' + str(torch.cuda.memory_allocated()))

        context, src_mask = self.encoder(src, grow=grow)

        if self.cudaBool and onmt.Constants.debug:
            print('Position 3')
            print('Max Memory allocated (After encoder forward): ' + str(torch.cuda.max_memory_allocated()))
            print('Real Memory allocated (After encoder forward): ' + str(torch.cuda.memory_allocated()))

        output, coverage = self.decoder(tgt, context, src, grow=grow)

        if self.cudaBool and onmt.Constants.debug:
            print('Position 4')
            print('Max Memory allocated : ' + str(torch.cuda.max_memory_allocated()))
            print('Real Memory allocated : ' + str(torch.cuda.memory_allocated()))

        output_dict = dict()
        output_dict['hiddens'] = output
        output_dict['coverage'] = coverage
        return output_dict

    def create_decoder_state(self, src, context, mask_src, beamSize=1, type='old'):

        from onmt.modules.ParallelTransformer.Models import ParallelTransformerEncoder, ParallelTransformerDecoder
        from onmt.modules.StochasticTransformer.Models import StochasticTransformerEncoder, StochasticTransformerDecoder
        from onmt.modules.UniversalTransformer.Models import UniversalTransformerDecoder

        if isinstance(self.decoder, TransformerDecoder) or isinstance(self.decoder,
                                                                      StochasticTransformerDecoder) or isinstance(
                self.decoder, UniversalTransformerDecoder):
            decoder_state = TransformerDecodingState(src, context, mask_src, beamSize=beamSize, type=type)
        elif isinstance(self.decoder, ParallelTransformerDecoder):
            from onmt.modules.ParallelTransformer.Models import ParallelTransformerDecodingState
            decoder_state = ParallelTransformerDecodingState(src, context, mask_src, beamSize=beamSize)
        return decoder_state

        
class Transformer(NMTModel):
    """Main model in 'Attention is all you need' """
    
        
    def forward(self, batch, grow=False):
        """
        Inputs Shapes: 
            src: len_src x batch_size
            tgt: len_tgt x batch_size
        
        Outputs Shapes:
            out:      batch_size*len_tgt x model_size
            
            
        """
        src = batch.get('source')
        tgt = batch.get('target_input')
        
        src = src.transpose(0, 1) # transpose to have batch first
        tgt = tgt.transpose(0, 1)
        
        context, src_mask = self.encoder(src, grow=grow)
        
        output, coverage = self.decoder(tgt, context, src, grow=grow)
            
        output_dict = dict()
        output_dict['hiddens'] = output
        output_dict['coverage'] = coverage
        return output_dict
        
    def create_decoder_state(self, src, context, mask_src, beamSize=1, type='old'):
        
        from onmt.modules.ParallelTransformer.Models import ParallelTransformerEncoder, ParallelTransformerDecoder
        from onmt.modules.StochasticTransformer.Models import StochasticTransformerEncoder, StochasticTransformerDecoder
        from onmt.modules.UniversalTransformer.Models import UniversalTransformerDecoder
        
        if isinstance(self.decoder, TransformerDecoder) or isinstance(self.decoder, StochasticTransformerDecoder) or isinstance(self.decoder, UniversalTransformerDecoder) :
            decoder_state = TransformerDecodingState(src, context, mask_src, beamSize=beamSize, type=type)
        elif isinstance(self.decoder, ParallelTransformerDecoder):
            from onmt.modules.ParallelTransformer.Models import ParallelTransformerDecodingState
            decoder_state = ParallelTransformerDecodingState(src, context, mask_src, beamSize=beamSize)
        return decoder_state


class TransformerDecodingState(DecoderState):
    
    def __init__(self, src, context, src_mask, beamSize=1, type='old'):
        
        
        self.beam_size = beamSize
        
        self.input_seq = None
        self.attention_buffers = dict()
        
        if type == 'old':
            self.src = src.repeat(1, beamSize)
            self.context = context.repeat(1, beamSize, 1)
            self.beamSize = beamSize
            self.src_mask = None
            self.concat_input_seq = True
        elif type == 'new':
            bsz = context.size(1)
            new_order = torch.arange(bsz).view(-1, 1).repeat(1, self.beam_size).view(-1)
            new_order = new_order.to(context.device)
            self.context = context.index_select(1, new_order)
            self.src_mask = src_mask.index_select(0, new_order)
            self.concat_input_seq = False
        
    def _update_attention_buffer(self, buffer, layer):
        
        self.attention_buffers[layer] = buffer # dict of 2 keys (k, v) : T x B x H
        
    def _debug_attention_buffer(self, layer):
        
        if layer not in self.attention_buffers:
            return
        buffer = self.attention_buffers[layer]
        
        for k in buffer.keys():
            print(k, buffer[k].size())
        
    def _update_beam(self, beam, b, remainingSents, idx):
        # here we have to reorder the beam data 
        # 
        for tensor in [self.src, self.input_seq]  :
                    
            t_, br = tensor.size()
            sent_states = tensor.view(t_, self.beamSize, remainingSents)[:, :, idx]
            
            if isinstance(tensor, Variable):
                sent_states.data.copy_(sent_states.data.index_select(
                            1, beam[b].getCurrentOrigin()))
            else:
                sent_states.copy_(sent_states.index_select(
                            1, beam[b].getCurrentOrigin()))
                            
        for l in self.attention_buffers:
            buffer_ = self.attention_buffers[l]
            if buffer_ is not None:
                for k in buffer_:
                    t_, br_, d_ = buffer_[k].size()
                    sent_states = buffer_[k].view(t_, self.beamSize, remainingSents, d_)[:, :, idx, :]
                    
                    sent_states.data.copy_(sent_states.data.index_select(
                                1, beam[b].getCurrentOrigin()))
    
    
    # in this section, the sentences that are still active are
    # compacted so that the decoder is not run on completed sentences
    def _prune_complete_beam(self, activeIdx, remainingSents):
        
        model_size = self.context.size(-1)
        
        def updateActive(t):
            # select only the remaining active sentences
            view = t.data.view(-1, remainingSents, model_size)
            newSize = list(t.size())
            newSize[-2] = newSize[-2] * len(activeIdx) // remainingSents
            return Variable(view.index_select(1, activeIdx)
                            .view(*newSize))
        
        def updateActive2D(t):
            if isinstance(t, Variable):
                # select only the remaining active sentences
                view = t.data.view(-1, remainingSents)
                newSize = list(t.size())
                newSize[-1] = newSize[-1] * len(activeIdx) // remainingSents
                return Variable(view.index_select(1, activeIdx)
                                .view(*newSize))
            else:
                view = t.view(-1, remainingSents)
                newSize = list(t.size())
                newSize[-1] = newSize[-1] * len(activeIdx) // remainingSents
                new_t = view.index_select(1, activeIdx).view(*newSize)
                                
                return new_t
        
        def updateActive4D(t):
            # select only the remaining active sentences
            nl, t_, br_, d_ = t.size()
            view = t.data.view(nl, -1, remainingSents, model_size)
            newSize = list(t.size())
            newSize[-2] = newSize[-2] * len(activeIdx) // remainingSents
            return Variable(view.index_select(2, activeIdx)
                            .view(*newSize)) 
        
        self.context = updateActive(self.context)
        
        self.input_seq = updateActive2D(self.input_seq)
        
        self.src = updateActive2D(self.src)
        
        for l in self.attention_buffers:
            buffer_ = self.attention_buffers[l]
            if buffer_ is not None:
                for k in buffer_:
                    buffer_[k] = updateActive(buffer_[k])

    # For the new decoder version only
    def _reorder_incremental_state(self, reorder_state):
        self.context = self.context.index_select(1, reorder_state)
        self.src_mask = self.src_mask.index_select(0, reorder_state)
                            
        for l in self.attention_buffers:
            buffer_ = self.attention_buffers[l]
            if buffer_ is not None:
                for k in buffer_.keys():
                    t_, br_, d_ = buffer_[k].size()
                    buffer_[k] = buffer_[k].index_select(1, reorder_state) # 1 for time first
        



