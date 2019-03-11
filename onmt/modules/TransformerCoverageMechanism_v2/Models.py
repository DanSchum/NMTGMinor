import numpy as np
import torch, math
import torch.nn as nn
from onmt.modules.Transformer.Layers import EncoderLayer, DecoderLayer, PositionalEncoding, variational_dropout, PrePostProcessing
from onmt.modules.BaseModel import NMTModel, Reconstructor, DecoderState
import onmt
from onmt.modules.WordDrop import embedded_dropout
#~ from onmt.modules.Checkpoint import checkpoint
from torch.utils.checkpoint import checkpoint
from torch.autograd import Variable
import torch.nn.functional as F



def custom_layer(module):
    def custom_forward(*args):
        output = module(*args)
        return output
    return custom_forward

        

class TransformerEncoder(nn.Module):
    """Encoder in 'Attention is all you need'
    
    Args:
        opt: list of options ( see train.py )
        dicts : dictionary (for source language)
        
    """
    
    def __init__(self, opt, dicts, positional_encoder):
    
        super(TransformerEncoder, self).__init__()
        
        self.model_size = opt.model_size #dmodel which is the dimension between sublayers
        self.n_heads = opt.n_heads #heads in multihead attention
        self.inner_size = opt.inner_size #Size of feed forward network in sublayer
        self.layers = opt.layers #Amount of stacked encoder/decoder layers in the model
        self.dropout = opt.dropout
        self.word_dropout = opt.word_dropout #D.S: Dropout which is applied by converting input to embedding
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

        #Performs Preprocessing (here its dropout)
        self.preprocess_layer = PrePostProcessing(self.model_size, self.emb_dropout, sequence='d', static=False)

        #Performs Postprocessing (here its layerNorm)
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
        #D.S: self.training is always 0
        #D.S: word_lut is look up table which contains embedding for each
        emb = embedded_dropout(self.word_lut, input, dropout=self.word_dropout if self.training else 0)
        
        """ Scale the emb by sqrt(d_model) """
        
        emb = emb * math.sqrt(self.model_size)
            
        """ Adding positional encoding """
        emb = self.time_transformer(emb)
        
        emb = self.preprocess_layer(emb)


        #D.S. tensor.eq computes elementwise equality (Compares each element. If elements are the same then return tensor has 1 at this element position, 0 otherwise.
        #D.S: Input tensor have to be the same dimensions
        #D.S: mask_src is 1 where input is 0. Mask is set size one in dimension 1
        #D.S: TODO: mask_src: Not sure how this is working??
        mask_src = input.eq(onmt.Constants.PAD).unsqueeze(1) # batch_size x len_src x 1 for broadcasting
        
        #~ pad_mask = input.ne(onmt.Constants.PAD)) # batch_size x len_src
        
        context = emb.transpose(0, 1).contiguous()
        
        for i, layer in enumerate(self.layer_modules):

            #D.S: TODO: self.training is never set, so if always fails
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

        #D.S: Iterate through all layers in Transformer Decoder
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
        #~ buffer = decoder_state.buffer
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
                
        return output
        
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
        
class GeneratorCoverageMechanism(nn.Module):

    def __init__(self, hidden_size, output_size):
        super(GeneratorCoverageMechanism, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        # ~ self.linear = onmt.modules.Transformer.Layers.XavierLinear(hidden_size, output_size)
        self.linear = nn.Linear(hidden_size, output_size)

        stdv = 1. / math.sqrt(self.linear.weight.size(1))

        torch.nn.init.uniform_(self.linear.weight, -stdv, stdv)

        self.linear.bias.data.zero_()
        intermediateSize = 100

        self.linearAvgProbInput = nn.Linear(output_size, intermediateSize)
        self.linearAvgProbOutput = nn.Linear(intermediateSize, output_size)
        #self.linearAcutalProb = nn.Linear(hidden_size, output_size)
        self.linearWordFrequencyModelInput = nn.Linear(output_size, intermediateSize)
        self.linearWordFrequencyModelOutput = nn.Linear(intermediateSize, output_size)

        if onmt.Constants.cudaActivated:
            print('Generator Linears are cuda')
            self.linearAvgProbInput = self.linearAvgProbInput.cuda()
            self.linearAvgProbOutput = self.linearAvgProbOutput.cuda()
            self.linearWordFrequencyModelInput = self.linearWordFrequencyModelInput.cuda()
            self.linearWordFrequencyModelOutput = self.linearWordFrequencyModelOutput.cuda()

        torch.nn.init.uniform_(self.linearAvgProbInput.weight, -stdv, stdv)
        torch.nn.init.uniform_(self.linearAvgProbOutput.weight, -stdv, stdv)
        torch.nn.init.uniform_(self.linearWordFrequencyModelInput.weight, -stdv, stdv)
        torch.nn.init.uniform_(self.linearWordFrequencyModelOutput.weight, -stdv, stdv)

        self.linearAvgProbInput.bias.data.zero_()
        self.linearAvgProbOutput.bias.data.zero_()
        self.linearWordFrequencyModelInput.bias.data.zero_()
        self.linearWordFrequencyModelOutput.bias.data.zero_()


        #D.S: New Tensor keeping the average probability of all previous words in this example
        self.avgProb = torch.zeros(output_size, dtype=torch.float) #D.S: Dimension (target_vocabulary)
        if onmt.Constants.cudaActivated:
            print('Avg model is cuda')
            #self.avgProb = self.avgProb.cuda()
        #Avg Word Probability containing the previous words, is used to reduce probability of future words, if they already used in output


    def forward(self, input, wordFrequencyModel, log_softmax=True):
        '''
        :param input:
        :param wordFrequencyModel:
        :param log_softmax:
        :return:
        '''
        # D.S: Input has dim: (batch_size_sentences x embedding_size)
        # added float to the end
        # print(input.size())


        logits = self.linear(input).float()



        # D.S: output has dim: (batch_size_sentences x Target_vocab)
        #For beam search the k (std 4) largest values by torch.topk are taken as the words with highest scores.

        # sumLogits = abs(torch.sum(logits))
        # maxLogits = torch.max(logits)
        # maxIndex = torch.argmax(logits)
        # minLogits = torch.min(logits)
        # minIndex = torch.argmin(logits)
        # meanLogits = torch.mean(logits)
        # testValue = abs(maxLogits/meanLogits)/meanLogits
        # #self.avgProb = (self.avgProb + logits) / sumLogits
        #

        if logits.is_cuda:
            logits = logits.cpu()
        meanLogits = torch.mean(logits)
        topScores = torch.topk(logits, 4, dim=-1)
        topScoresTensor = topScores[0]
        topScoresTensor = torch.abs(topScoresTensor / meanLogits)
        self.avgProb[topScores[1]] = torch.sigmoid(self.avgProb[topScores[1]] + topScoresTensor)

        # sumavgprob = abs(torch.sum(self.avgProb ))
        # maxavgprob= torch.max(self.avgProb )
        # maxIndexavgprob = torch.argmax(self.avgProb )
        # minavgprob = torch.min(self.avgProb )
        # minIndexavgprob = torch.argmin(self.avgProb )
        # meanavgprob = torch.mean(self.avgProb )
        # topScoresAvg = torch.topk(self.avgProb, 100, dim=0)


        #logits and output contains negative values. In beam search the maximum values are taken as top scores (means the smallest negative values)
        #To reduce the probability of token, reduce value, to increase probability, increase the value

        #if logits.is_cuda:
        #    wordFrequencyModel = wordFrequencyModel.cuda()
        #    self.avgProb = self.avgProb.cuda()

        if onmt.Constants.cudaActivated:
            self.avgProb = self.avgProb.cuda()
            wordFrequencyModel = wordFrequencyModel.cuda()
        weightedAvgProb = self.linearAvgProbInput(self.avgProb).float()
        weightedAvgProb = self.linearAvgProbOutput(weightedAvgProb).float()
        weightedWordFrequencyModel = self.linearWordFrequencyModelInput(wordFrequencyModel).float()
        weightedWordFrequencyModel = self.linearWordFrequencyModelOutput(weightedWordFrequencyModel).float()

        if not logits.is_cuda and onmt.Constants.cudaActivated:
            logits = logits.cuda()

        logitsMixed = (logits + weightedWordFrequencyModel - weightedAvgProb)

        if onmt.Constants.cudaActivated:
            logitsMixed = logitsMixed.cuda()
        #if logits.is_cuda:
            #self.avgProb = self.avgProb.cpu()

        # sumLogits = abs(torch.sum(logitsMixed))
        # maxLogits = torch.max(logitsMixed)
        # maxIndex = torch.argmax(logitsMixed)
        # minLogits = torch.min(logitsMixed)
        # minIndex = torch.argmin(logitsMixed)
        # meanLogits = torch.mean(logitsMixed)

        if log_softmax:
            output = F.log_softmax(logitsMixed, dim=-1)
        else:
            output = logits

        # maxLogitsAfterSoftmax = torch.max(output)
        # maxIndexAfterSoftmax = torch.argmax(output)
        # minLogitsAfterSoftmax = torch.min(output)
        # minIndexAfterSoftmax = torch.argmin(output)
        # meanAfterSoftmax = torch.mean(output)

        return output

    def resetAfterExample(self):
        #topScoresAvg = torch.topk(self.avgProb, 100, dim=0)
        self.avgProb = torch.zeros(self.output_size, dtype=torch.float)
        if onmt.Constants.cudaActivated:
            print('Reset of avg model for new sequence is done')
            print('Avg model is cuda')
            #self.avgProb = self.avgProb.cuda()

