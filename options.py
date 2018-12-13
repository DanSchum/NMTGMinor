import argparse

def make_parser(parser):
    
    
    # Data options
    parser.add_argument('-data', required=True,
                        help='Path to the *-train.pt file from preprocess.py')
    parser.add_argument('-data_format', required=True, default='bin',
                        help='Default data format: raw')
    parser.add_argument('-sort_by_target', action='store_true',
                        help='Training data sorted by target')                    
    parser.add_argument('-pad_count', action='store_true',
                        help='Training data sorted by target')                    
    parser.add_argument('-save_model', default='model',
                        help="""Model filename (the model will be saved as
                        <save_model>_epochN_PPL.pt where PPL is the
                        validation perplexity""")
    parser.add_argument('-load_from', default='', type=str,
                        help="""If training from a checkpoint then this is the
                        path to the pretrained model.""")
    parser.add_argument('-model', default='recurrent',
                        help="Optimization method. [recurrent|transformer|stochastic_transformer]")
    # TODO: D.S: Figure this out. What is meant by layers here: In paper there are 6 layers stacked per encoder/decoder.
    #Answer 07.11.18: Yes, this should be set to 6 in respect to original paper
    parser.add_argument('-layers', type=int, default=2,
                        help='Number of layers in the LSTM encoder/decoder')                   


    # Recurrent Model options
    parser.add_argument('-rnn_size', type=int, default=512,
                        help='Size of LSTM hidden states')
    parser.add_argument('-word_vec_size', type=int, default=512,
                        help='Word embedding sizes')
    #D.S: Attention Settings?
    parser.add_argument('-input_feed', type=int, default=1, #D.S: Default is yes. Its standard implementation of transformer
                        help="""Feed the context vector at each time step as
                        additional input (via concatenation with the word
                        embeddings) to the decoder.""")
    #D.S: Attention Settings?
    parser.add_argument('-brnn_merge', default='concat', #D.S: Standard way is here to contact them to a matrix, where attention vector is used to select the important source parts
                        help="""Merge action for the bidirectional hidden states:
                        [concat|sum]""")


    # Transforer Model
    parser.add_argument('-model_size', type=int, default=512,
        help='Size of embedding / transformer hidden')  #D.S: dmodel which controls the output/input size of all transformer sublayers
    parser.add_argument('-inner_size', type=int, default=2048,
        help='Size of inner feed forward layer')  #D.S: Size of FF layer in each transformer sublayer
    parser.add_argument('-n_heads', type=int, default=8,
        help='Number of heads for multi-head attention') #D.S: Multihead Attention (h=8), which is default from original paper
    parser.add_argument('-checkpointing', type=int, default=0,
        help='Number of checkpointed layers in the Transformer')  #D.S: TODO: ??
    parser.add_argument('-attn_dropout', type=float, default=0.1,
                        help='Dropout probability; applied on multi-head attention.') #D.S: Std from t2t is 0.0 (v1) and 0.1 (v2)
    parser.add_argument('-emb_dropout', type=float, default=0.1,
                        help='Dropout probability; applied on top of embedding.')    
    parser.add_argument('-residual_dropout', type=float, default=0.2,
                        help='Dropout probability; applied on residual connection.')    
    parser.add_argument('-weight_norm', action='store_true',
                      help='Apply weight normalization on linear modules') #D.S: Why not?
    parser.add_argument('-layer_norm', default='fast',
                      help='Layer normalization type')
    parser.add_argument('-death_rate', type=float, default=0.5,
                        help='Stochastic layer death rate')  
    parser.add_argument('-death_type', type=str, default='linear_decay',
                        help='Stochastic layer death type: linear decay or uniform')  
    parser.add_argument('-activation_layer', default='linear_relu_linear', type=str,
                        help='The activation layer in each transformer block') #D.S: std: dense_relu_dense
    parser.add_argument('-time', default='positional_encoding', type=str,
                        help='Type of time representation positional_encoding|gru|lstm') #D.S: Transformer default is positional encoding via trigonometic functions
    parser.add_argument('-version', type=float, default=1.0,
                        help='Transformer version. 1.0 = Google type | 2.0 is different')    #D.S: TODO: ??
    parser.add_argument('-attention_out', default='default',
                      help='Type of attention out. default|combine') #D.S: TODO: What is default (concat??)
    parser.add_argument('-residual_type', default='regular',
                      help='Type of residual type. regular|gated') #D.S: https://arxiv.org/pdf/1611.01260.pdf
    #Gates residual connection do apply additional control how much of the input x is directly forwarded to output of the layer which is using the residual connection
    # Optimization options
    parser.add_argument('-encoder_type', default='text',
                        help="Type of encoder to use. Options are [text|img].") #D.S: Default
    parser.add_argument('-init_embedding', default='normal',
                        help="How to init the embedding matrices. Xavier or Normal.") #D.S: http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization
    parser.add_argument('-batch_size_words', type=int, default=2048, #This is the maximum number of words which can be processed within one batch
                        help='Maximum batch size in word dimension') #D.S: TODO: Is this the maxiumum of words that can be processed in one cycle?
    parser.add_argument('-batch_size_sents', type=int, default=128,
                        help='Maximum number of sentences in a batch') #D.S: TODO: ??
    parser.add_argument('-max_generator_batches', type=int, default=32,
                        help="""Maximum batches of words in a sequence to run
                        the generator on in parallel. Higher is faster, but uses
                        more memory.""") #D.S: Maybe set down due to memory problems
    parser.add_argument('-batch_size_update', type=int, default=2048,
                        help='Maximum number of words per update') #D.S: TODO: ??
    parser.add_argument('-batch_size_multiplier', type=int, default=1,
                        help='Maximum number of words per update')  #D.S: TODO: The default is pretty low. Why?
    parser.add_argument('-max_position_length', type=int, default=1024,
        help='Maximum length for positional embedding') #D.S: TODO: Should be the same as dmodel

    parser.add_argument('-epochs', type=int, default=13,
                        help='Number of training epochs')
    parser.add_argument('-start_epoch', type=int, default=1,
                        help='The epoch from which to start')
    parser.add_argument('-param_init', type=float, default=0.1,
                        help="""Parameters are initialized over uniform distribution
                        with support (-param_init, param_init)""")
    parser.add_argument('-optim', default='adam',
                        help="Optimization method. [sgd|adagrad|adadelta|adam]")
    parser.add_argument('-max_grad_norm', type=float, default=0,
                        help="""If the norm of the gradient vector exceeds this,
                        renormalize it to have the norm equal to max_grad_norm""")
    parser.add_argument('-dropout', type=float, default=0.3,
                        help='Dropout probability; applied between LSTM stacks.')
    parser.add_argument('-word_dropout', type=float, default=0.0,
                        help='Dropout probability; applied on embedding indices.')
    parser.add_argument('-label_smoothing', type=float, default=0.0,
                        help='Label smoothing value for loss functions.')
    parser.add_argument('-scheduled_sampling_rate', type=float, default=0.0,
                        help='Scheduled sampling rate.')
    parser.add_argument('-curriculum', type=int, default=-1,
                        help="""For this many epochs, order the minibatches based
                        on source sequence length. Sometimes setting this to 1 will
                        increase convergence speed.""") #D.S: Ordering the source sequences in regard of their length brings the following advantage:
                        # Because of sequences on one batch need have the same length, they get padded with specal token. The amount of special tokens
                        # should be as low as possible, because of increased performance usage. Thats why they are sorted to keep all inputs of a batch nearly the same length
    parser.add_argument('-extra_shuffle', action="store_true",
                        help="""By default only shuffle mini-batch order; when true,
                        shuffle and re-assign mini-batches""") #D.S: Try No


    parser.add_argument('-normalize_gradient', action="store_true",
                        help="""Normalize the gradients by number of tokens before updates""")
    parser.add_argument('-virtual_gpu', type=int, default=1,
                        help='Number of virtual gpus. The trainer will try to mimic asynchronous multi-gpu training')
    # learning rate
    parser.add_argument('-learning_rate', type=float, default=1.0,
                        help="""Starting learning rate. If adagrad/adadelta/adam is
                        used, then this is the global learning rate. Recommended
                        settings: sgd = 1, adagrad = 0.1,
                        adadelta = 1, adam = 0.001""") #D.S: TODO: Set on 0.001
    parser.add_argument('-learning_rate_decay', type=float, default=1,
                        help="""If update_learning_rate, decay learning rate by
                        this much if (i) perplexity does not decrease on the
                        validation set or (ii) epoch has gone past
                        start_decay_at""")
    parser.add_argument('-start_decay_at', type=int, default=99999,
                        help="""Start decaying every epoch after and including this
                        epoch""")
    parser.add_argument('-warmup_steps', type=int, default=4096,
                        help="""Number of steps to increase the lr in noam""")
    parser.add_argument('-noam_step_interval', type=int, default=1,
                        help="""How many steps before updating the parameters""")

    parser.add_argument('-reset_optim', action='store_true',
                        help='Reset the optimizer running variables')
    parser.add_argument('-beta1', type=float, default=0.9,
                        help="""beta_1 value for adam""") #D.S: Fine due to tensor2tensor
    parser.add_argument('-beta2', type=float, default=0.98,
                        help="""beta_2 value for adam""") #D.S: Fine due to tensor2tensor
    parser.add_argument('-weight_decay', type=float, default=0.0,
                        help="""weight decay (L2 penalty)""") #D.S: Fine
    parser.add_argument('-amsgrad', action='store_true',
                        help='Using AMSGRad for adam')    
    parser.add_argument('-update_method', default='regular',
                        help="Type of update rule to use. Options are [regular|noam].")                                    
    # pretrained word vectors
    parser.add_argument('-tie_weights', action='store_true',
                        help='Tie the weights of the encoder and decoder layer')
    parser.add_argument('-join_embedding', action='store_true',
                        help='Jointly train the embedding of encoder and decoder in one weight') #D.S: TODO: ??
    parser.add_argument('-pre_word_vecs_enc',
                        help="""If a valid path is specified, then this will load
                        pretrained word embeddings on the encoder side.
                        See README for specific formatting instructions.""")
    parser.add_argument('-pre_word_vecs_dec',
                        help="""If a valid path is specified, then this will load
                        pretrained word embeddings on the decoder side.
                        See README for specific formatting instructions.""")

    # GPU
    parser.add_argument('-gpus', default=[], nargs='+', type=int,
                        help="Use CUDA on the listed devices.")
    parser.add_argument('-fp16', action='store_true',
                        help='Use half precision training')     
    parser.add_argument('-fp16_loss_scale', type=float, default=8,
                        help="""Loss scale for fp16 loss (to avoid overflowing in fp16).""")
    parser.add_argument('-seed', default=9999, type=int,
                        help="Seed for deterministic runs.")

    parser.add_argument('-log_interval', type=int, default=100,
                        help="Print stats at this interval.")
    parser.add_argument('-save_every', type=int, default=-1,
                        help="Save every this interval.")
    
    
    return parser
