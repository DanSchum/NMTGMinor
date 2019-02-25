
PAD = 0
UNK = 1
BOS = 2
EOS = 3

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'

checkpointing = 0
static = False
residual_type = 'regular'
max_position_length = 512

cuda = 0
debug = 0
cudaActivated = 0

#D.S: Added for Coverage Mechanism
weightAvgProb = 0.00
weightWordFrequency = 0.00
weightStdSoftmax = 1 - (weightAvgProb + weightWordFrequency)