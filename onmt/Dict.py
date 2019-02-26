import torch
import onmt.Constants

class Dict(object):
    def __init__(self, data=None, lower=False):
        self.idxToLabel = {}
        self.labelToIdx = {}
        self.frequencies = {}
        self.lower = lower

        # Special entries will not be pruned.
        self.special = []

        if data is not None:
            if type(data) == str:
                self.loadFile(data)
            else:
                self.addSpecials(data)

    def size(self):
        return len(self.idxToLabel)

    def loadFile(self, filename):
        "Load entries from a file."
        for line in open(filename):
            fields = line.split()
            label = fields[0]
            idx = int(fields[1])
            self.add(label, idx)
        
        

    def writeFile(self, filename):
        "Write entries to a file."
        with open(filename, 'w') as file:
            for i in range(self.size()):
                label = self.idxToLabel[i]
                file.write('%s %d\n' % (label, i))

        file.close()

    def lookup(self, key, default=None):
        key = key.lower() if self.lower else key
        try:
            return self.labelToIdx[key]
        except KeyError:
            return default

    def getLabel(self, idx, default=None):
        try:
            return self.idxToLabel[idx]
        except KeyError:
            return default

    def addSpecial(self, label, idx=None):
        "Mark this `label` and `idx` as special (i.e. will not be pruned)."
        idx = self.add(label, idx)
        self.special += [idx]

    def addSpecials(self, labels):
        "Mark all labels in `labels` as specials (i.e. will not be pruned)."
        for label in labels:
            self.addSpecial(label)

    def add(self, label, idx=None):
        "Add `label` in the dictionary. Use `idx` as its index if given."
        label = label.lower() if self.lower else label
        if idx is not None:
            self.idxToLabel[idx] = label
            self.labelToIdx[label] = idx
        else:
            if label in self.labelToIdx:
                idx = self.labelToIdx[label]
            else:
                idx = len(self.idxToLabel)
                self.idxToLabel[idx] = label
                self.labelToIdx[label] = idx

        if idx not in self.frequencies:
            self.frequencies[idx] = 1
        else:
            self.frequencies[idx] += 1

        return idx

    def prune(self, size):
        "Return a new dictionary with the `size` most frequent entries."
        if size >= self.size():
            return self

        # Only keep the `size` most frequent entries.
        freq = torch.Tensor(
                [self.frequencies[i] for i in range(len(self.frequencies))])
        _, idx = torch.sort(freq, 0, True)

        newDict = Dict()
        newDict.lower = self.lower
        
        count = 0
        # Add special entries in all cases.
        for i in self.special:
            newDict.addSpecial(self.idxToLabel[i])
            count = count + 1

        for i in idx.tolist():
            newDict.add(self.idxToLabel[i])
            count = count + 1
            
            if count >= size:
                break

        return newDict

    def convertToIdx(self, labels, unkWord, bosWord=None, eosWord=None):
        '''
        Convert `labels` to indices. Use `unkWord` if not found.
        Optionally insert `bosWord` at the beginning and `eosWord` at the end.
        Returns list/array of labels (e.g. a sentence) with BOS at beginning, then the tokens and EOS at the end.

        :param labels: List of labels which should be added to Vocabulary
        :param unkWord: Token for unknown word
        :param bosWord: Token for begin of sentence
        :param eosWord: Token for end of sentence
        :return:
        '''
        vec = []

        if bosWord is not None:
            vec += [self.lookup(bosWord)]

        unk = self.lookup(unkWord) #Check if unkown word token is already part of vocabulary
        #vec += adds array to array
        vec += [self.lookup(label, default=unk) for label in labels] #Check for all labels in list of labels if they
                    # already part of the vocabulary. If not it returns unkown token, which is then added to cev list

        if eosWord is not None:
            vec += [self.lookup(eosWord)] #Add EOS at the end of sequence

        return torch.LongTensor(vec) #Returns array with whole sequence including special tokens

    def convertToLabels(self, idx, stop):
        """
        Convert `idx` to labels.
        If index `stop` is reached, convert it and return.
        """
        #~ print(self.idxToLabel)
        labels = []

        for i in idx:
            
            labels += [self.getLabel(int(i))]
            if i == stop:
                break

        return labels

    def createWordFrequencyModel(self, srcBatch, lenTargetVocabulary, unkWord):
        '''
        Create wordFrequencyModel on word frequencies from sequence. wordFrequencyModel is based on target vocabulary.
        :param sequence:
        :param unkWord:
        :return:
        '''
        vec = []
        wordFrequencyModel = torch.zeros(len(srcBatch), lenTargetVocabulary)
        if onmt.Constants.cudaActivated:
            print('Word Fre is cuda')
            #swordFrequencyModel = wordFrequencyModel.cuda()

        unk = self.lookup(unkWord)

        for index, sequence in enumerate(srcBatch):
            vec += [self.lookup(label, default=unk) for label in sequence]

            for word in vec:
                wordFrequencyModel[index, word] = wordFrequencyModel[index, word] + 1
            wordFrequencyModel[index] = wordFrequencyModel[index] / len(vec)

        return wordFrequencyModel