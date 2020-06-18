import torch
import os
import sys
from torch.utils.data import Dataset

trees = ["alpha","beta","charlie"]
default_length = 1000
default_amount = {"train":50000,"test":5000,"dev":1000}
def Generate(amount=default_amount, length=default_length, m="HKY",TSR=0.5):
    """
    Generate data:
        amount: dictionary (key=folder, value=n to generate)
        length: length of each sequence
        m: type of generation? (JC69 = Juke's Cantor)
        TSR: the transition transversion ratio 
        NOTE: for any given amount, triple the number of sequences will be generated (one for reach tree type)
    """
    print("Generating...")
    for key,n in amount.items():
        for tree in trees:
            os.system("seq-gen -m{m} -n{n} -l{l} -t{t} <trees/{tree}.tre> data/{type}/{tree}.dat".format(m=m,n=n,l=length,t=TSR,tree=tree,type=key))
    print("Done Generating!")

def _hotencode(sequence):
    """ 
        Hot encodes inputted sequnce
        "ATGC" -> [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    """
    code_map = {"A":[1,0,0,0],
                "T":[0,1,0,0],
                "G":[0,0,1,0],
                "C":[0,0,0,1]}
    final = []
    for char in sequence:
        final.append(code_map[char])
    return final

class SequenceDataset(Dataset):
    def __init__(self,folder,doHotencode=True,preprocess=True):
        #Define constants
        self.folder = folder
        self.doHotencode = doHotencode
        self.preprocess = preprocess
        #Define partitions
        self.partition = []
        for tree in trees:
            self.partition.append(self._num_entries(tree))
        #Preprocess
        if preprocess:
            #print("Preprocessing {}...".format(folder))
            self.X_data = list()
            self.Y_data = list()
            for t in range(3):
                tree = trees[t]
                data = self._readAllSequences(tree)
                self.X_data.extend(data)
                self.Y_data.extend([t]*len(data))

    def _readAllSequences(self,tree):
        file = open("data/{}/{}.dat".format(self.folder,tree),"r")
        data = []
        for pos,line in enumerate(file):
            if pos%5 == 0:
                data.append(list())
            else:
                sequence = line[10:-1]
                #Hot encode
                if self.doHotencode:
                    sequence = _hotencode(sequence)
                #Add sequence to list
                data[pos//5].append(sequence)
                #Convert to Tensor
                if (pos+1)%5 == 0: 
                    data[pos//5] = torch.Tensor(data[pos//5])
        file.close()
        return data

    def _readsequences(self,tree,index):
        """
        Reads the sequences from a certain tree's file and index
        """
        file = open("data/{}/{}.dat".format(self.folder,tree),"r")
        startLine = index*5+1
        sequences = []
        for pos,line in enumerate(file):
            #Check if needed line
            diff = pos - (startLine)
            if diff >= 0 and diff < 4:
                #Trim excess characters
                sequence = line[10:-1]
                #Hot encode
                if self.doHotencode:
                    sequence = _hotencode(sequence)
                #Add sequence to list
                sequences.append(sequence)
            elif pos > startLine + 5:
                break
        file.close()
        return sequences

    def _getsequences(self,index):
        for t in range(3):
            partition_size = self.partition[t]
            if index < partition_size:
                #return appropriate data
                tree = trees[t]
                sequences = self._readsequences(tree,index)
                return torch.Tensor(sequences),t
            else:
                #progress to next tree.dat file and reduce index accordingly
                index -= partition_size
    
    def __getitem__(self,index):
        """
        Gets a certain tree across all three trees (alpha,beta,charlie)
        """
        if self.preprocess:
            return self.X_data[index],self.Y_data[index]
        else:
            return self._getsequences(index)
    
    
    def _num_entries(self,tree):
        """
        Counts the number of entries for a given tree
        (Should always be 1/3 of the total len)
        """
        file = open("data/{}/{}.dat".format(self.folder,tree),"r")
        lines = len(file.read().split('\n'))
        entries = (lines-1)//5
        file.close()
        return entries

    def __len__(self):
        """
        Returns the number of entries in this dataset 
        """
        return sum(self.partition)

# Shorthand access
def train(preprocess=True):
    return SequenceDataset("train",preprocess=preprocess)
def test(preprocess=True):
    return SequenceDataset("test",preprocess=preprocess)
def dev(preprocess=True):
    return SequenceDataset("dev",preprocess=preprocess)

# Handler terminal prompt
if len(sys.argv) > 1 and sys.argv[1] == "generate":
    length = default_length
    amount = default_amount.copy()
    if len(sys.argv) > 2:
        length = sys.argv[2]
        if len(sys.argv) > 3:
            amount["train"] = sys.argv[3]
            if len(sys.argv) > 4:
                amount["test"] = sys.argv[4]
                if len(sys.argv) > 5:
                    amount["dev"] = sys.argv[5]
    print("Generating Sequence triplets of length {length} with the following amount:{amount}".format(length=length,amount=amount))
    Generate(amount=amount,length=length)

#Generate()

val = dev(preprocess=False)

print(val[0])
print(val[0][0].shape)