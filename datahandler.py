import torch
import os
import sys
from torch.utils.data import Dataset
trees = ["alpha","beta","charlie"]
default_length = 1000
default_amount = {"train":10000,"test":1000,"dev":100}
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
    def __init__(self,folder,doHotencode=True):
        #Define folder
        self.folder = folder
        self.doHotencode = doHotencode
        #Define partitions
        self.partition = []
        for tree in trees:
            self.partition.append(self._num_entries(tree))

    def _getsequences(self,tree,index,):
        """
        Returns the sequences of a certain tree and index
        """
        file = open("data/{}/{}.dat".format(self.folder,tree),"r")
        startLine = index*5+1
        sequences = []
        for pos,line in enumerate(file):
            #Check if needed line
            diff = pos - (startLine)
            if diff >= 0 and diff < 4:
                #Trim excess characters
                sequence = line[15:-1]
                #Hot encode
                if self.doHotencode:
                    sequence = _hotencode(sequence)
                #Add sequence to list
                sequences.append(sequence)
        file.close()
        return sequences

    def __getitem__(self,index):
        """
        Gets a certain tree across all three trees (alpha,beta,charlie)
        """
        for t in range(3):
            partition_size = self.partition[t]
            if index < partition_size:
                #return appropriate data
                tree = trees[t]
                sequences = self._getsequences(tree,index)
                return sequences,t
            else:
                #progress to next tree.dat file and reduce index accordingly
                index -= partition_size
    
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
def train():
    return SequenceDataset("trian")
def test():
    return SequenceDataset("test")
def dev():
    return SequenceDataset("dev")
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