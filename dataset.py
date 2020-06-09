import torch
import os
from torch.utils.data import Dataset
trees = ["alpha","beta","charlie"]

def hotencode(sequence):
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

def generate(amount={"train":1000,"test":100,"dev":10}, m="HKY"):
    """
    Generate data:
        amount: dictionary (key=folder, value=n to generate)
    m: type of generation?
    """
    for key,n in amount.items():
        for tree in trees:
            os.system("seq-gen -m{m} -n{n} <trees/{tree}.tre> data/{type}/{tree}.dat".format(m=m,n=n,tree=tree,type=key))

class SequenceDataset(Dataset):
    def __init__(self,folder):
        #Define folder
        self.folder = folder
        #Define partitions
        self.partition = []
        for tree in trees:
            self.partition.append(self._num_entries(tree))

    def _getsequences(self,tree,index,doHotencode=True):
        file = open("data/{}/{}.dat".format(self.folder,tree),"r")
        startLine = index*5+1
        sequences = []
        for pos,line in enumerate(file):
            #Check if needed line
            diff = pos - (startLine)
            if diff >= 0 and diff < 4:
                #Trim excess characters
                sequence = line[15:-1]
                if doHotencode:
                    sequence = hotencode(sequence)
                sequences.append(sequence)
        file.close()
        return sequences

    def __getitem__(self,index):
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
        file = open("data/{}/{}.dat".format(self.folder,tree),"r")
        lines = len(file.read().split('\n'))
        entries = (lines-1)//5
        file.close()
        return entries

    def __len__(self):
        size = 0
        for tree in trees:
             size += self._num_entries(tree)
        return size


generate(amount={"dev":5,"train":100,"test":10})
dataset = SequenceDataset("dev")
print(dataset[0])