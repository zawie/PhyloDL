import torch
import os
import sys
from torch.utils.data import Dataset
trees = ["alpha","beta","charlie"]

#Instructions
    #To execute generate command from terminal simply type:
    #python dataHandler.py generate
def Generate(amount={"train":1000,"test":100,"dev":10}, length=100, m="HKY"):
    """
    Generate data:
        amount: dictionary (key=folder, value=n to generate)
        length: length of each sequence
        m: type of generation?
    """
    print("Generating...")
    for key,n in amount.items():
        for tree in trees:
            os.system("seq-gen -m{m} -n{n} -l{l} <trees/{tree}.tre> data/{type}/{tree}.dat".format(m=m,n=n,l=length,tree=tree,type=key))
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
    def __init__(self,folder):
        #Define folder
        self.folder = folder
        #Define partitions
        self.partition = []
        for tree in trees:
            self.partition.append(self._num_entries(tree))

    def _getsequences(self,tree,index,doHotencode=True):
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
                if doHotencode:
                    sequence = _hotencode(sequence)
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
        size = 0
        for tree in trees:
             size += self._num_entries(tree)
        return size

# Handler terminal prompt
if len(sys.argv) > 1 and sys.argv[1] == "generate":
    Generate()