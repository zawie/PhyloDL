import torch
import os
import sys
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random

#Constants
TREES = ['alpha','beta','gamma']

#Helper Function
def WriteToTre(txt):
    f = open("tree.tre", "w")
    f.write(txt)
    f.close()

def seq_gen(file,m="HKY",n=1,l=200):
    os.system(f"seq-gen -m{m} -n{n} -l{l} <tree.tre> {file}")

#Generator
def GenerateTrees(amount={"train":2500,"test":1000,"dev":100}, sequenceLength=200, std=0,mean=0.5, model="HKY", symmetricOnly=False):
    #Define structures
    template_trees = ["((A:_,B:_):_,(C:_,D:_):_)",
                      "(((A:_,B:_):_,C:_):_,D:_)",
                      "(A:_,(B:_,(C:_,D:_):_):_)",
                      "(((A:_,B:_):_,D:_):_,C:_)",
                      "(B:_,(A:_,(C:_,D:_):_):_)",
                      ]
    if symmetricOnly:
        template_trees = ["((A:_,B:_):_,(C:_,D:_):_)"]
    #Create as many structures
    for key,n in amount.items():
        tre_str = ""
        for i in range(n):
            tree = template_trees[i%len(template_trees)]
            for _ in range(6):
                r = max(0.001,random.gauss(mean,std))
                tree = tree.replace("_",str(r),1)
            tre_str += tree + ";\n"
        WriteToTre(tre_str)
        #Generate
        seq_gen(f"data/{key}.dat",m=model,n=1,l=sequenceLength)

#Sequence modifiers
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

def toBeta(alphaSeqeunces):
    """
    Transforms a given alpha sequences to a beta tree
    """
    (A,B,C,D) = alphaSeqeunces
    return [A,D,C,B]

def toGamma(alphaSeqeunces):
    """
    Transforms a given alpha sequences to a gamma tree
    """
    (A,B,C,D) = alphaSeqeunces
    return [A,C,B,D]

def permute(sequences):
    """
    Permutes the set of sequences into all possible orders that maintains
    the same tree class
    """
    (A,B,C,D) = sequences
    return [
        [A,B,C,D],
        [A,B,D,C],
        [B,A,C,D],
        [B,A,D,C],
        [C,D,A,B],
        [C,D,B,A],
        [D,C,A,B],
        [D,C,B,A]
    ]

#Readers
def getInstances(file_path,tree='alpha'):
    """
    Reads all seqeunces generated into a python list
    Inputs: file_path: which seq-gen .dat file should be read from
            tree: the type of tree that's being read (VERY IMPORTANT)
    Outputs: A list of lists of hotencoded sequences.
    """
    file = open(file_path,"r")
    data = []
    for pos,line in enumerate(file):
        if pos%5 == 0:
            data.append(list())
        else:
            #Trim
            sequence = line[10:-1]
            #Hot encode
            sequence = hotencode(sequence)
            #Add sequence to list
            data[pos//5].append(sequence)
        if (pos+1)%5==0:
            #refactor
            if tree == "beta":
                (A,D,C,B) = data[pos//5]
                data[pos//5] = [A,B,C,D]
            elif tree == 'gamma':
                (A,C,B,D) = data[pos//5]
                data[pos//5] = [A,B,C,D]
    file.close()
    return data

#Datasets
class SequenceDataset(Dataset):
    def __init__(self,folder,augment_function,expand_function,preprocess=True):
        """
        Initializes the Dataset.
        This primarily entiles reading the generated sequeences into a python list
        """
        #Define constants
        self.folder = folder
        self.preprocess = preprocess
        self.instances = getInstances(f"data/{folder}.dat")
        self._augment = augment_function
        self.expand = expand_function
        #Preprocess
        if preprocess:
            self.X_data = list()
            self.Y_data = list()
            for instance in self.instances:
                X,y = self._augment(instance)
                self.X_data.append(X)
                self.Y_data.append(y)

    def __getitem__(self,index):
        """
        Gets a certain tree across all three trees (alpha,beta,charlie)
        """
        if self.preprocess:
            return self.X_data[index],self.Y_data[index]
        else:
            X,y = self._augment(self.instances[index])
            return X,y

    def __len__(self):
        """
        Returns the number of entries in this dataset
        """
        return len(self.instances)

def PermutedDataset(folder,preprocess=True):
    """
    Returns a SequenceDataset that will transform and permute each tree instance
    This will grow the dataset by 24
    """
    def _augment(instance):
        X = list()
        y = list() #alpha + beta + gamma
        for alpha in permute(instance):
            y.append(0)
            X.extend(alpha)
            y.append(1)
            X.extend(toBeta(alpha))
            y.append(2)
            X.extend(toGamma(alpha))
        return torch.tensor(X,dtype=torch.float),torch.tensor(y,dtype=torch.long)
    def _expand(data,labels):
        batchsize = data.size()[0]
        expanded_data = torch.reshape(data,[batchsize*24,4,-1,4])
        expanded_labels = torch.reshape(labels,[batchsize*24])
        return expanded_data,expanded_labels
    return SequenceDataset(folder,_augment,_expand,preprocess=preprocess)

def UnpermutedDataset(folder,preprocess=True):
    """
    Returns a SequenceDataset that will ONLY transforme ach tree instance
    This will grow the dataset by 3
    """
    def _augment(instance):
        X = list()
        y = list()
        y.append(0)
        X.extend(instance)
        y.append(1)
        X.extend(toBeta(instance))
        y.append(2)
        X.extend(toGamma(instance))
        return torch.tensor(X,dtype=torch.float),torch.tensor(y,dtype=torch.long)
    def _expand(data,labels):
        batchsize = data.size()[0]
        expanded_data = torch.reshape(data,[batchsize*3,4,-1,4])
        expanded_labels = torch.reshape(labels,[batchsize*3])
        return expanded_data,expanded_labels
    return SequenceDataset(folder,_augment,_expand,preprocess=preprocess)
