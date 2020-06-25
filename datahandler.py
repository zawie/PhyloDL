import torch
import os
import sys
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random

TREES = ['alpha','beta','gamma']
default_length = 200
default_amount = {"train":2500,"test":1000,"dev":100}

def WriteRandomTrees(mean,std,amount=100):
    print("Modifying branch length...")
    template_tree = "((A:_,B:_):_,(C:_,D:_):_)"
    tree_str = ""
    for _ in range(amount):
        tree = template_tree
        for _ in range(6):
            r = max(0.001,random.gauss(mean,std))
            tree = tree.replace("_",str(r),1)
        tree_str += tree + ";\n"
    f = open("tree.tre", "w")
    f.write(tree_str)
    f.close()
    print(f"Random branched trees generated!")

def SetBranchLength(length):
    print("Modifying branch length...")
    template_tree = "((A:_,B:_):_,(C:_,D:_):_)"
    new_tree = template_tree.replace("_",str(length))
    f = open("tree.tre", "w")
    f.write(new_tree)
    f.close()
    print(f"Branch length set to {length}!")

def Generate(amount=default_amount, b=.1, l=default_length, m="HKY",TSR=0.5):
    """
    Generate data:
        amount: dictionary (key=folder, value=n to generate)
        length: length of each sequence
        m: type of generation? (JC69 = Juke's Cantor)
        TSR: the transition transversion ratio
        NOTE: for any given amount, triple the number of sequences will be generated (one for reach tree type)
    """
    print("Generating...")
    SetBranchLength(b)
    for key,n in amount.items():
        os.system(f"seq-gen -m{m} -n{n} -l{l} <tree.tre> data/{key}.dat")
    print("Done Generating!")

def GenerateRandomBranchLengths(amount=default_amount, l=default_length, std=1,mean=0.5, m="HKY"):
    print("Random Generating...")
    for key,n in amount.items():
        WriteRandomTrees(mean,std,amount=n)
        os.system(f"seq-gen -m{m} -n{1} -l{l} <tree.tre> data/{key}.dat")
    print("Done Generating!")

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
    (A,B,C,D) = alphaSeqeunces
    return [A,D,C,B]

def toGamma(alphaSeqeunces):
    (A,B,C,D) = alphaSeqeunces
    return [A,C,B,D]

def permute(sequences):
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

def getInstances(file_path,tree='alpha'):
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

class SequenceDataset(Dataset):
    def __init__(self,folder,augment_function,expand_function,preprocess=True):
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
