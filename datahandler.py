import torch
import os
import sys
from torch.utils.data import Dataset

trees = ["alpha","beta","charlie"]
default_length = 250
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
        os.system("seq-gen -m{m} -n{n} -l{l} -t{t} <tree.tre> data/{type}.dat".format(m=m,n=n,l=length,t=TSR,type=key))
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

def toBeta(alpaSeqeunces):
    (A,B,C,D) = alpaSeqeunces
    return [A,D,C,B]

def toGamma(alpaSeqeunces):
    (A,B,C,D) = alpaSeqeunces
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

def augment(instance):
    X = list()
    y = [0]*8 + [1]*8 + [2]*8
    X.extend(permute(instance))
    X.extend(permute(toBeta(instance)))
    X.extend(permute(toGamma(instance)))
    return torch.Tensor(X),torch.Tensor(y)
    
class SequenceDataset(Dataset):
    def __init__(self,folder,preprocess=True):
        #Define constants
        self.folder = folder
        self.preprocess = preprocess
        self.instances = self._getAllInstances()
        #Preprocess
        if preprocess:
            self.X_data = list()
            self.Y_data = list()
            for instance in self.instances:
                X,y = augment(instance)
                self.X_data.append(X)
                self.Y_data.append(y)

    def _getAllInstances(self):
        file = open("data/{}.dat".format(self.folder),"r")
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
        file.close()
        return data

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
        return len(self.instances)*24

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

val = dev()
print(len(val))
print(val[0])
#Generate()
