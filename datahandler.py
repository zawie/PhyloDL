import torch
import os
import sys
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

default_length = 200
default_amount = {"train":25000,"test":1,"dev":1000}

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
        os.system("seq-gen -m{m} -n{n} -l{l} <tree.tre> data/{type}.dat".format(m=m,n=n,l=length,type=key))
    print("Done Generating!")

def expandBatch(data,labels):
    batchsize = data.size()[0]
    expanded_data = torch.reshape(data,[batchsize*24,4,-1,4])
    expanded_labels = torch.reshape(labels,[batchsize*24])
    return expanded_data,expanded_labels

def listToString(lst):
    st = ""
    for char in lst:
        st += char
    return st

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

def unhotencode(sequence):
    code_map = {(1,0,0,0):"A",
                (0,1,0,0):"T",
                (0,0,1,0):"G",
                (0,0,0,1):"C"
            }
    final = []
    for char in sequence:
        final.append(code_map[tuple(char)])
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

def augment(instance):
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
        return len(self.instances)

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

Generate()

"""loader = DataLoader(dataset=test(), batch_size=1, shuffle=True)
for i in range(len(loader)):
    #Run model
    data,labels = next(iter(loader))
    data,labels = expandBatch(data,labels)
    data,labels = data.tolist(),labels.tolist()
    for i in range(len(labels)):
        label = labels[i]
        print(label)
        for sequence in data[i]:
            print(listToString(unhotencode(sequence)))
"""


