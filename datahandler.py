import torch
import os
import sys
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random
import dendropy
import treeClassifier
import hotEncoder

#Helper Function
def WriteToTre(txt):
    f = open("tree.tre", "w")
    f.write(txt)
    f.close()

def seq_gen(file,m="HKY",n=1,l=200,r=None,f=None):
    """
    Makes an os.system seq gen call
    file: file name to write to
    m: Model to generate under
    n: Number of sets of sequences to Generate
    l: Sequence length
    r: r_matrix
    f: f_matrix
    """
    if r!=None and f!=None:
        r_str = "_, "*(len(r)-1) + "_"
        f_str = "_, "*(len(f)-1) + "_"
        for i in range(6):
            r_str = r_str.replace("_",str(r[i]),1)
        for j in range(4):
            f_str = f_str.replace("_",str(f[j]),1)
        #print(f"\n\n\n\n\n\n\n\n\n-r{r_str}\n-f{f_str}\n\n\n\n\n\n\n\n")
        os.system(f'seq-gen -m{m} -n{n} -l{l} -r{r_str} -f{f_str} <tree.tre> {file}')
    elif r != None:
        r_str = "_, "*(len(r)-1) + "_"
        for i in range(6):
            r_str = r_str.replace("_",str(r[i]),1)
        os.system(f'seq-gen -m{m} -n{n} -l{l} -r{r_str} <tree.tre> {file}')
    else:
        os.system(f"seq-gen -m{m} -n{n} -l{l} <tree.tre> {file}")

def PureKingmanTreeConstructor(amount,pop_size=1,minimum=0.1,maximum=1):
    """
    Generates trees under the unconstrained Kingman’s coalescent process.
    amount: amount of trees to Create
    pop_size: some parameter of dendropy's pure_kingman_tree function
    minimum: minimum tolerable branch length
    maximum: maximum tolerable branch length

    Writes to the .tre file
    """
    TaxonNamespace = dendropy.TaxonNamespace(["A","B","C","D"])
    #Gemerate trees
    trees = []
    while len(trees) < amount:
        tree = dendropy.simulate.treesim.pure_kingman_tree(TaxonNamespace,pop_size)
        treeClass = treeClassifier.getClass(str(tree))
        #Only add alphas
        if treeClass == 0:
            #Remove if tree has too short branch Length
            invalid = False
            for edge in tree.edges():
                if (edge.length < minimum and edge.length > 0) or (edge.length > maximum):
                    invalid = True
                    break
            if not invalid:
                trees.append(tree)
    #Create string
    tre_str = ""
    for tree in trees:
        tre_str += str(tree) + ";\n"
    WriteToTre(tre_str)

#Generator
def Generate(file_name,amount,sequenceLength=200,model="HKY",r_matrix=None,f_matrix=None,pop_size=1):
    """
    Creates tree structures & generates sequences based off of htem
    """
    #Define structures
    PureKingmanTreeConstructor(amount,pop_size=pop_size)
    #Generate
    seq_gen(f"data/{file_name}.dat",m=model,n=1,l=sequenceLength,r=r_matrix,f=f_matrix)

def GenerateDatasets(amount_dictionary,sequenceLength=200,model="HKY",r_matrix=None,f_matrix=None,pop_size=1):
    """
    Creates tree structures, generates sequences, returns dataset, for each key in amount_dictionary
    """
    dataset_dictionary = dict()
    for key,amount in amount_dictionary.items():
        Generate(key,amount,sequenceLength=sequenceLength,mean=mean,std=std,model=model,r_matrix=r_matrix,f_matrix=f_matrix,pop_size=pop_size)
        dataset_dictionary[key] = NonpermutedDataset(key)
    return dataset_dictionary

#Sequence modifiers
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

#Readers
def getInstances(file_path):
    """
    Reads all seqeunces generated into a python list
    Inputs: file_path: which seq-gen .dat file should be read from
    Outputs: A list of lists of hotencoded sequences.
    """
    file = open(file_path,"r")
    data = []
    taxaDict = dict()
    for pos,line in enumerate(file):
        if pos%5 == 0:
            taxaDict = dict()
            data.append(list())
        else:
            #Trim
            taxaChar = line[0]
            sequence = line[10:-1]
            #Hot encode
            sequence = hotEncoder.encode(sequence)
            #Add sequence to dict
            taxaDict[taxaChar] = sequence
        if (pos+1)%5==0:
            data[pos//5] = [taxaDict['A'],taxaDict['B'],taxaDict['C'],taxaDict['D']]
            taxaDict = dict()
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

    def __add__(self, other):
        """
        Merges to datasets
        """
        return MergedSequenceDataset(self,other)

class MergedSequenceDataset(Dataset):
    def __init__(self,data0,data1):
        """
        """
        #assert(data0.expand == data1.expand)
        #assert(data0.preprocess and data1.preprocess)
        self.expand = data0.expand
        self.X_data = data0.X_data + data1.X_data
        self.Y_data = data0.Y_data + data1.Y_data

    def __getitem__(self,index):
        """
        Gets a certain tree across all three trees (alpha,beta,charlie)
        """
        return self.X_data[index],self.Y_data[index]

    def __len__(self):
        """
        Returns the number of entries in this dataset
        """
        return len(self.X_data)

    def __add__(self, other):
        """
        Merges to datasets
        """
        return MergedSequenceDataset(self,other)

def NonpermutedDataset(folder,preprocess=True):
    """
    Returns a SequenceDataset that will ONLY transforme ach tree instance
    This will grow the dataset by 3
    """
    def _augment(instance):
        X = list()
        y = list()
        y.append(0)
        X.append(instance)
        y.append(1)
        X.append(toBeta(instance))
        y.append(2)
        X.append(toGamma(instance))
        return torch.tensor(X,dtype=torch.float),torch.tensor(y,dtype=torch.long)
    def _expand(data,labels):
        batchsize = data.size()[0]
        expanded_data = torch.reshape(data,[batchsize*3,4,-1,4])
        expanded_labels = torch.reshape(labels,[batchsize*3])
        return expanded_data,expanded_labels
    return SequenceDataset(folder,_augment,_expand,preprocess=preprocess)

# def PermutedDataset(folder,preprocess=True):
#     """
#     Returns a SequenceDataset that will transform and permute each tree instance
#     This will grow the dataset by 24
#     """
#     print("ONLY PERMUTES SYMMETRIC CORRECTLY!!!!")
#     def _augment(instance):
#         X = list()
#         y = list() #alpha + beta + gamma
#         for alpha in permute(instance):
#             X.extend(alpha)
#             X.extend(toBeta(alpha))
#             X.extend(toGamma(alpha))
#             y.extend([0,1,2])
#         return torch.tensor(X,dtype=torch.float),torch.tensor(y,dtype=torch.long)
#     def _expand(data,labels):
#         batchsize = data.size()[0]
#         expanded_data = torch.reshape(data,[batchsize*24,4,-1,4])
#         expanded_labels = torch.reshape(labels,[batchsize*24])
#         return expanded_data,expanded_labels
#     return SequenceDataset(folder,_augment,_expand,preprocess=preprocess)
