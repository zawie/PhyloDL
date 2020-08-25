import torch
import os
import sys
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random
import dendropy
import treeClassifier
import hotEncoder
from transformations import transformSequences

#File path lambdas
getDatPath = lambda folderName: f'data/{folderName}/sequences.dat'
getTrePath = lambda folderName: f'data/{folderName}/trees.tre'

#Readers
def getSequenceSets(file_path):
    """
    Reads all seqeunces generated into a python list
    Inputs: file_path: which seq-gen .dat file should be read from
    Outputs: A list of tensors of hotencoded sequences.
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
            #Store into data
            properOrder = [taxaDict['A'],taxaDict['B'],taxaDict['C'],taxaDict['D']]
            tensor = torch.tensor(properOrder,dtype=torch.float)
            data[pos//5] = tensor
            #Clear dict
            taxaDict = dict()
    file.close()
    return data

def getTreeLabels(file_path):
    """
    Gets all the labels of a given .tre file, in order of appearence
    Inputs: file_path: the .tre file to extract classes from
    Outputs: A list of tree tensors
    """

    file = open(file_path,"r")
    labels = []
    for pos,line in enumerate(file):
        treeClass = treeClassifier.getClass(line)
        treeClassTensor = torch.tensor(treeClass,dtype=torch.long)
        labels.append(treeClassTensor)
    file.close()
    return labels

#Datasets
class SequenceDataset(Dataset):
    def __init__(self,folderName,doTransform=False):
        """
        Initializes the Dataset.
        This primarily entiles reading the generated sequeences into a python list
        """
        #Define constants
        self.folders = [folderName]
        sequenceSets = getSequenceSets(getDatPath(folderName))
        treeLabels =  getTreeLabels(getTrePath(folderName))
        if doTransform:
            self.X_data = list()
            self.Y_data = list()
            for (sequences,label) in zip(sequenceSets,treeLabels):
                (transformedSequences,labels) = transformSequences(sequences,label)
                self.X_data.extend(transformedSequences)
                self.Y_data.extend(labels)
        else:
            self.X_data = sequenceSets
            self.Y_data = treeLabels

    def __getitem__(self,index):
        """
        Gets a certain tree across all three trees (alpha,beta,charlie)
        """
        print("Y DATA:",self.Y_data[index])
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
        self.folders += other.folders
        self.X_data += other.X_data
        self.Y_data += other.Y_data
        return self

#Helper Function
def seq_gen(folderName,m="HKY",n=1,l=200,r=None,f=None):
    """
    Makes an os.system seq gen call
    file: file name to write to
    m: Model to generate under
    n: Number of sets of sequences to Generate
    l: Sequence length
    r: r_matrix
    f: f_matrix
    """
    TRE_FILE = getTrePath(folderName)
    DAT_FILE = getDatPath(folderName)
    if r!=None and f!=None:
        r_str = r
        f_str = f
        if type(r) != type(str()):
            r_str = "_, "*(len(r)-1) + "_"
            f_str = "_, "*(len(f)-1) + "_"
            for i in range(6):
                r_str = r_str.replace("_",str(r[i]),1)
            for j in range(4):
                f_str = f_str.replace("_",str(f[j]),1)
        #print(f"\n\n\n\n\n\n\n\n\n-r{r_str}\n-f{f_str}\n\n\n\n\n\n\n\n")
        os.system(f'seq-gen -m{m} -n{n} -l{l} -r{r_str} -f{f_str} <{TRE_FILE}> {DAT_FILE}')
    elif r != None:
        r_str = "_, "*(len(r)-1) + "_"
        for i in range(6):
            r_str = r_str.replace("_",str(r[i]),1)
        os.system(f'seq-gen -m{m} -n{n} -l{l} -r{r_str} <{TRE_FILE}> {DAT_FILE}')
    else:
        os.system(f"seq-gen -m{m} -n{n} -l{l} <{TRE_FILE}> {DAT_FILE}")

def PureKingmanTreeConstructor(tre_path,amount,pop_size=1,minimum=0.1,maximum=1):
    """
    Generates trees under the unconstrained Kingman’s coalescent process.
    amount: amount of trees to Create
    pop_size: some parameter of dendropy's pure_kingman_tree function
    minimum: minimum tolerable branch length
    maximum: maximum tolerable branch length

    Write to a .tre file
    """
    TaxonNamespace = dendropy.TaxonNamespace(["A","B","C","D"])
    #Gemerate trees
    trees = []
    while len(trees) < amount:
        tree = dendropy.simulate.treesim.pure_kingman_tree(TaxonNamespace,pop_size)
        #Remove if tree has too short branch Length
        invalid = False
        for edge in tree.edges():
            if (edge.length < minimum and edge.length != 0) or (edge.length > maximum):
                invalid = True
                break
        if not invalid:
            trees.append(tree)
    #Create string
    tre_str = ""
    for tree in trees:
        tre_str += str(tree) + ";\n"
    #Write to tre file
    with open(tre_path, "w") as f:
        f.write(tre_str)

def GenerateDatasets(amount_dictionary,sequenceLength=200,model="HKY",r_matrix=None,f_matrix=None,pop_size=1):
    """
    Creates tree structures, generates sequences, returns dataset, for each key in amount_dictionary
    """
    dataset_dictionary = dict()
    for folderName,amount in amount_dictionary.items():
        #Check if directory needs to be created & create it
        if not os.path.exists(f"data/{folderName}"):
            os.mkdir(f"data/{folderName}")
        #Generate
        ##Define structures
        PureKingmanTreeConstructor(getTrePath(folderName),amount,pop_size=pop_size)
        ##Call seq-gen
        seq_gen(folderName,m=model,n=1,l=sequenceLength,r=r_matrix,f=f_matrix)
        #Generate and save dataset
        dataset_dictionary[folderName] = SequenceDataset(folderName)
    return dataset_dictionary

def GenerateMergedGTRDatasets(amount_dictionary,models,sequenceLength=200,pop_size=1):
    merged_dicionary = dict()
    for i in range(len(models)):
        #Extrace frequencies from model
        (base_freq,rate_mx) = models[i]
        #Generate datasets
        dataset_dictionary = GenerateDatasets(amount_dictionary,sequenceLength=sequenceLength,model="GTR",r_matrix=rate_mx,f_matrix=base_freq,pop_size=pop_size)
        for name,dataset in dataset_dictionary.items():
            #Default dictioanry, more or less
            if i > 0:
                merged_dicionary[name] += dataset
            else:
                merged_dicionary[name] = dataset
    return merged_dicionary

def GenerateMergedSpecificDatasets(amount_dictionary,model_dictionary,sequenceLength=200,pop_size=1):
    mergedData = {}
    #Create and merge all data
    for name,settings in model_dictionary.items():
        #Generate data
        datasets = None
        m = settings['m']
        if m == 'JC':
            datasets = GenerateDatasets(amount_dictionary,sequenceLength=sequenceLength,pop_size=pop_size)
        else:
            datasets = GenerateDatasets(amount_dictionary,sequenceLength=sequenceLength,model=m,r_matrix=settings['r'],f_matrix=settings['f'],pop_size=pop_size)
        for key, dataset in datasets.items():
            if key in mergedData:
                mergedData[key] += dataset
            else:
                mergedData[key] = dataset
    return mergedData
