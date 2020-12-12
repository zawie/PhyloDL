from torch.utils.data import Dataset
import numpy as np
import torch
import os
import pickle
from transformations import transformData, toYTensor
from datetime import datetime

def saveDataset(name,dataset):
    print("Saving dataset...")
    pickle.dump(dataset,open("data/"+name+".p","wb"))
    print("Dataset saved!")

def loadDataset(name):
    print("Loading dataset...")
    return pickle.load(open("data/"+name+".p","rb"))
    print("Dataset loaded!")

def splitDatasets(initialDatset, setProbabilities = [80, 5, 15]):
    """
    Forms SimpleDataset class datasets with the correct probabilities

    Input:
    setProbabilities - list of probabilities: [trainProb, devProb, testProb]

    Output:
    newSets - [trainSet, devSet, testSet]
    """
    assert sum(setProbabilities) == 100 #is a probability distribution

    numAllDatapoints = len(initialDatset.Y_data)
    newSets = list()
    indexCounter = 0
    for setProbability in setProbabilities:
        numDatapoints = int(setProbability/100 * numAllDatapoints)

        newData = initialDatset.X_data[indexCounter:indexCounter+numDatapoints]
        newLabels = initialDatset.Y_data[indexCounter:indexCounter+numDatapoints]

        newSets.append(SimpleDataset(newData, newLabels))

        indexCounter += numDatapoints

    (trainSet, devSet, testSet) = tuple(newSets)
    return  {"train":trainSet, "dev":devSet, "test":testSet}

class SimpleDataset(Dataset):
    def __init__(self, data, labels, doTransform=True):
        """
        Initializes the Dataset.
        This primarily entiles reading the generated sequeences into a python list

        Input:
        data - list of quartet sequeences
        labels - list of corresponding labesl
        """
        #Validate input
        assert len(data) == len(labels)

        #Create data fields
        self.X_data = []
        self.Y_data = []
        self.metadata = dict()
        self.writeToMetadata("Creation Date",datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
        if doTransform:
            #Transform data
            (X,Y) = transformData(data,labels)
            self.X_data = X
            self.Y_data = Y
        else:
            self.X_data = data
            self.Y_data = labels

        #Validate output
        assert len(self.X_data) == len(self.Y_data)

    def getMetadata(self):
        return self.metadata.copy()

    def writeToMetadata(self,key,value):
        self.metadata[key] = value

    def __getitem__(self, index):
        """
        Gets a certain tree across all three trees (alpha,beta,charlie)
        """
        return self.X_data[index], self.Y_data[index]

    def getData(self):
        """
        Gets class instances data and labels
        """
        return self.X_data, self.Y_data

    def __len__(self):
        """
        Returns the number of entries in this dataset
        """
        return len(self.X_data)

    def __add__(self, other):
        """
        Merges to datasets
        """
        (X_self,Y_self) = self.getData()
        (X_other,Y_other) = other.getData()
        return SimpleDataset(X_self + X_other, Y_self + Y_other, doTransform=False)

# if __name__ == "__main__":
#     dataPath = "/Users/rhuck/Downloads/DL_Phylogeny/Recombination/dataClassData/recombination_data0.npy"
#     labelsPath = "/Users/rhuck/Downloads/DL_Phylogeny/Recombination/dataClassData/recombination_labels0.npy"


#     X_data = np.load(dataPath, allow_pickle=True)
#     Y_data = np.load(labelsPath, allow_pickle=True)
#     data = X_data.tolist()
#     labels = Y_data.tolist()
#     dataset = SimpleDataset(data, labels)
#     data, labels = dataset.getData()
#     print("data: ", data)
#     print("labels: ", labels)
#     print("length: ", len(dataset))

#     print("=======================\n")
#     (trainSet, devSet, testSet) = dataset.formDatasets()
#     print("training set: ", trainSet.getData())
#     print("dev set: ", devSet.getData())
#     print("test set: ", testSet.getData())
