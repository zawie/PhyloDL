from Recombination.datasetClass import SimpleDataset
from Recombination.main import generate as main_generate
import numpy as np
import torch

#Helpder Functions
def _getDatasets(dataPath, labelsPath):
    """
    1. Reads path files
    2. Forms SimpleDataset class
    3. Returns train, dev, test datasets in dictionary format
        {"train":trainSet, "dev":devSet, "test":testSet}
    """

    data = np.load(dataPath)
    labels = np.load(labelsPath)

    X_Data = data.tolist()
    Y_Data = labels.tolist()

    #Y and X should contain same amount of data
    assert(len(X_Data) == len(Y_Data))
    amount = len(X_Data)
    #Convert elements to tensors
    for i in range(amount):
        X_Data[i] = torch.tensor(X_Data[i],dtype=torch.float)
        Y_Data[i] = torch.tensor(Y_Data[i],dtype=torch.long)

    initialDataSet = SimpleDataset(X_Data, Y_Data)
    (trainSet, devSet, testSet) = initialDataSet.formDatasets()

    datasets = {"train":trainSet, "dev":devSet, "test":testSet}

    return datasets

#Public funcitons
def generateData():
    #In the future this needs to take parameters and generate data
    main_generate()
    pass

def getRecombinationDatasets(int):
    dataPath = f"Recombination/dataClassData/recombination_data{int}.npy"
    labelsPath = f"Recombination/dataClassData/recombination_labels{int}.npy"
    return _getDatasets(dataPath, labelsPath)