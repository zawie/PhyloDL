from datasetClass import SimpleDataset
import numpy as np

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

    initialDataSet = SimpleDataset(X_Data, Y_Data)

    trainSet, devSet, testSet = initialDataSet.formDatasets()

    datasets = {"train":trainSet, "dev":devSet, "test":testSet}

    return datasets

#Public funcitons
def generateData():
    #In the future this needs to take parameters and generate data
    pass

def getRecombinationDatasets(int):
    dataPath = f"/dataClassData/recombination_data{int}.npy"
    labelsPath = f"/dataClassData/recombination_labels{int}.npy"
    return _getDatasets(dataPath, labelsPath)