from modelHandler import TrainAndTest
from dataHandler import GenerateDatasets,GenerateMergedGTRDatasets,GenerateMergedSpecificDatasets
from models import dnn3,dnn3NoRes
from IQRAX import runRAxML,runIQTREE,runRAxMLClassification
from plotter import line
from evomodels import GTR as getRandomGTRValues

import datasetClass
import numpy as np

def getDataSets(dataPath, labelsPath):
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

    initialDataSet = datasetClass.SimpleDataset(X_Data, Y_Data)

    trainSet, devSet, testSet = initialDataSet.formDatasets()

    datasets = {"train":trainSet, "dev":devSet, "test":testSet}

    return datasets


#Run Sequence Length vs. Accuracy Test
NUM_EPOCHS = 3

dataPath = "/Users/rhuck/Downloads/DL_Phylogeny/Recombination/dataClassData/recombination_data0.npy"
labelsPath = "/Users/rhuck/Downloads/DL_Phylogeny/Recombination/dataClassData/recombination_labels0.npy"
datasets = getDataSets(dataPath, labelsPath)

#Define results dictionary
results = dict()

#DL Models Train & Testing
# convnet = dnn3()
# results['ConvNet (dnn3)']  = TrainAndTest(convnet,datasets,NUM_EPOCHS,f"ConvNet",doPlot=True)
# print(results)
resnet = dnn3NoRes()
results['ResNet (dnn3)']  = TrainAndTest(resnet,datasets,NUM_EPOCHS,f"ResNet",doPlot=True)
print(results)
