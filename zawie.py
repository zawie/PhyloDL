from modelHandler import TrainAndTest, Test, Load
from Recombination.dataHandler import splitDataset, transformDataset, loadDataset, saveDataset
from models import dnn3
from Recombination.main import generateData
from ML.IQRAX import runIQTREE, runRAxML, runRAxMLClassification
import time

#Generate Data
dataset = generateData("Alpha",amountOfTrees=1000, numTrials=10, sequenceLength = 10000, anomolyOnly=False)
#Augment the data
transformedData = transformDataset(dataset)
#Split into train, dev, test
datasets = splitDataset(transformDataset,[89,1,10])
#Train and test model
model = dnn3()
DLAccuracy = TrainAndTest(model,datasets,1,"ConvNet1")
#Test IQTree
MLAccuracy = runIQTREE(datasets['test'])
#Print results
print("-"*100+f"\n\tDL:{DLAccuracy}\n\tML:{MLAccuracy}")
