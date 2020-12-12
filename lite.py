from modelHandler import TrainAndTest
from Recombination.dataHandler import splitDatasets, loadDataset
from util.plotter import line
from models import dnn3
from Recombination.main import generateData
from ML.IQRAX import runIQTREE, runRAxML, runRAxMLClassification
import time


dataset = generateData(name="Ryan",amountOfTrees=10, numTrials=10, anomolyOnly=False)

#Generate data
# dataset = generateData(name="Alpha10k",amountOfTrees=1000, numTrials=10)
# dataset = generateData(name="Alpha50k",amountOfTrees=2500, numTrials=20)
# dataset = generateData(name="Alpha100k",amountOfTrees=5000, numTrials=40)

dataset = loadDataset("Alpha10k")
datasets = splitDatasets(dataset,setProbabilities = [80, 5, 15])

# #Compute accuacy using ML
MLaccuracy = runIQTREE(datasets['test'])
print("-"*100+"\n ML Accuracy:")
print(MLaccuracy)

#Compute accuray using DL
DLaccuracy = TrainAndTest(dnn3(),datasets,3,f"ConvNet",doPlot=True)

#Compute and print
print("-"*100+"\n ML Accuracy:")
print(MLaccuracy)
print("\n DL Accuracy:")
print(DLaccuracy)
