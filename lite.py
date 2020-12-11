from modelHandler import TrainAndTest
from Recombination.dataHandler import splitDatasets, loadDataset
from util.plotter import line
from models import dnn3
from Recombination.main import generateData
from ML.IQRAX import runIQTREE, runRAxML, runRAxMLClassification
import time

#Generate data
#dataset = generateData(name="Alpha",amountOfTrees=100, numTrials=10)
dataset = loadDataset("Alpha")
datasets = splitDatasets(dataset,setProbabilities = [80, 5, 15])

#Compute accuacy using ML
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