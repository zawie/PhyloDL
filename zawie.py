from modelHandler import TrainAndTest, Test, Load
from Recombination.dataHandler import splitDatasets, loadDataset, saveDataset
from util.plotter import line
from models import dnn3
from Recombination.main import generateData
from ML.IQRAX import runIQTREE, runRAxML, runRAxMLClassification
import time

dataset = generateData(name="Anomaly10k",amountOfTrees=1000, numTrials=10, anomolyOnly=True)

# Compute accuray using DL
model = dnn3()
Load(model,"-ConvNet-Epoch2",doPrint=True)
DLaccuracy = Test(model,dataset)

# Computer accuracy using ML
MLaccuracy = runIQTREE(dataset)

#Print Results
print("-"*100+"\n ML Accuracy:")
print(MLaccuracy)
print("\n DL Accuracy:")
print(DLaccuracy)








# #dataset = generateData(name="Alpha10k",amountOfTrees=1000, numTrials=10)

# #Generate data
# # dataset = generateData(name="Alpha10k",amountOfTrees=1000, numTrials=10)
# # dataset = generateData(name="Alpha50k",amountOfTrees=2500, numTrials=20)
# # dataset = generateData(name="Alpha100k",amountOfTrees=5000, numTrials=40)

# dataset = loadDataset("Alpha10k")
# #datasets = splitDatasets(dataset,setProbabilities = [846, 4, 150])

# #Compute accuray using DL
# # model = dnn3()
# # Load(model,"-ConvNet-Epoch2",doPrint=True)
# # DLaccuracy = Test(model,dataset)

# # print("\n DL Accuracy:")
# # print(DLaccuracy)

# # #Compute accuacy using ML
# MLaccuracy = runIQTREE(dataset)
# print("-"*100+"\n ML Accuracy:")
# print(MLaccuracy)

# # #Compute and print
# # print("-"*100+"\n ML Accuracy:")
# # print(MLaccuracy)
# # print("\n DL Accuracy:")
# # print(DLaccuracy)
