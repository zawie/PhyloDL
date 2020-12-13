from modelHandler import TrainAndTest
from Recombination.dataHandler import splitDatasets, loadDataset
from util.plotter import line
from models import dnn3
from Recombination.main import generateData
from ML.IQRAX import runIQTREE, runRAxML, runRAxMLClassification
import time

dataset = generateData(name="1k Trees 10k SL",amountOfTrees=1000, numTrials=10,
                        sequenceLength=10000, anomolyOnly=False)
IQAcc = runIQTREE(dataset)
# RaxAcc = runRAxML(dataset)
# RaxClassAcc = runRAxMLClassification(dataset)

print("[IQ ACCURACY] " + "-"*100+"\n IQ Accuracy:\n")
print(IQAcc)
# print("-"*100+"\n RAxML Accuracy:\n")
# print(RaxAcc)
# print("-"*100+"\n RAxML Accuracy:\n")
# print(RaxClassAcc)
