from modelHandler import TrainAndTest
from util.plotter import line
from models import dnn3
from Recombination.main import generateData
from ML.IQRAX import runIQTREE, runRAxML, runRAxMLClassification
import time

#Generate data
datasets = generateData()

#Compute accuacy using ML
MLaccuracy = runIQTREE(datasets['test'])

#Compute accuray using DL
DLaccuracy = TrainAndTest(dnn3(),datasets,5,f"ConvNet",doPlot=True)

#Compute and print
print("-"*100+"\n ML Accuracy:")
print(MLaccuracy)
print("\n DL Accuracy:")
print(DLaccuracy)