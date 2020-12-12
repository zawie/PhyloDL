from modelHandler import TrainAndTest
from Recombination.dataHandler import splitDatasets, loadDataset
from util.plotter import line
from models import dnn3
from Recombination.main import generateData
from ML.IQRAX import runIQTREE, runRAxML, runRAxMLClassification
import time

dataset = generateData(name="Ryan",amountOfTrees=10, numTrials=10, anomolyOnly=False)
