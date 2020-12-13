from modelHandler import TrainAndTest, Test, Load
from Recombination.dataHandler import splitDatasets, loadDataset, saveDataset
from util.plotter import line
from models import dnn3
from Recombination.main import generateData
from ML.IQRAX import runIQTREE, runRAxML, runRAxMLClassification
import time

dataset = generateData("")