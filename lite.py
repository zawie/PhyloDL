from modelHandler import TrainAndTest
#from dataHandler import GenerateDatasets,GenerateMergedGTRDatasets,GenerateMergedSpecificDatasets
from models import dnn3,dnn3NoRes
#from IQRAX import runRAxML,runIQTREE,runRAxMLClassification
#from plotter import line
#from evomodels import GTR as getRandomGTRValues
import datasetClass

#Run Sequence Length vs. Accuracy Test
NUM_EPOCHS = 3

dataPath = "/Users/rhuck/Downloads/DL_Phylogeny/Recombination/dataClassData/recombination_data0.npy"
labelsPath = "/Users/rhuck/Downloads/DL_Phylogeny/Recombination/dataClassData/recombination_labels0.npy"
datasets = datasetClass.getDataSets(dataPath, labelsPath)

#Define results dictionary
results = dict()

#DL Models Train & Testing
# convnet = dnn3()
# results['ConvNet (dnn3)']  = TrainAndTest(convnet,datasets,NUM_EPOCHS,f"ConvNet",doPlot=True)
# print(results)
resnet = dnn3NoRes()
results['ResNet (dnn3)']  = TrainAndTest(resnet,datasets,NUM_EPOCHS,f"Recombination_ResNet",doPlot=True)
print(results)
