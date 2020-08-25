from modelHandler import TrainAndTest
from dataHandler import GenerateDatasets,GenerateMergedGTRDatasets,GenerateMergedSpecificDatasets
from models import dnn3,dnn3NoRes
from IQRAX import runRAxML,runIQTREE,runRAxMLClassification
from plotter import line
from evomodels import GTR as getRandomGTRValues

#Run Sequence Length vs. Accuracy Test
NUM_EPOCHS = 3
amounts = {"train":10**4,"test":2500,"dev":500}
sL = 200

#Define results dictionary
results = dict()

#Angiosperm
datasets = GenerateDatasets(amounts,sequenceLength=sL)

#DL Models Train & Testing
# convnet = dnn3()
# results['ConvNet (dnn3)']  = TrainAndTest(convnet,datasets,NUM_EPOCHS,f"ConvNet",doPlot=True)
# print(results)
print(len(datasets['train']))
resnet = dnn3NoRes()
results['ResNet (dnn3)']  = TrainAndTest(resnet,datasets,NUM_EPOCHS,f"ResNet",doPlot=True)
print(results)
