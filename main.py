from modelHandler import TrainAndTest
from models import dnn3,dnn3NoRes
from ML.IQRAX import runRAxML,runIQTREE,runRAxMLClassification
from Recombination.API import getRecombinationDatasets
from util.plotter import line

#Run Sequence Length vs. Accuracy Test
NUM_EPOCHS = 3

#Define results dictionary
results = dict()

#Get data
datasets = getRecombinationDatasets(0)

#DL Models Train & Testing
# convnet = dnn3()
# results['ConvNet (dnn3)']  = TrainAndTest(convnet,datasets,NUM_EPOCHS,f"ConvNet",doPlot=True)
# print(results)
resnet = dnn3NoRes()
results['ResNet (dnn3)']  = TrainAndTest(resnet,datasets,NUM_EPOCHS,f"ResNet",doPlot=True)
print(results)
