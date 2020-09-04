from modelHandler import TrainAndTest
from models import dnn3,dnn3NoRes
from Recombination.dataHandler import getDataSets
from Recombination.generate import generate
from ML.IQRAX import runIQTREE, runRAxML, runRAxMLClassification
#Run Sequence Length vs. Accuracy Test
NUM_EPOCHS = 3

#Generate and get data
generate(200,2)
datasets = getDataSets(0)

#Train and test model
# resnet = dnn3NoRes()
# accuracy = TrainAndTest(resnet,datasets,NUM_EPOCHS,f"Recombination_ResNet",doPlot=True)
# print("MODEL ACCURACY:",accuracy)


print("IQTREE ACCURACY:",runIQTREE(datasets['train']))
