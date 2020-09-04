from modelHandler import TrainAndTest
from models import dnn3,dnn3NoRes
from Recombination.dataHandler import getDataSets
from Recombination.generate import generate

#Run Sequence Length vs. Accuracy Test
NUM_EPOCHS = 3

#Generate and get data
generate(1000,2)
datasets = getDataSets(0)

#Train and test model
resnet = dnn3NoRes()
accuracy = TrainAndTest(resnet,datasets,NUM_EPOCHS,f"Recombination_ResNet",doPlot=True)
print(accuracy)
