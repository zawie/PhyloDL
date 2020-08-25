from modelHandler import TrainAndTest
from dataHandler import GenerateDatasets,GenerateMergedGTRDatasets,GenerateMergedSpecificDatasets
from models import dnn3,dnn3NoRes
from IQRAX import runRAxML,runIQTREE,runRAxMLClassification
from plotter import line
from evomodels import GTR as getRandomGTRValues

#Run Sequence Length vs. Accuracy Test
NUM_EPOCHS = 3
amounts = {"train":1000,"test":250,"dev":50}
sL = 200

#Define results dictionary
results = dict()

#Angiosperm
datasets = GenerateDatasets(amounts,sequenceLength=sL)

#DL Models Train & Testing
# convnet = dnn3()
# results['ConvNet (dnn3)']  = TrainAndTest(convnet,datasets,NUM_EPOCHS,f"ConvNet",doPlot=True)
# print(results)

heat_map = {0:0,1:1,2:2}
dataset = datasets['train']
for i in range(len(dataset)):
    for datapoint in dataset[i]:
        print(datapoint[0])
        (X,y) = datapoint
        heat_map[y.to_list()]+=1

print(heat_map)
print(len(datasets['train']))

resnet = dnn3NoRes()
results['ResNet (dnn3)']  = TrainAndTest(resnet,datasets,NUM_EPOCHS,f"ResNet",doPlot=True)
print(results)
