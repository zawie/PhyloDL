from modelHandler import TrainAndTest
from dataHandler import GenerateDatasets,GenerateMergedGTRDatasets,GenerateMergedSpecificDatasets
from models import dnn3,dnn3NoRes
from IQRAX import runRAxML,runIQTREE,runRAxMLClassification
from plotter import line
from evomodels import GTR as getRandomGTRValues

#Run Sequence Length vs. Accuracy Test
NUM_EPOCHS = 3
amounts = {"train":2500,"test":2500,"dev":500}
sL = 200

#Define results dictionary
results = dict()

#Angiosperm
datasets = GenerateDatasets(amounts,sequenceLength=sL,model="GTR",r_matrix=[1.61,3.82,0.27,1.56,4.99,1],f_matrix=[0.34,0.15,0.18,0.33],pop_size=1)

#DL Models Train & Testing
# convnet = dnn3()
# results['ConvNet (dnn3)']  = TrainAndTest(convnet,datasets,NUM_EPOCHS,f"ConvNet",doPlot=True)
# print(results)
resnet = dnn3NoRes()
results['ResNet (dnn3)']  = TrainAndTest(resnet,datasets,NUM_EPOCHS,f"ResNet",doPlot=True)
print(results)
