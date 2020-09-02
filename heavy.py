from modelHandler import TrainAndTest
from models import dnn3,dnn3NoRes
from ML.IQRAX import runRAxML,runIQTREE,runRAxMLClassification
import Recombination.API as Recombo
getRecombinationDatasets = Recombo.getRecombinationDatasets
generateData = Recombo.generateData
import util.plotter as plotter

#Run Sequence Length vs. Accuracy Test
NUM_EPOCHS = 3

#Define results dictionary
results = dict()

#Get data
generateData()
datasets = getRecombinationDatasets(0)

for key,dataset in datasets.items():
    print(key,dataset,len(dataset))
#DL Models Train & Testing
# convnet = dnn3()
# results['ConvNet (dnn3)']  = TrainAndTest(convnet,datasets,NUM_EPOCHS,f"ConvNet",doPlot=True)
# print(results)
resnet = dnn3NoRes()
results['ResNet (dnn3)']  = TrainAndTest(resnet,datasets,NUM_EPOCHS,f"ResNet",doPlot=True)
print(results)
