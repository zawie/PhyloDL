from modelHandler import TrainAndTest
from models import dnn3,dnn3NoRes
from Recombination.dataHandler import getDataSets
from Recombination.ctrlGenPar import SpeciesTreeInfo
from Recombination.generate import generate
from ML.IQRAX import runIQTREE, runRAxML, runRAxMLClassification
#Run Sequence Length vs. Accuracy Test
NUM_EPOCHS = 3

#Generate and get data
HCGInfo = SpeciesTreeInfo(name="HCG",mutationRate=2.5e-6, indelRate=0, defaultRecombRate=1.5e-8, popSize=10000, taxaCount=4,
                          postR="-I 4 1 1 1 1 -n 1 1.0 -n 2 1.0 -n 3 1.0 -n 4 1.0 -ej 0.5 1 4 -ej 0.5 2 3 -ej 1.0 4 3")

generate(numDatapoints=10,treeLabel=2,sequenceLength=1000,recombFactor=1,speciesTreeInfo=HCGInfo)
datasets = getDataSets()

#Train and test model
# resnet = dnn3NoRes()
# accuracy = TrainAndTest(resnet,datasets,NUM_EPOCHS,f"Recombination_ResNet",doPlot=True)
# print("MODEL ACCURACY:",accuracy)

print("IQTREE ACCURACY:",runIQTREE(datasets['train']))
