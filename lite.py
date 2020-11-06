from modelHandler import TrainAndTest
from util.plotter import line
from Recombination.main import generateData
from ML.IQRAX import runIQTREE, runRAxML, runRAxMLClassification

results = list()
#Generate data
datasets = generateData()
#Compute accuacy using ML
accuracy = runIQTREE(datasets['train'])
results.append(accuracy)

#Compute and print
print("-"*100+"\nAverage:")
print(sum(results)/len(results))
