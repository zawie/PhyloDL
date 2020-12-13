from modelHandler import TrainAndTest, Test, Load
from Recombination.dataHandler import splitDataset, transformDataset, loadDataset, saveDataset
from util.plotter import line
from models import dnn3
from Recombination.main import generateData
from ML.IQRAX import runIQTREE, runRAxML, runRAxMLClassification
import time

treeCount = 100
pointsPerTree = 10
for sL in [100,1000,10000,100000]:
    generateData(f"Mixed_sL={sL}",amountOfTrees=treeCount, numTrials=pointsPerTree, sequenceLength = sL, anomolyOnly=False,constrainTrees=False)
    generateData(f"Anomaly_sL={sL}",amountOfTrees=treeCount, numTrials=pointsPerTree, sequenceLength = sL, anomolyOnly=True,constrainTrees=False)
    # generateData(f"ConstrainedMixed_sL={sL}",amountOfTrees=treeCount, numTrials=pointsPerTree, sequenceLength = sL, anomolyOnly=False,constrainTrees=True)
    # generateData(f"ConstrainedAnomaly_sL={sL}",amountOfTrees=treeCount, numTrials=pointsPerTree, sequenceLength = sL, anomolyOnly=True,constrainTrees=True)

results = dict()
for sL in [100,1000,10000,100000]:
    try:
        mixed = loadDataset(f"Mixed_sL={sL}")
        anomaly = loadDataset(f"Anomaly_sL={sL}")
        mixedAccuracy = runIQTREE(mixed)
        anomalyAccuracy = runIQTREE(anomaly)
        results[sL] = (mixedAccuracy,anomalyAccuracy)
    except:
        print("Op, something went wrong :(!",str(results))
    finally:
        print("Mid-way:",str(results))
print("Finished",str(results))


# dataset = generateData(f"smol",amountOfTrees=10, numTrials=10, sequenceLength = 100)
# datsets = splitDataset(dataset,[80,5,10])
# print(runIQTREE(datsets["test"]))