from modelHandler import TrainAndTest
from util.plotter import line
from Recombination.ctrlGenPar import SpeciesTreeInfo
from Recombination.generate import generateAndGet
from ML.IQRAX import runIQTREE, runRAxML, runRAxMLClassification
from TreeGenerator import generate as generateTrees
import math
import random


amountOfTrees = 10
sequenceLength = 100
numTrials = 10

rF = 10
mR = 1.25e-6

results = list()
for prStr in generateTrees(amountOfTrees):
    structureName= str(random.random())
    label = 0  #SpeciesTree info
    TreeInfo = SpeciesTreeInfo(name = structureName, mutationRate=mR, indelRate=0, defaultRecombRate=1.5e-7, popSize=10000, taxaCount=4,postR = prStr)
    #Generate data
    datasets = generateAndGet(numDatapoints=numTrials,treeLabel=label,sequenceLength=sequenceLength,recombFactor=rF,speciesTreeInfo=TreeInfo)
    #Compute accuacy using ML
    accuracy = runIQTREE(datasets['train'])
    results.append(accuracy)

#Compute and print
print("-"*100+"\nAverage:")
print(sum(results)/len(results))
