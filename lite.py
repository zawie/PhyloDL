from modelHandler import TrainAndTest
from util.plotter import line
from Recombination.ctrlGenPar import SpeciesTreeInfo
from Recombination.generate import generateAndGet
from ML.IQRAX import runIQTREE, runRAxML, runRAxMLClassification

import math

numTrials = 200
rF = 10
mR = 1.25e-6
label = 0
for name,s in structures.items():
    print()
    
for sL in [int(1e3)]:
for sL in [int(10**2.67),int(1e3),int(10**3.33)]:
    for structureName, prStr in structures.items():
        label = 0            #SpeciesTree info
        TreeInfo = SpeciesTreeInfo(name = structureName, mutationRate=mR, indelRate=0, defaultRecombRate=1.5e-7, popSize=10000, taxaCount=4,postR = prStr)
        #Generate data
        datasets = generateAndGet(numDatapoints=numTrials,treeLabel=label,sequenceLength=sL,recombFactor=rF,speciesTreeInfo=TreeInfo)
        #Compute accuacy using ML
        accuracy = runIQTREE(datasets['train'])
        line(structureName,[math.log(sL, 10)],[accuracy],window=f'Accuracy v. Sequence Length | count={numTrials}', xlabel="Sequence Length - logbase 10")
