from modelHandler import TrainAndTest
from util.plotter import line
from Recombination.ctrlGenPar import SpeciesTreeInfo
from Recombination.generate import generateAndGet
from ML.IQRAX import runIQTREE, runRAxML, runRAxMLClassification

import math

numTrials = 100
for sL in [int(1e2),int(1e3),int(1e4),int(1e5),int(1e6)]:
    for mR in [1.25e-8,1.25e-7,1.25e-6]:
        prStr ="-I 4 1 1 1 1 -n 2 1.0 -n 3 1.0 -n 1 1.0 -n 4 1.0 -ej 2.5 1 2 -en 2.5 2 4.0 -ej 4.0 2 3 -en 4.0 3 4.0 -ej 11.25 3 4 -en 11.25 4 4.0"
        XinhaoInfo= SpeciesTreeInfo(name = 'XinhaoTree', mutationRate=mR, indelRate=0, defaultRecombRate=1.5e-7, popSize=10000, taxaCount=4,postR = prStr)
        for rF in [0]:
            datasets = generateAndGet(numDatapoints=numTrials,treeLabel=2,sequenceLength=sL,recombFactor=rF,speciesTreeInfo=XinhaoInfo)
            accuracy = runIQTREE(datasets['train'])
            line(f"Mutation Rate: {mR}",[math.ceil(math.log(sL, 10))],[accuracy],window=f'Accuracy v. Mutation Rate| count={numTrials}, recomboFactor={rF}',xlabel="Sequence Length - logbase 10")
            #line(f"Mutation Rate = {mR}",[rF],[accuracy],window=f'Accuracy v. Recombo Factor | length={sequenceLength}, count={numTrials}',xlabel="Recombo Factor")
            print(f"RecomboFactor={rF}\t MutationRate={mR}\t Accuracy={accuracy}")
