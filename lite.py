from modelHandler import TrainAndTest
from util.plotter import line
from Recombination.ctrlGenPar import SpeciesTreeInfo
from Recombination.generate import generateAndGet
from ML.IQRAX import runIQTREE, runRAxML, runRAxMLClassification

numTrials = 100
for sL in [100,1000,10000,100000,1000000]:
    for mR in [2.5e-8,2.5e-7,2.5e-6]:
        HCGInfo = SpeciesTreeInfo(name="HCG",mutationRate=mR, indelRate=0, defaultRecombRate=1.5e-8, popSize=10000, taxaCount=4,
                                   postR="-I 4 1 1 1 1 -n 1 1.0 -n 2 1.0 -n 3 1.0 -n 4 1.0 -ej 0.5 1 4 -ej 0.5 2 3 -ej 1.0 4 3")
        for rF in [0.000000000000000000001]:
            datasets = generateAndGet(numDatapoints=numTrials,treeLabel=2,sequenceLength=sL,recombFactor=rF,speciesTreeInfo=HCGInfo)
            accuracy = runIQTREE(datasets['train'])
            line(f"Mutation Rate: {mR}",[sL],[accuracy],window=f'Accuracy v. Mutation Rate| count={numTrials}, recomboFactor={rF}',xlabel="Sequence Length")
            #line(f"Mutation Rate = {mR}",[rF],[accuracy],window=f'Accuracy v. Recombo Factor | length={sequenceLength}, count={numTrials}',xlabel="Recombo Factor")
            print(f"RecomboFactor={rF}\t MutationRate={mR}\t Accuracy={accuracy}")
