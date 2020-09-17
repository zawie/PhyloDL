from modelHandler import TrainAndTest
from util.plotter import line
from Recombination.ctrlGenPar import SpeciesTreeInfo
from Recombination.generate import generateAndGet
from ML.IQRAX import runIQTREE, runRAxML, runRAxMLClassification

sequenceLength = 10000
numDatapoints = 250

for mR in [2.5e-8,2.5e-7,2.5e-6]:
    HCGInfo = SpeciesTreeInfo(name="HCG",mutationRate=mR, indelRate=0, defaultRecombRate=1.5e-8, popSize=10000, taxaCount=4,
                               postR="-I 4 1 1 1 1 -n 1 1.0 -n 2 1.0 -n 3 1.0 -n 4 1.0 -ej 0.5 1 4 -ej 1.0 4 2 -ej 1.5 2 3")
    for rF in [1,2,3,4,5,6,7,8,9,10]:
        datasets = generateAndGet(numDatapoints=numDatapoints,treeLabel=2,sequenceLength=sequenceLength,recombFactor=rF,speciesTreeInfo=HCGInfo)
        accuracy = runIQTREE(datasets['train'])
        line(f"Mutation Rate = {mR}",[rF],[accuracy],window=f'Accuracy v. Recombo Factor | length={sequenceLength}, count={numDatapoints}',xlabel="Recombo Factor")
        print(f"RecomboFactor={rF}\t MutationRate={mR}\t Accuracy={accuracy}")
