from generate import generateAndGet
from ctrlGenPar import SpeciesTreeInfo
from TreeGenerator import generate as generateTrees

def generateData(amountOfTrees=1, sequenceLength = 1000, numTrials = 20, rF=10, mR=1.25e-6):
    data = list()
    i = 0
    for prStr in generateTrees(amountOfTrees):
        #SpeciesTree info
        structureName = str(i)
        label = 0
        TreeInfo = SpeciesTreeInfo(name = structureName, mutationRate=mR, indelRate=0, defaultRecombRate=1.5e-7, popSize=10000, taxaCount=4,postR = prStr)
        #Generate data
        data.append(generateAndGet(numDatapoints=numTrials,treeLabel=label,sequenceLength=sequenceLength,recombFactor=rF,speciesTreeInfo=TreeInfo))
        i += 1
    #Merge all like keys
    final = dict()
    for datasets in data:
        for key,dataset in datasets.items():
            if key in final:
                final[key] += dataset
            else:
                final[key] = dataset
    print(final)
    return final
