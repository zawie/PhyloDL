from generate import generateSequences
from ctrlGenPar import SpeciesTreeInfo
from TreeGenerator import generate as generateTrees
from dataHandler import saveDataset

def generateData(name="Unnamed",amountOfTrees=10, numTrials=10, sequenceLength = 1000, rF=10, mR=1.25e-6,anomolyOnly=False,constrainTrees=False):
    data = list()
    i = 0
    for prStr in generateTrees(amountOfTrees,anomolyOnly=anomolyOnly,constrainTrees=constrainTrees,name=name):
        #SpeciesTree info
        structureName = str(i)
        label = 0
        TreeInfo = SpeciesTreeInfo(name = structureName, mutationRate=mR, indelRate=0, defaultRecombRate=1.5e-7, popSize=10000, taxaCount=4,postR = prStr)
        #Generate data
        data.append(generateSequences(numDatapoints=numTrials,treeLabel=label,sequenceLength=sequenceLength,recombFactor=rF,speciesTreeInfo=TreeInfo))
        i += 1
    #Merge all the datasets
    final = data[0]
    for i in range(1,len(data)):
        final = final.__add__(data[i])

    #write meta data
    final.writeToMetadata("amountOfTrees",amountOfTrees)
    final.writeToMetadata("numTrials",numTrials)
    final.writeToMetadata("sequenceLength",sequenceLength)
    final.writeToMetadata("recombFactor",rF)
    final.writeToMetadata("mutationRate",mR)
    final.writeToMetadata("anomalyOnly",anomolyOnly)
    final.writeToMetadata("constrainTrees",constrainTrees)

    #Save the data
    saveDataset(name,final)
    return final
