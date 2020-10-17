from modelHandler import TrainAndTest
from util.plotter import line
from Recombination.ctrlGenPar import SpeciesTreeInfo
from Recombination.generate import generateAndGet
from ML.IQRAX import runIQTREE, runRAxML, runRAxMLClassification

import math

def generateStructure(x,y,z):
    a = z
    b = z + y
    c = z + y + x
    return f"-I 4 1 1 1 1 -n 2 1.0 -n 3 1.0 -n 1 1.0 -n 4 1.0 -ej {a} 1 2 -en {a} 2 {b} -ej {b} 2 3 -en {b} 3 {b} -ej {c} 3 4 -en {c} 4 {b}"

structures = {
    #'XinhaoTree': "-I 4 1 1 1 1 -n 2 1.0 -n 3 1.0 -n 1 1.0 -n 4 1.0 -ej {2.5} 1 2 -en 2.5 2 4.0 -ej {4.0} 2 3 -en 4.0 3 4.0 -ej {11.25} 3 4 -en 11.25 4 4.0",
    'O': generateStructure(2.5,1.5,7.25),
    'A': generateStructure(1.0,1.0,7.25),
    'B': generateStructure(.5,.5,7.25),
    'C': generateStructure(.15,.5,7.25),
    'D': generateStructure(.15,.15,7.25),
    'E': generateStructure(.1,.1,7.25)
}
numTrials = 200
rF = 10
mR = 1.25e-6
label = 0
for name,s in structures.items():
    print(name,": ",s)
#for sL in [int(1e3)]:
# for sL in [int(10**2.67),int(1e3),int(10**3.33)]:
#     for structureName, prStr in structures.items():
#         label = 0            #SpeciesTree info
#         TreeInfo = SpeciesTreeInfo(name = structureName, mutationRate=mR, indelRate=0, defaultRecombRate=1.5e-7, popSize=10000, taxaCount=4,postR = prStr)
#         #Generate data
#         datasets = generateAndGet(numDatapoints=numTrials,treeLabel=label,sequenceLength=sL,recombFactor=rF,speciesTreeInfo=TreeInfo)
#         #Compute accuacy using ML
#         accuracy = runIQTREE(datasets['train'])
#         line(structureName,[math.log(sL, 10)],[accuracy],window=f'Accuracy v. Sequence Length | count={numTrials}', xlabel="Sequence Length - logbase 10")
