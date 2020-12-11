import Recombination.ctrlGenPar as ctrlGen
from Recombination.ctrlGenPar import SpeciesTreeInfo
import Recombination.runINDELible as runINDELible
import os
import shutil
import multiprocessing as mp
import statistics

import datetime
import recombinationPreprocess as recombinationPreprocess
import recombinationMerge as recombinationMerge

def run(speciesTreeInfo, recombFactor, seqLen, trialIndex, output):
    print("RUN INPUT:",speciesTreeInfo, recombFactor, seqLen, trialIndex, output)
    """
	Worker thread. Runs one trial of ms and INDELible.
	"""

    subFolderName, log = ctrlGen.main(speciesTreeInfo, seqLen, recombFactor, trialIndex)
    log = runINDELible.main(subFolderName, log)
    output.put(log)


def main(speciesTreeInfo, recombFactor, seqLen, numTrial,doPrint=False):
    if doPrint:
        print(
            "=========================================================================================================================")
        print("[LOG] speciesTree = " + speciesTreeInfo.name + "; recombFactor = " + str(recombFactor) + "; seqLen = " + str(
            seqLen) + "; numTrial = " + str(numTrial))

    folderName = speciesTreeInfo.name + "_" + str(seqLen) + "_" + str(recombFactor) + "_" + str(numTrial)
    try:
        os.mkdir(folderName)
    except OSError:
        print("[ERROR] Creation of directory %s failed. Abort process." % folderName)
        return
    else:
        if doPrint:
            print("[LOG] Successfully created directory: %s" % folderName)

    shutil.copy("Recombination/ms", folderName)
    shutil.copy("Recombination/indelible", folderName)
    os.chdir(folderName)

    # Spawn numTrial processes. Each process runs one trial.
    processes = []
    output = mp.Queue()
    for i in range(numTrial):
        processes.append(mp.Process(target=run, args=(speciesTreeInfo, recombFactor, seqLen, i, output)))

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    # Collect log results
    allLogs = [output.get() for p in processes]

    os.chdir("..")
    return folderName #required to run recombinationPreprocess

HCGInfo = SpeciesTreeInfo(name="HCG",mutationRate=2.5e-6, indelRate=0, defaultRecombRate=1.5e-8, popSize=10000, taxaCount=4,
                          postR="-I 4 1 1 1 1 -n 1 1.0 -n 2 1.0 -n 3 1.0 -n 4 1.0 -ej 0.5 1 4 -ej 0.5 2 3 -ej 1.0 4 3")

def generateSequences(numDatapoints=1000,treeLabel=2,sequenceLength=1000,recombFactor=1,speciesTreeInfo=HCGInfo):

    begin_time = datetime.datetime.now()
    num_trials = 6 #dont make bigger than number of cores (parallel processing)
    preprocessDirectory = "preprocessedData"

    iterations = int(numDatapoints / num_trials)

    for i in range(iterations):
        #generate data
        data_directory = main(speciesTreeInfo=speciesTreeInfo, recombFactor=recombFactor, seqLen=sequenceLength, numTrial=num_trials)
        #recombFactor=1, seqLen: 5000000

        #preprocess data
        recombinationPreprocess.preprocess_data(data_directory, treeLabel, preprocessDirectory,f"/recombinant_{i}")

    #merge preprocessed data
    dataset = recombinationMerge.getDataset(preprocessDirectory)

    print("Total Execution Time: ", datetime.datetime.now() - begin_time)
    return dataset
