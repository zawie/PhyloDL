import ctrlGenPar as ctrlGen
import runINDELible
import os
import shutil
import multiprocessing as mp
import statistics

import datetime
import recombinationPreprocess
import recombinationMerge


def run(speciesTree, recombFactor, seqLen, trialIndex, output):
    """
	Worker thread. Runs one trial of ms and INDELible.
	"""

    subFolderName, log = ctrlGen.main(speciesTree, seqLen, recombFactor, trialIndex)
    log = runINDELible.main(subFolderName, log)
    output.put(log)


def main(speciesTree, recombFactor, seqLen, numTrial):
    print(
        "=========================================================================================================================")
    print("[LOG] speciesTree = " + speciesTree + "; recombFactor = " + str(recombFactor) + "; seqLen = " + str(
        seqLen) + "; numTrial = " + str(numTrial))

    folderName = speciesTree + "_" + str(seqLen) + "_" + str(recombFactor) + "_" + str(numTrial)
    try:
        os.mkdir(folderName)
    except OSError:
        print("[ERROR] Creation of directory %s failed. Abort process." % folderName)
        return
    else:
        print("[LOG] Successfully created directory: %s" % folderName)

    shutil.copy("ms", folderName)
    shutil.copy("indelible", folderName)
    os.chdir(folderName)

    # Spawn numTrial processes. Each process runs one trial.
    processes = []
    output = mp.Queue()
    for i in range(numTrial):
        processes.append(mp.Process(target=run, args=(speciesTree, recombFactor, seqLen, i, output)))

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    # Collect log results
    allLogs = [output.get() for p in processes]

    os.chdir("..")

    return folderName #required to run recombinationPreprocess

if __name__ == "__main__":

    begin_time = datetime.datetime.now()

    #Change these values:
    num_datapoints = 2604 #make num_datapoints divisible by num_trials
    label = 2
    output_path = "recombination_data/gamma1_fact5_sl10000"

    num_trials = 6 #dont make bigger than number of cores (parallel processing)
    directory = "preprocessed_data" #"recombination_data"#"preprocessed_data"
    dev_percentage = 0.3

    iterations = int(num_datapoints / num_trials)

    for i in range(iterations):
        #generate data
        print("generating")
        data_directory = main(speciesTree="HCG", recombFactor=5, seqLen=10000, numTrial=num_trials)
        #recombFactor=1
        #seqLen: 5000000

        #preprocess data
        print("preprocessing")
        data_path = f"{directory}/recombinant_data{i}.npy"
        recombinationPreprocess.preprocess_data(data_directory, label, data_path)

    #merging data
    recombinationMerge.save_data(directory, output_path)

    # directory = "recombination_data" #"recombination_data"#"preprocessed_data"
    # output_path = "recombination_data/beta2"
    # recombinationMerge.generate_standardized_datasets(directory, dev_percentage, output_path)

    print("Total Execution Time: ", datetime.datetime.now() - begin_time)
