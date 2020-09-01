import os
import random
import numpy as np
import shutil
import util.hotEncoder

#This script assumes that you have already run main.py to generate data from
#ms and INDELible

def preprocess_data(data_directory, label, data_path="recombinant_data.npy"):
    """
    Takes in ouput path from main.py (ms & INDELible generation) and preprocesses
    it so it can be run by the neural network

    Input:
    data_directory - path to output of main.py
    label - data label
    data_path - path to output of preprocessed data
    """
    #remove executables
    os.remove(data_directory + "/ms")
    os.remove(data_directory + "/indelible")

    #collect data
    data = [] # list of the quartet sequences
    labels = [] # list of the labels

    for trial in os.scandir(data_directory): #os.scandir returns some class type
        if not trial.path.endswith(".DS_Store"): #ignore hidden .DS_Store files
            for filename in os.listdir(trial.path): #os.lsitdir returns file names
                if filename == "aligned.fasta":
                    dp_file = open(trial.path + "/" + filename, "r")
                    sequences = [None for _ in range(4)] #assuming a quartet tree
                    tree_structure = []

                    #add leaves (taxons & sequences) to datapoint
                    #*Note: must not include "\n"
                    taxon_index = None
                    for index, line in enumerate(dp_file, 1):
                        if index % 2 == 1:
                            taxon_index = int(line[1]) #odd lines are taxon labels
                            tree_structure.append(taxon_index)
                        else:
                            sequence = line[:-1] #even lines are sequences (remove "\n")
                            assert taxon_index != None
                            assert sequences[taxon_index-1] == None
                            sequences[taxon_index-1] = sequence
                            taxon_index = None

                    #hot encode sequences
                    hot_encode_seq = []
                    for sequence in sequences:
                        hot_encode_seq.append(hotEncoder.encode(sequence))

                    #form data and labels lists
                    data.append(hot_encode_seq)
                    labels.append(label)

    #save data
    np.save(data_path[:-1] + "data" + data_path[-1], data) #quartet tree data save
    np.save(data_path[:-1] + "labels" + data_path[-1], labels) #labels data save

    #remove data folder
    shutil.rmtree(data_directory)


if __name__ == "__main__":
    data_directory = "/Users/rhuck/Downloads/DL_Phylo/Recombination/data_generation/recombination_data/HCG_200_1_6"#path to output of main.py
    data_path = "recombinant_data.npy"#path to output of preprocessed data

    preprocess_data(data_directory, data_path)
