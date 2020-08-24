import os
import random
import numpy as np
import shutil

#This script assumes that you have already run main.py to generate data from
#ms and INDELible

def preprocess_data(data_directory, label, data_path="recombinant_data.npy"):
    """
    Takes in ouput path from main.py (ms & INDELible generation) and preprocesses
    it so it can be run by the neural network

    Input:
    data_directory - path to output of main.py
    data_path - path to output of preprocessed data
    """
    #remove executables
    os.remove(data_directory + "/ms")
    os.remove(data_directory + "/indelible")

    #collect data
    data = []
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
                        hot_encode_seq.append(_hot_encode(sequence))

                    #place datapoint into data with labels
                    datapoint = (hot_encode_seq, label)
                    data.append(datapoint)

    #shuffle data
    random.shuffle(data)

    #save data
    np.save(data_path, data)

    #remove data folder
    shutil.rmtree(data_directory)

def _hot_encode(dna_seq):
    """
    Hot encodes a singular DNA sequence as follows:

    A --> [1, 0, 0, 0]
    C --> [0, 1, 0, 0]
    T --> [0, 0, 1, 0]
    G --> [0, 0, 0, 1]
    """

    hot_encoding = {"A":[1, 0, 0, 0],
                    "C":[0, 1, 0, 0],
                    "T": [0, 0, 1, 0],
                    "G": [0, 0, 0, 1]}

    hot_encode_seq = []
    for position in dna_seq:
        hot_encode_seq.append(hot_encoding[position])

    return hot_encode_seq

# def tree_classifier(tree_structure):
#     """
#     Given a list of taxon integers - returns the corresponding tree label
#     Ex: [4, 2, 3, 1] --> Beta tree --> 1
#     *Assumes quartet phylogeny tree
      # **Not necessary/wrong function becuase .dat file no work this way
#     """
#     #tree structure mapping to labels
#     tree_structure_mapping = {frozenset([frozenset([1,2]), frozenset([3,4])]) : 0,
#                               frozenset([frozenset([1,3]), frozenset([2,4])]) : 1,
#                               frozenset([frozenset([1,4]), frozenset([2,3])]) : 2
#                               }
#     #convert input to sets
#     input_set = frozenset([frozenset([tree_structure[0], tree_structure[1]]),
#                            frozenset([tree_structure[2], tree_structure[3]])])
#
#     return tree_structure_mapping[input_set]


if __name__ == "__main__":
    data_directory = "/Users/rhuck/Downloads/DL_Phylo/Recombination/data_generation/recombination_data/HCG_200_1_6"#path to output of main.py
    data_path = "recombinant_data.npy"#path to output of preprocessed data

    preprocess_data(data_directory, data_path)
