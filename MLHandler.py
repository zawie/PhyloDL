#input an output of seq-gen
#return a ML file modeled on every set of quartet trees

import os
import treeClassifier
import dataHandler
import numpy as np
from torch.utils.data import DataLoader

IQTREE_PATH = "/Users/Adam/Desktop/iqtree-1.6.12-MacOSX/bin/iqtree "
ML_PATH = "ml_path" #directory name and write write files to

def unhotencode(sequence):
    """
        Hot encodes inputted sequnce
        "ATGC" -> [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    """
    code_map = {(1,0,0,0):"A",
                (0,1,0,0):"T",
                (0,0,1,0):"G",
                (0,0,0,1):"C"}
    final = ""
    for char in sequence:
        final += (code_map[tuple(char.tolist())])
    return final

def processDataset(name,dataset):
    """
    Creats a .dat file to feed to ML
    Returns a list of labels
    """
    #file names
    output_file = f'data/ML/{name}_dataset.dat'
    #Create Files
    output_f = open(output_file, "w")
    output_f = open(output_file, "a")
    #Create list of labels
    labels_list = list()
    #Find Lengths
    for i, (data, labels) in enumerate(DataLoader(dataset=dataset, batch_size=1, shuffle=False)):
        size = data.size()
        num_datapoints = size[0]*size[1]
        num_seq = size[2]
        seq_len = size[3]

        data,labels = dataset.expand(data,labels)

        for i in range(num_datapoints):
            #Write buffer string
            output_f.write(str(num_seq) + " " + str(seq_len) + "\n")
            #Get label and sequences
            label = labels[i]
            sequences = data[i]
            #Write sequence lines
            for i, sequence in enumerate(sequences):
                output_f.write(["A","B","C","D"][i] + " " + unhotencode(sequence) + "\n")
            #Add new line
            labels_list.append(label)
    return labels_list

def readFileData(input_path):
    """
    Reads data from file in given path
    """
    file = open(input_path, "r")

    line_state = 0
    file_data = []
    datapoint = []
    for line in file:

        if line_state == 4 and len(datapoint) == 4:
            datapoint.append(line)
            file_data.append(datapoint)
            datapoint = []

        else:
            datapoint.append(line)

        line_state = (line_state + 1) % 5
    # os.remove(path)
    return file_data

def runML(name,dataset):
    """
    Runs ML on file
    """
    #Create .dat file and get labels
    labels = processDataset(name,dataset)
    #define input and final path
    input_path = f'data/ML/{name}_dataset.dat'
    final_path = f'data/ML/{name}_results.txt'
    #extact data from .dat file
    file_data = readFileData(input_path)
    #creates working directory
    os.mkdir(ML_PATH)

    #make file paths
    WRITE_FILE = "/removable_file.dat"
    WRITE_FILE_PATH = ML_PATH + WRITE_FILE
    write_f = open(WRITE_FILE_PATH, "x")

    final_f = open(final_path, "x")
    final_f = open(final_path, "a")

    #iterate through each quartet tree
    for datapoint in file_data:
        write_f = open(WRITE_FILE_PATH, "w") #removes old data in file
        write_f = open(WRITE_FILE_PATH, "a") #append mode

        #put quartet tree data in write_f file
        for line in datapoint:
            write_f.write(line)

        #run ML and delete not wanted files
        write_f.close()
        #print(IQTREE_PATH + " -s " + WRITE_FILE_PATH + " -m JC")
        os.system(IQTREE_PATH + " -s " + WRITE_FILE_PATH + " -m JC")

        os.remove(WRITE_FILE_PATH + ".mldist")
        os.remove(WRITE_FILE_PATH + ".log")
        os.remove(WRITE_FILE_PATH + ".iqtree")
        os.remove(WRITE_FILE_PATH + ".ckp.gz")
        os.remove(WRITE_FILE_PATH + ".bionj")

        #access ML output data
        ML_data = open(WRITE_FILE_PATH + ".treefile", "r")
        newick_tree = []
        for line in ML_data:
            assert(newick_tree == [])
            treeClass = treeCompare.getClass(line)
            newick_tree = f'Label: {label}\t ML: {mlGuess}\t TreeClassifier:{treeClass}\t Tree:{line}'

        #put ML data in other file
        final_f.write(newick_tree)
        #remove ML file
        os.remove(WRITE_FILE_PATH + ".treefile")

    #remove unnecessary directory and file
    os.remove(WRITE_FILE_PATH)
    #close final_f file, where output data is stored
    final_f.close()
    #Remove working folder
    os.rmdir(ML_PATH) #removes directory


#Run program
dataset = dataHandler.NonpermutedDataset("train")
runML("train",dataset)
