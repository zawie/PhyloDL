import random
import numpy as np
from torch.utils.data import DataLoader

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


def makeDatFile(dataset, output_file='ML_run_file.dat', label_file='labels.txt'):
    """
    Makes a file for ML_test_auto.py to run ML test on
    """
    #Create Files
    output_f = open(output_file, "w")
    output_f = open(output_file, "a")
    label_f = open(label_file, "w")
    label_f = open(label_file, "a")

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
            label_f.write(str(label) + "\n")




import dataHandler
#dataHandler.GenerateAll(200,20,20)
data = dataHandler.NonpermutedDataset("train")
makeDatFile(data)
