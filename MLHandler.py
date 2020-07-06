#input an output of seq-gen
#return a ML file modeled on every set of quartet trees

import os
import treeClassifier
import dataHandler
import numpy as np
from torch.utils.data import DataLoader

IQTREE_PATH = "/Users/Adam/Desktop/iqtree-1.6.12-MacOSX/bin/iqtree "
ML_PATH = "ML_WORKING_DIRECTORY" #directory name and write write files to

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

def runML(name,dataset):
    """
    Runs ML on file
    """
    successes = 0
    trials = 0
    #creates working directory
    os.mkdir(ML_PATH)
    #make file paths
    WRITE_FILE = "/removable_file.dat"
    WRITE_FILE_PATH = ML_PATH + WRITE_FILE
    write_f = open(WRITE_FILE_PATH, "x")
    final_path = f'ML_results/{name}_results.txt'
    final_f = open(final_path, "w")
    final_f = open(final_path, "a")
    final_f.write('(Label,Guess)\t<NewickTree>\n')
    #Create loader
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    #Iterate through dataloader
    for i, (inputs, labels) in enumerate(loader):
        #expand data
        inputs, labels = dataset.expand(inputs, labels)
        j = -1
        for (sequences,label) in zip(inputs,labels):
            j += 1
            if j % 3 == 0:
                sequenceLength = sequences.size()[1]
                #Write to temp file
                write_f = open(WRITE_FILE_PATH, "w") #Clear file
                write_f = open(WRITE_FILE_PATH, "a") #Set to append mode
                write_f.write(f" 4 {sequenceLength}\n") #Writeheader
                for j in range(4):
                    sequence = unhotencode(sequences[j])
                    taxaChar = ["A","B","C","D"][j]
                    write_f.write(f"{taxaChar}         {sequence}\n")
                write_f.close()

                #Run ML
                os.system(IQTREE_PATH + " -s " + WRITE_FILE_PATH + " -m JC")

                #Access ML Output
                ML_data = open(WRITE_FILE_PATH + ".treefile", "r")
                line = ML_data.read()
                treeClass = treeClassifier.getClass(line)
                final_f.write(f'({label},{treeClass})\t {line}')

                #log succes/trial
                trials += 1
                if label == treeClass:
                    successes += 1
                #removes files
                suffixes = ["mldist","log","iqtree","ckp.gz","bionj","treefile"]
                for suffix in suffixes:
                    os.remove(f"{WRITE_FILE_PATH}.{suffix}")

    #Calculate accuracy
    accuracy = successes/trials
    str_accuracy = str(int(accuracy*100*1000)/1000)+"%"
    final_f.write(f'\nAccuracy = {str_accuracy}')
    #Close final file
    final_f.close()
    #remove unnecessary directory and file
    os.remove(WRITE_FILE_PATH)
    os.rmdir(ML_PATH)
    #Return success rate
    return accuracy
#Run program
"""
dataset = dataHandler.NonpermutedDataset("dev")
accuracy = runML("dev",dataset)
print(accuracy)
"""
