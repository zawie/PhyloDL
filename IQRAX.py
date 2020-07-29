#input an output of seq-gen
#return a ML file modeled on every set of quartet trees
import shutil
import os
import re
import treeClassifier
import dataHandler
import numpy as np
import hotEncoder
from torch.utils.data import DataLoader

IQTREE_PATH = "executables/iqtree"
RAXML_PATH = "executables/raxml"
ML_PATH = "WORKING_DIRECTORY" #directory name and write write files to

def run(dataset,cmd,name=None):
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
    if name:
        final_path = f'ML_results/{name}_results.txt'
        final_f = open(final_path, "w")
        final_f = open(final_path, "a")
        final_f.write('(Label,Guess)\t<NewickTree>\n')
    #Create loader
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    #Iterate through dataloader
    for i, (inputs, labels) in enumerate(loader):
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
                    sequence = hotEncoder.decode(sequences[j])
                    taxaChar = ["A","B","C","D"][j]
                    write_f.write(f"{taxaChar}         {sequence}\n")
                write_f.close()

                #Run Test
                try:
                    line = cmd(WRITE_FILE_PATH)
                    treeClass = treeClassifier.getClass(line)
                    if name:
                        final_f.write(f'({label},{treeClass})\t {line}')
                    #log succes/trial
                    trials += 1
                    if label == treeClass:
                        successes += 1
                except:
                    print("Oops! Something went wrong...")
                    #Sometimes ML fails when two sequences are identical
                    #So this is a nice save for now...

    #Calculate accuracy
    accuracy = successes/trials
    if name:
        str_accuracy = str(int(accuracy*100*1000)/1000)+"%"
        final_f.write(f'\nAccuracy = {str_accuracy}')
        final_f.close()
    #remove unnecessary directory and file
    os.remove(WRITE_FILE_PATH)
    shutil.rmtree(ML_PATH)
    #Return success rate
    return accuracy


def runIQTREE(dataset,name=None):
    cmd = lambda path: os.system(f"{IQTREE_PATH} -s {path} -m GTR")
    def ML(WRITE_FILE_PATH):
        #Run os
        os.system(f"{IQTREE_PATH} -s {WRITE_FILE_PATH} -m GTR")
        #Read tree prediction
        ML_data = open(WRITE_FILE_PATH + ".treefile", "r")
        line = ML_data.read()
        #Delete files
        suffixes = ["mldist","log","iqtree","ckp.gz","bionj","treefile"]
        for suffix in suffixes:
            os.remove(f"{WRITE_FILE_PATH}.{suffix}")
        #Return line (string)
        return line
    return run(dataset,ML,name=name)

def runRAxML(dataset,name=None):
    def HC(WRITE_FILE_PATH):
        #Run os
        os.system(f"{RAXML_PATH} -s {WRITE_FILE_PATH} -m GTRCAT -T 2 -n {name} -p 69")
        #Read tree prediction
        HC_data = open(f"RAxML_result.{name}", "r")
        line = HC_data.read()
        #Delete files
        suffixes = ["info","log","parsimonyTree","result","bestTree"]
        for suffix in suffixes:
            os.remove(f"RAxML_{suffix}.{name}")
        #return line
        return line
    return run(dataset,HC,name=name)

def runRAxMLClassification(dataset,name=None):
    def doRegex(txt):
        regex = r" Tree #\d, final logLikelihood\: ([-+]?\d*\.\d+|\d+)"
        matches = re.finditer(regex, txt, re.MULTILINE)
        groups = []
        for matchNum, match in enumerate(matches, start=1):
            for groupNum in range(0, len(match.groups())):
                groupNum = groupNum + 1
                group = match.group(groupNum)
                groups.append(float(group))
        return groups

    def HC(WRITE_FILE_PATH):
        #Run os
        os.system(f"raxml-ng --evaluate --msa {WRITE_FILE_PATH} --tree topologies.newick --model JC --threads 1")
        #Read tree prediction
        matches = []
        txt = open("WORKING_DIRECTORY/removable_file.dat.raxml.log", "r").read()
        matches = doRegex(txt)
        index = matches.index(max(matches))
        line = None
        for position,l in enumerate(open("topologies.newick")):
            if position == index:
                line = l
                break
        #Delete files
        suffixes = ["bestModel","bestTree","log","mlTrees","rba","startTree"]
        for suffix in suffixes:
            os.remove(f"{WRITE_FILE_PATH}.raxml.{suffix}")
        #return line
        return line
    return run(dataset,HC,name=name)
