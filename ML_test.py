#input an output of seq-gen
#return a ML file modeled on every set of quartet trees

import os
import treeCompare

#INPUT_PATH = "/Users/rhuck/documents/CompProjects/Research_NN_Practice/phydl/bin/permute_test/ML_run_file.dat" # path to file to get data from
INPUT_PATH = "/Users/Adam/GitHub/seq-gen-dataset/ML_run_file.dat"
IQTREE_PATH = "/Users/Adam/Desktop/iqtree-1.6.12-MacOSX/bin/iqtree "
ML_PATH = "ml_path" #directory name and write write files to
FINAL_PATH = "ML_output.dat" #file name to final output file

def _read_file_data(input_path):
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

def _run_ML(ml_path, iqtree_path, final_path, file_data):
    """
    Runs ML on file
    """

    #make file paths
    WRITE_FILE = "/removable_file.dat"
    WRITE_FILE_PATH = ml_path + WRITE_FILE
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
        print(iqtree_path + " -s " + WRITE_FILE_PATH + " -m JC")
        os.system(iqtree_path + " -s " + WRITE_FILE_PATH + " -m JC")

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
            newick_tree = str(treeClass) + " " + line

        #put ML data in other file
        final_f.write(newick_tree)

        #remove ML file
        os.remove(WRITE_FILE_PATH + ".treefile")

    #remove unnecessary directory and file
    os.remove(WRITE_FILE_PATH)

    #close final_f file, where output data is stored
    final_f.close()

def run_all_ML_test(input_path, iqtree_path, ml_path, final_path):
    """
    Runs all the other functions and removed unnecessary files

    ~Final ML newick tree data is in FINAL_PATH
    """
    file_data = _read_file_data(input_path)
    os.mkdir(ml_path) #creates a directory
    _run_ML(ml_path, iqtree_path, final_path, file_data)
    os.rmdir(ml_path) #removes directory


#Run program
run_all_ML_test(INPUT_PATH, IQTREE_PATH, ML_PATH, FINAL_PATH)
