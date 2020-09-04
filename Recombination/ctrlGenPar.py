from typing import NamedTuple
# from more_itertools import split_at
import time
import statistics
import dendropy
import subprocess
import os
import random


class SpeciesTreeInfo(NamedTuple):
    mutationRate: float
    indelRate: float
    defaultRecombRate: float  # in unit of per site per generation
    popSize: float
    taxaCount: int
    postR: str


# HCGInfo = SpeciesTreeInfo(mutationRate=2.5e-8, indelRate=0, defaultRecombRate=1.5e-8, popSize=10000, taxaCount=3,
#                           postR="-I 3 1 1 1 -n 2 1.0 -n 3 1.0 -n 1 1.0 -ej 4.0 1 2 -en 4.0 2 4.0 -ej 5.5 2 3 -en 5.5 3 4.0")
HCGInfo = SpeciesTreeInfo(mutationRate=2.5e-6, indelRate=0, defaultRecombRate=1.5e-8, popSize=10000, taxaCount=4,
                          postR="-I 4 1 1 1 1 -n 1 1.0 -n 2 1.0 -n 3 1.0 -n 4 1.0 -ej 0.5 1 4 -ej 0.5 2 3 -ej 1.0 4 3")

speciesTreeMapping = {"HCG": HCGInfo}


# def generateTreeFromMS(speciesTree):
#     info = speciesTreeMapping[speciesTree]
#     split_line = info.postR.split(" -ej ")[1:]
# 	# print(split_line)
#     numTaxa = info.taxaCount
#     tree_pieces = {}
#     for i in range(1, numTaxa + 1):
#         tree_pieces[str(i)] = str(i)
# 	# print(tree_pieces)
#     tree = ""
#     for i in range(len(split_line)):
#         event = split_line[i]
#         split_event = event.split()
# 		# print(split_event)
#         first = split_event[1]
#         second = split_event[2]
#         new_second = "("
#         new_second += tree_pieces[second] + ":_,"
#         new_second += tree_pieces[first] + ":_)"
# 		# print(new_second)
#         tree_pieces[second] = new_second
# 		# print(tree_pieces)
#         if i == len(split_line) - 1:
#             tree = new_second
#             tree += ";"
#     print(speciesTree, tree)
#     return tree
#
#
# # generateTreeFromMS("avian")
# # generateTreeFromMS("butterfly")
# # generateTreeFromMS("primate")
# # generateTreeFromMS("mosquito")
# # generateTreeFromMS("HCGO")
# # generateTreeFromMS("sim1")
# # generateTreeFromMS("sim2")
# # generateTreeFromMS("sim3")


def generateMSCommand(speciesTree, seqLen, recombFactor, numTrial):
    info = speciesTreeMapping[speciesTree]
    rho = 4 * info.popSize * info.defaultRecombRate * recombFactor * seqLen
    seed1 = str(random.randint(5000, 50000))
    seed2 = str(random.randint(5000, 50000))
    seed3 = str(random.randint(5000, 50000))
    command = "mspms " + str(info.taxaCount) + " " + str(numTrial) + " -T -r" + " " + str(rho) + " " + str(
        seqLen) + " " + info.postR + " -seeds " + seed1 + " " + seed2 + " " + seed3
    return command

# print(generateMSCommand("TrueHCGO", 1000, 1, 1))

def condenseGT(trial):
    """
    :param trial: a list of rows (each row is a string) from one trial of ms
    :return: a list of tuples, each corresponding to a segment. First element is segment length. Second element is gene tree for that segment.
    :return: the percentage of ancestral recombination breakpoints.
    """
    condensed = []
    prev = None
    num_ancestral = 0
    total = len(trial) - 1
    for i in range(len(trial)):
        segment = trial[i]
        curr = (int(segment.split("]")[0][1:]), segment.split("]")[-1])
        if i == 0:
            prev = curr
        else:
            if prev[1] == curr[1]:
                prev = (prev[0] + curr[0], prev[1])
                num_ancestral += 1
            else:
                condensed.append(prev)
                prev = curr
    condensed.append(prev)
    # print("[LOG] Result from condenseGT:", condensed)
    per_ancestral = -1
    if total == 0:
        per_ancestral = 0.0
    else:
        per_ancestral = num_ancestral / total
    # print("[LOG] Percentage of ancestral breakpoints:", per_ancestral)
    return condensed, per_ancestral


def generateINDELibleCtrl(speciesTree, trial, fileDirectory, zeroRecomb, seqLen):
    """
    Generate an INDELible control file for the input trial at the input file directory
    :param speciesTree: Name of species tree
    :param trial: List of rows in each trial of ms. Each row corresponds to a segment.
    :param fileDirectory: Name of control file to generate
    :param zeroRecomb: flag indicating whether there is no recombination
    :param seqLen: total length of sequence
    """
    if (zeroRecomb):
        condensedSegments = [(seqLen, trial[0])]
        percentAncestral = 0.0
    else:
        condensedSegments, percentAncestral = condenseGT(trial=trial)
    # [indelrate] command is relative to substitution rate of 1, and insertion rate = deletion rate = indelrate.
    # Hence the factor of 2.
    # indelRate = (speciesTreeMapping[speciesTree].indelRate / speciesTreeMapping[speciesTree].mutationRate) / 2
    with open(fileDirectory, "a") as file:
        # Number of gene trees. Used later for running INDELible
        file.write("// " + str(len(condensedSegments)) + "\n\n")
        # Actual control begins here
        file.write("[TYPE] NUCLEOTIDE 1	\n\n")
        file.write("[MODEL] JCmodel\n")
        file.write("	[submodel] JC\n\n")
        # file.write("	[indelmodel] POW 1.65 500\n")
        # file.write("	[indelrate] ")
        # file.write(str(indelRate) + "\n\n")

        # Generate a TREE block for each segment
        for i in range(len(condensedSegments)):
            segment = condensedSegments[i]
            tree = dendropy.Tree.get(data=segment[1], schema='newick')
            coalescentTreeLength = tree.length()
            # print("[LOG] coalescent tree length is %f" % coalescentTreeLength)
            generationTreeLength = coalescentTreeLength * 4 * speciesTreeMapping[speciesTree].popSize
            # print("[LOG] generation tree length is %f" % generationTreeLength)
            # yearTreeLength = generationTreeLength * speciesTreeMapping[speciesTree].generationTime
            # print("year tree length is %f" % yearTreeLength)
            treeLength = generationTreeLength * speciesTreeMapping[speciesTree].mutationRate
            # file.write("[TREE] t" + str(i) + "  " + tree.as_string(schema='newick'))
            file.write("[TREE] t" + str(i) + "  " + segment[1] + "\n")
            file.write("[treelength] " + str(treeLength) + "\n\n")

        # Generate a PARTITIONS block for each segment
        file.write("\n")
        for i in range(len(condensedSegments)):
            segment = condensedSegments[i]
            file.write(
                "[PARTITIONS] " + "t" + str(i) + "Seq" + "  " + "[t" + str(i) + " JCmodel " + str(segment[0]) + "]\n")

        # Generate EVOLVE block
        file.write("\n[EVOLVE]\n")
        for i in range(len(condensedSegments)):
            file.write("	t" + str(i) + "Seq 1 t" + str(i) + "Seq\n")

        file.write("\n// The true alignment will be output in a file named t*Seq_TRUE.phy \n"
                   "// The unaligned sequences will be output in a file named t*Seq.fas\n")
        return percentAncestral


# def get_topology_info(treelist, speciesTree):
#     """
#     Reduces trees to just their topologies (branch lengths all become "_")
#     and get info about them
#
#     Input:
#     treelist, a list where each element is a row of ms output
#
#     Output:
#     the number of switches in topology, and the number of topologies
#     """
# 	# oneTree = (len(treelist) == 1)
#     oneTree = (len(treelist) == 1 and treelist[0].find("]") == -1)
# 	# print(oneTree)
#     num_switches = 0
#     prev = ""
#     treelengths = []
#     topology_set = set([])
#     total_seq_len = 0
#     ils_subseq_len = 0
#     species_top = speciesTreeMapping[speciesTree].topology
# 	# segments = [(int(i.split("]")[0][1:]), i.split("]")[-1]) for i in trial]
#     # for tree in treelist:
# 	# print(species_top)
# 	# print("\t")
#     for i in range(len(treelist)):
#         full_tree = treelist[i]
#         tree = ""
#         if oneTree:
#             tree = full_tree
#             subseq_len = 1
#             total_seq_len = 1
#         else:
#             tree = full_tree[full_tree.find("]") + 1:]
#             subseq_len = int(full_tree.split("]")[0][1:])
#             total_seq_len += subseq_len
#         currlen = 0.0
#         # print tree
#         spl = tree.split(":")
#         # print spl
#         reducedtree = "" + spl[0]
#         for ind in range(1, len(spl)):
#             curr = spl[ind]
#             # print curr
#             if curr.find(",") != -1:
#                 currlen += float(curr[:curr.find(",")])
#                 spl[ind] = "_" + curr[curr.find(","):]
#             elif curr.find(")") != -1:
#                 currlen += float(curr[:curr.find(")")])
#                 spl[ind] = "_" + curr[curr.find(")"):]
#             else:
#                 print("what's going on?")
#                 print(curr)
#             reducedtree += ":" + spl[ind]
#         treelengths.append(currlen)
# 		# print(reducedtree)
#         if reducedtree != species_top:
#             ils_subseq_len += subseq_len
#         if i == 0:
#             prev = reducedtree
#             topology_set.add(reducedtree)
#         else:
#             if reducedtree != prev:
#                 num_switches += 1
#                 prev = reducedtree
#                 topology_set.add(reducedtree)
#     avglen = sum(treelengths) / len(treelengths)
#     perILS = ils_subseq_len / total_seq_len
# 	# print(perILS)
#     return num_switches, avglen, len(topology_set), perILS


# commandstr = "./ms 11 20 -T -r 80.0 100000 -I 11 1 1 1 1 1 1 1 1 1 1 1 -ej 0.025 8 7 -ej 0.046875 9 7 -ej 0.0546875 3 2 -ej 0.0903125 5 4 -ej 0.1309375 4 2 -ej 0.1515625 11 10 -ej 0.2721875 6 2 -ej 1.1765625 2 1 -ej 1.7559375 7 1 -ej 3.480625 10 1"
# commandstr = "./ms 11 1 -T -r 0.4 500 -I 11 1 1 1 1 1 1 1 1 1 1 1 -ej 0.025 8 7 -ej 0.046875 9 7 -ej 0.0546875 3 2 -ej 0.0903125 5 4 -ej 0.1309375 4 2 -ej 0.1515625 11 10 -ej 0.2721875 6 2 -ej 1.1765625 2 1 -ej 1.7559375 7 1 -ej 3.480625 10 1"
# commandstr = "./ms 11 20 -T -r 480.0 30000 -I 11 1 1 1 1 1 1 1 1 1 1 1 -ej 0.025 8 7 -ej 0.046875 9 7 -ej 0.0546875 3 2 -ej 0.0903125 5 4 -ej 0.1309375 4 2 -ej 0.1515625 11 10 -ej 0.2721875 6 2 -ej 1.1765625 2 1 -ej 1.7559375 7 1 -ej 3.480625 10 1"
# commandstr = "./ms 4 1 -T -r 40 1000 -I 4 1 1 1 1 -ej 16.75 4 3 -ej 21.75 3 2 -ej 25.0 2 1"
# commandstr = "./ms 4 1 -T -r 32 100 -I 4 1 1 1 1 -ej 0.05 4 3 -ej 0.65 3 2 -ej 1.05 2 1"
# commandstr = "./ms 4 1 -T -I 4 1 1 1 1 -ej 0.05 4 3 -ej 0.65 3 2 -ej 1.05 2 1"
# commandList = commandstr.split()
# rawOutput = str(subprocess.check_output(commandList)).split("\\n")[4:]
# output = list(split_at(rawOutput, lambda x: x == "//"))
# for trial in output:
#	del trial[-1]
# for trial in output:
#	get_topology_info(trial, "butterfly")
#	print("\t")


# def trialRecombStat(trial, seqLen, zeroRecomb, speciesTree):
#     """
#     Calculate the statistics of one trial of ms.
#
#     :param trial: a list of rows (each row is a string) from one trial of ms
#     :param seqLen: total length of sequence
#     :param zeroRecomb: whether recombination factor is 0
#     :return:	Number of recombination breakpoints;
#                 Percentage of recombination breakpoints;
#                 Percentage of topology-changing recombination breakpoints;
#                 Average total branch length of gene trees
#     """
#     numRecombBreakpoints = len(trial) - 1
#     if zeroRecomb or numRecombBreakpoints == 0:
#         numTopologyChange, avgTreeLength, numTopologies, perILS = get_topology_info(trial, speciesTree)
#         return 0, 0.0, 0.0, avgTreeLength, numTopologies, perILS
#
#     percentRecombBreakpoints = numRecombBreakpoints / float(seqLen)
#
#     numTopologyChange, avgTreeLength, numTopologies, perILS = get_topology_info(trial, speciesTree)
# 	# segments = [(int(i.split("]")[0][1:]), i.split("]")[-1]) for i in trial]
#     percentTopologyChange = numTopologyChange / float(numRecombBreakpoints)
#
#     return numRecombBreakpoints, percentRecombBreakpoints, percentTopologyChange, avgTreeLength, numTopologies, perILS


# print(trialRecombStat(["[1](A:1,(B:2,C:3));","[1](A:2,(B:1,C:3));","[1](B:1,(A:2,C:3));","[3](A:2,(B:1,C:3));"], 6))


def main(speciesTree, seqLen, recombFactor, trialIndex, doPrint=True):
    if doPrint:
        print("[LOG] Generating control file for trial " + str(trialIndex) + "...")
    msCommand = generateMSCommand(speciesTree, seqLen, recombFactor, 1)
    if doPrint:
        print("[LOG] Generated MS command is", msCommand)
    commandList = msCommand.split()
    trial = str(subprocess.check_output(commandList)).split("\\n")[4:]
    del trial[-1]

    # Generate an INDELible control file for this trial
    subFolderName = "trial" + str(trialIndex)
    try:
        os.mkdir(subFolderName)
    except OSError:
        print("[ERROR] Creation of directory %s failed. Abort process." % subFolderName)
        return

    if recombFactor == 0:
        # Get trial statistics
        # numRecombBreakpoints, percentRecombBreakpoints, percentTopologyChange, avgTreeLength, numTopologies, perILS = trialRecombStat(trial, seqLen, True, speciesTree)
        fileDirectory = subFolderName + "/" + "control.txt"
        percentAncestralBreakpoints = generateINDELibleCtrl(speciesTree=speciesTree, trial=trial,
                                                            fileDirectory=fileDirectory, zeroRecomb=True, seqLen=seqLen)
    else:
        # Get trial statistics
        # numRecombBreakpoints, percentRecombBreakpoints, percentTopologyChange, avgTreeLength, numTopologies, perILS = trialRecombStat(trial, seqLen, False, speciesTree)
        fileDirectory = subFolderName + "/" + "control.txt"
        percentAncestralBreakpoints = generateINDELibleCtrl(speciesTree=speciesTree, trial=trial,
                                                            fileDirectory=fileDirectory, zeroRecomb=False, seqLen=-1)
        log = {}
    return subFolderName, log


# main(speciesTree="HCG", seqLen=1000000, recombFactor=1, trialIndex=1)
