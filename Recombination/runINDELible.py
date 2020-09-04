import shutil
import os
import subprocess
import re
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def concatenateAligned(numTrees):
    """
    Concatenate aligned sequences for each segment into a large sequence
    :param numTrees: number of segments
    :return: cleaned alignment dictionary
    """
    # total_cutoff = 0

    concatenated = {}
    for record in SeqIO.parse("t0Seq_TRUE.phy", "phylip-relaxed"):
        concatenated[record.id] = ""

    for i in range(numTrees):
        if i == 0:
            for record in SeqIO.parse("t" + str(i) + "Seq_TRUE.phy", "phylip-relaxed"):
                concatenated[record.id] += str(record.seq)
        else:
            segment = {}
            for record in SeqIO.parse("t" + str(i) + "Seq_TRUE.phy", "phylip-relaxed"):
                segment[record.id] = str(record.seq)
            # segment, num_cutoff = correctInsertion(segment)
            for taxa in segment:
                concatenated[taxa] += segment[taxa]
            # total_cutoff += num_cutoff

    # cleaned = cleanAlignment(concatenated)
    # # Convert dictionary into list of SeqRecord objects for writing out
    # seqRecords = []
    # for key in cleaned:
    #     seq = Seq(cleaned[key])
    #     newRecord = SeqRecord(seq)
    #     newRecord.id = key
    #     newRecord.description = key
    #     seqRecords.append(newRecord)
    #
    # SeqIO.write(seqRecords, "aligned.fasta", "fasta")
    #
    # return cleaned

    f = open("aligned.fasta", "a")
    for key in concatenated:
        f.write(">" + key)
        f.write("\n")
        f.write(concatenated[key])
        f.write("\n")


def main(subFolderName, log, doPrint=False):
    # copy INDELible into sub folder
    shutil.copy("indelible", subFolderName)
    os.chdir(subFolderName)

    with open("control.txt", "r") as ctrlFile:
        firstLine = ctrlFile.readline()
        numTrees = int(firstLine[3:])

    if doPrint:
        print("[LOG] Calling INDELible...")
    subprocess.check_call(["./indelible"])  # Run INDELible
    if doPrint:
        print("[LOG] Processing " + subFolderName + "...")

    # concatenateUnaligned(numTrees)
    alignment = concatenateAligned(numTrees)

    # #  For each trial, get precentage of indels in generated alignment, percentage of constant sites in
    # #  generated alignement, and average gap length in generated alignment.
    # percentIndels, percentConstantSites, avgGapLenth = trialAlignmentStat(alignment)

    # log["percentIndels"] = percentIndels
    # log["percentConstantSites"] = percentConstantSites
    # log["avgGapLength"] = avgGapLenth

    return log
