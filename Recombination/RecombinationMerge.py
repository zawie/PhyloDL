import os
import numpy as np

def _getdataPaths(directory):
    """
    Given a directory, returns a list of the paths of all .npy datasets that are
    data files and labels files
    """
    dataPaths = []
    labelsPaths = []

    for datafile in os.scandir(directory):
        if datafile.path.endswith(".npy"):

            if "labels" in datafile.path:
                labelsPaths.append(datafile.path)
            elif "data" in datafile.path:
                dataPaths.append(datafile.path)
            else:
                print("[Error] Unknown File Data")
                return

    return dataPaths, labelsPaths

def _loadDatasets(dataPaths):
    """
    Given a set of dataPaths, returns a list of loaded and appended data
    ~Also deletes datasets
    """
    #load data
    load_data = []
    for dataset in dataPaths:
        data = np.load(dataset, allow_pickle=True)
        load_data.append(data)
        os.remove(dataset) #remove data file

    #concatenate data
    all_data = np.concatenate(load_data)

    return all_data

# def _standardizeDataTypes(all_data):
#     """
#     Given labeled all_data it returns a maximum set of data that has same number
#     of every type of label.
#         ~Let A, B, C,... be sets of data labels. Returns set where every type of data
#          has min(|A|, |B|, |C|, ...) datapoints
#
#     Input:
#     data - numpy array of quartet sequences
#     labels - numpy array of labels
#     """
#
#     #count data
#     datatype_amount = {} #map of evolution model to (int of datapoints, list of datapoints)
#     for i, label in enumerate(labels):
#         if label in datatype_amount:
#             datatype_amount[label][0] += 1
#             datatype_amount[label][1].append(datapoint)
#         else:
#             datatype_amount[label] = [1, [i]]
#
#     #datapoint amount for each type of data
#     dp_val = min([num for num, _ in datatype_amount.values()])
#
#     #below is not corrected yet
#
#     #make new data set
#     new_data = []
#     lost_data = []
#     for _, indexList in datatype_amount.values():
#         new_data.extend(indexList[:dp_val])
#         lost_data.extend(indexList[dp_val:])
#
#     #make new labels set
#     new_labels = []
#     lost_labels = []
#     for _, sequence_list in datatype_amount.values():
#         new_all_data.extend(sequence_list[:dp_val])
#         lost_datapoints.extend(sequence_list[dp_val:])
#
#     #print stats
#     print("Datapoints per Label: ", dp_val)
#     print("New Dataset Size: ", len(new_all_data))
#     print("Lost Datapoints: ", len(lost_datapoints))
#
#     return dp_val, new_all_data, lost_datapoints

def saveData(directory, outputPath):
    """
    Given a directory and an outputPath, saves all the data in the directory to
    the outputPathPrefix + either "data" or "labels" with numpy.save()
    """
    dataPaths, labelsPaths = _getdataPaths(directory)
    print("AHHH",dataPaths)
    data = _loadDatasets(dataPaths)
    labels = _loadDatasets(labelsPaths)

    #find largest path index
    i = -1 #so plus one later will make indices start at 0
    for datafile in os.scandir(outputPath):
        print(datafile)
        if datafile.path.endswith(".npy"):
            j = int(datafile.path[-5])
            print("i:", i, "j:", j)
            if j > i:
                i = j

    #use next path index
    i += 1
    dataOutputPath = outputPath + f"/recombination_data{i}"
    labelsOutputPath = outputPath + f"/recombination_labels{i}"

    np.save(dataOutputPath, data)
    np.save(labelsOutputPath, labels)