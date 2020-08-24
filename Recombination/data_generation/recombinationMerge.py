import os
import numpy as np

def _get_datapaths(directory):
    """
    Given a directory, returns a list of the paths of all .npy datasets
    """
    datapaths = []

    for datafile in os.scandir(directory):
        if datafile.path.endswith(".npy"):
            datapaths.append(datafile.path)

    return datapaths

def _load_datasets(datapaths):
    """
    Given a set of datapaths, returns a list of loaded and shuffled data
    ~Also deletes datasets
    """
    #load data
    load_data = []
    for dataset in datapaths:
        data = np.load(dataset, allow_pickle=True)
        load_data.append(data)
        os.remove(dataset) #remove data file

    #concatenate data
    all_data = np.concatenate(load_data)

    #shuffle data
    np.random.shuffle(all_data)

    return all_data

def _standardize_data_types(all_data):
    """
    Given labeled all_data it returns a maximum set of data that has same number
    of every type of label.
        ~Let A, B, C,... be sets of data. Returns set where every type of data
         has min(|A|, |B|, |C|, ...) datapoints

    Input:
    all_data - numpy array of labeled datapoints
    """

    #count data
    datatype_amount = {} #map of evolution model to (int of datapoints, list of datapoints)
    for sequences, label in all_data:
        datapoint = (sequences, label)
        if label in datatype_amount:
            datatype_amount[label][0] += 1
            datatype_amount[label][1].append(datapoint)
        else:
            datatype_amount[label] = [1, [datapoint]]

    #datapoint amount for each type of data
    dp_val = min([num for num, _ in datatype_amount.values()])

    #make set from datapoints
    new_all_data = []
    lost_datapoints = []
    for _, sequence_list in datatype_amount.values():
        new_all_data.extend(sequence_list[:dp_val])
        lost_datapoints.extend(sequence_list[dp_val:])

    #shuffle new_all_data
    np.random.shuffle(new_all_data)

    #print stats
    print("Datapoints per Label: ", dp_val)
    print("New Dataset Size: ", len(new_all_data))
    print("Lost Datapoints: ", len(lost_datapoints))

    return dp_val, new_all_data, lost_datapoints


def _new_sets(all_data, dev_percentage, output_path, lost_datapoints = None):
    """
    Takes datasets and then generates and dev and train set based on the given
    dev set percentage by merging the data in the given datasets and then
    dividing it up accordingly
    ~ calls standardize_data_types so that all types of data (labels) have the
      same amounts

    Input:
    datasets - list of strings of dataset paths
    dev_percentage - decimal of how much data should be in the dev set
    output_path - string to put final sets ("train" or "dev" will be appended
                  to end of string)
    """

    #divide all_data
    divide_index = int(len(all_data) * dev_percentage)
    new_dev = all_data[:divide_index]
    new_train = all_data[divide_index:]

    #print stats
    print("Dev size: ", len(new_dev))
    print("Train size: ", len(new_train))
    print("Total size: ", len(new_dev) + len(new_train))

    #save new datasets
    np.save(output_path + "_dev", new_dev)
    np.save(output_path + "_train", new_train)

    if lost_datapoints != None:
        np.save(output_path + "_lost", lost_datapoints)

    return new_dev, new_train

def save_data(directory, output_path):
    """
    Given a directory and an output_path, saves all the data in the directory to
    the output path with numpy.save()
    """
    datapaths = _get_datapaths(directory)
    all_data = _load_datasets(datapaths)
    np.save(output_path, all_data)

def generate_datasets(directory, dev_percentage, output_path):
    """
    Given a directory, this calls functions to read in the data in the directory
    and form train/ dev sets based on dev_percentage and place them in output_path
    """
    datapaths = _get_datapaths(directory)
    all_data = _load_datasets(datapaths)
    _new_sets(all_data, dev_percentage, output_path)

def generate_standardized_datasets(directory, dev_percentage, output_path):
    """
    Given a directory, this calls functions to read in the data in the directory
    and form "standardized" (same number of all labels) train/ dev sets based on
    dev_percentage and place them in output_path
    """
    datapaths = _get_datapaths(directory)
    all_data = _load_datasets(datapaths)
    dp_val, new_all_data, lost_datapoints = _standardize_data_types(all_data)
    _new_sets(new_all_data, dev_percentage, output_path, lost_datapoints)
    return dp_val, new_all_data, lost_datapoints


if __name__ == "__main__":
    directory = "recombination_data" #"recombination_data"#"preprocessed_data"
    dev_percentage = 0.3
    output_path = "recombination_data/test1_fact5_sl10000"

    #save_data(directory, output_path)
    #generate_datasets(directory, dev_percentage, output_path)
    generate_standardized_datasets(directory, dev_percentage, output_path)
