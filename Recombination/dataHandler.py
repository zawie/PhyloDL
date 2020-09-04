from torch.utils.data import Dataset
import numpy as np
import torch

def getDataSets(int):
    """
    1. Reads path files
    2. Forms SimpleDataset class
    3. Returns train, dev, test datasets in dictionary format
        {"train":trainSet, "dev":devSet, "test":testSet}
    """
    dataPath = f"dataClassData/recombination_data{int}.npy"
    labelsPath = f"dataClassData/recombination_labels{int}.npy"

    data = np.load(dataPath)
    labels = np.load(labelsPath)

    X_Data = data.tolist()
    Y_Data = labels.tolist()

    initialDataSet = SimpleDataset(X_Data, Y_Data)

    trainSet, devSet, testSet = initialDataSet.formDatasets()

    datasets = {"train":trainSet, "dev":devSet, "test":testSet}

    return datasets

class SimpleDataset(Dataset):
    def __init__(self, data, labels):
        """
        Initializes the Dataset.
        This primarily entiles reading the generated sequeences into a python list

        Input:
        data - list of quartet sequeences
        labels - list of corresponding labesl
        """
        assert len(data) == len(labels)

        self.X_data = data
        self.Y_data = labels

        assert len(self.X_data) == len(self.Y_data)

    def __getitem__(self, index):
        """
        Gets a certain tree across all three trees (alpha,beta,charlie)
        """
        return self.X_data[index], self.Y_data[index]

    def getData(self):
        """
        Gets class instances data and labels
        """
        return self.X_data, self.Y_data

    def __len__(self):
        """
        Returns the number of entries in this dataset
        """
        assert len(self.X_data) == len(self.Y_data)
        return len(self.X_data)

    def __add__(self, other):
        """
        Merges to datasets
        """
        return SimpleDataset(self.X_data+other.X_data, self.Y_data+other.Y_data)

    def formDatasets(self, setProbabilities = [50, 25, 25]):
        """
        Forms SimpleDataset class datasets with the correct probabilities

        Input:
        setProbabilities - list of probabilities: [trainProb, devProb, testProb]

        Output:
        newSets - [trainSet, devSet, testSet]
        """
        assert sum(setProbabilities) == 100 #is a probability distribution

        numAllDatapoints = len(self.Y_data)
        newSets = []
        indexCounter = 0
        for setProbability in setProbabilities:
            numDatapoints = int(setProbability/100 * numAllDatapoints)
            print(numDatapoints)

            #check for mutation??
            newData = self.X_data[indexCounter:indexCounter+numDatapoints]
            newLabels = self.Y_data[indexCounter:indexCounter+numDatapoints]

            newSet = SimpleDataset(torch.tensor(newData,dtype=torch.float),
                                   torch.tensor(newLabels, dtype=torch.long))
            newSets.append(newSet)

            indexCounter += numDatapoints

        return tuple(newSets)

    def saveData(self, pathPrefix):
        """
        Saves the datasets data and labels
        """
        np.save(pathPrefix + "_data", self.X_data)
        np.save(pathPrefix + "_labels", self.Y_data)


if __name__ == "__main__":
    dataPath = "/Users/rhuck/Downloads/DL_Phylogeny/Recombination/dataClassData/recombination_data0.npy"
    labelsPath = "/Users/rhuck/Downloads/DL_Phylogeny/Recombination/dataClassData/recombination_labels0.npy"


    X_data = np.load(dataPath, allow_pickle=True)
    Y_data = np.load(labelsPath, allow_pickle=True)
    data = X_data.tolist()
    labels = Y_data.tolist()
    dataset = SimpleDataset(data, labels)
    data, labels = dataset.getData()
    print("data: ", data)
    print("labels: ", labels)
    print("length: ", len(dataset))

    print("=======================\n")
    (trainSet, devSet, testSet) = dataset.formDatasets()
    print("training set: ", trainSet.getData())
    print("dev set: ", devSet.getData())
    print("test set: ", testSet.getData())
