from torch.utils.data import Dataset
import numpy as np

class SimpleDataset(Dataset):
    def __init__(self, dataPath, labelsPath):
        """
        Initializes the Dataset.
        This primarily entiles reading the generated sequeences into a python list
        """
        X_data = np.load(dataPath, allow_pickle=True)
        Y_data = np.load(labelsPath, allow_pickle=True)

        self.X_data = X_data.tolist()
        self.Y_data = Y_data.tolist()

        assert len(self.X_data) == len(self.Y_data)

    def __getitem__(self,index):
        """
        Gets a certain tree across all three trees (alpha,beta,charlie)
        """
        return self.X_data[index],self.Y_data[index]

    def __len__(self):
        """
        Returns the number of entries in this dataset
        """
        return len(self.X_data)

    def __add__(self, other):
        """
        Merges to datasets
        """
        return SimpleDataset(self.X_data+other.X_data,self.Y_data+other.Y_data)

    def saveData(self, pathPrefix):
        """
        Saves the datasets data and labels
        """
        np.save(pathPrefix + "_data", self.X_data)
        np.save(pathPrefix + "_labels", self.Y_data)


if __name__ == "__main__":
    dataPath = "/Users/rhuck/Downloads/DL_Phylogeny/Recombination/preprocessed_data/recombinant_data0.npy"
    labelsPath = "/Users/rhuck/Downloads/DL_Phylogeny/Recombination/preprocessed_data/recombinant_labels0.npy"

    dataset = SimpleDataset(dataPath, labelsPath)

    print(len(dataset))
    dataset.saveData("dataset")
