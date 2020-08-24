from torch.utils.data import Dataset
class SimpleDataset(Dataset):
    def __init__(self,X_data,Y_data):
        """
        Initializes the Dataset.
        This primarily entiles reading the generated sequeences into a python list
        """
        self.X_data = X_data
        self.Y_data = Y_data

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
