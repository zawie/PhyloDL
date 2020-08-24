from torch.utils.data import Dataset
class SequenceDataset(Dataset):
    def __init__(self,folderName,doTransform=True):
        """
        Initializes the Dataset.
        This primarily entiles reading the generated sequeences into a python list
        """
        #Define constants
        self.folders = [folderName]
        sequenceSets = getSequenceSets(getDatPath(folderName))
        treeLabels =  getTreeLabels(getTrePath(folderName))
        if doTransform:
            self.X_data = list()
            self.Y_data = list()
            for (sequences,label) in zip(sequenceSets,treeLabels):
                toTensor = lambda n: torch.tensor(n,dtype=torch.long)
                self.X_data.extend(transformSequences(sequences,label.tolist()))
                self.Y_data.extend([toTensor(0),toTensor(1),toTensor(2)])
        else:
            self.X_data = sequenceSets
            self.Y_data = treeLabels

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
        self.folders += other.folders
        self.X_data += other.X_data
        self.Y_data += other.Y_data
        return self
