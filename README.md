# seq-gen-dataset
## Generation
Simply run `python dataHandler.py generate`. To change generation parameters you gotta tweak code (default arguments in Generate funciton)

## pytorch Dataset
To access the dataset simply import the python file, dataHandler, to your project and access a dataset by typing `dataHandler.SequenceDataset(<set>)`, where your set options are "train","dev", or "test"
