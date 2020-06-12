# seq-gen-dataset
## Generation
Simply run `python dataHandler.py generate {sequence length} {amount of training sequence triplets} { amount ... testing ...} {amount ... dev ...}` to generate sequence. Or call the function Generate() within dataHandler.py

NOTE: For any amount requested, triple the number of sequences will be generated (one for each tree).
## pytorch Dataset
To access the dataset simply import the python file, dataHandler, to your project and access a dataset by typing `dataHandler.SequenceDataset(<set>)`, where your set options are `"train"`,`"dev"`, or `"test"`

For quick access you can also simply do: `dataHandler.train()`, `dataHandler.test()`, or `dataHandler.dev()` to get quick access to the datasets.
