# seq-gen-dataset
This repo easily converts seq-gen generated data to torch datasets and simplifies the generation process. 

## Setup
This heavily relies on properly installing the original seq-gen program, found here: https://snoweye.github.io/phyclust/document/Seq-Gen.v.1.3.2/Seq-Gen.Manual.html

This can installed via conda:
`conda install seq-gen`

This also requires pytorch to be installed.
## Generation
Simply run `python dataHandler.py generate {sequence length} {amount of training sequence triplets} { amount ... testing ...} {amount ... dev ...}` to generate sequence. Or call the function Generate() within dataHandler.py

NOTE: For any amount requested, triple the number of sequences will be generated (one for each tree).
## Accessing Torch Dataset
To access the dataset simply import the python file, dataHandler, to your project and access a dataset by typing `dataHandler.SequenceDataset(<set>)`, where your set options are `"train"`,`"dev"`, or `"test"`

For shorthand access you can also simply do: `dataHandler.train()`, `dataHandler.test()`, or `dataHandler.dev()` to get quick access to the datasets.

Moreover, both the long hand and shorthand access methods, each has an optional argument `preprocess` which defaults to true. If preprocess is true, access time will be constant, however it will take extra time to build the dataset. If false, building the dataset will take negligble time, but accessing will take linear time.
