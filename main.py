#Import necessary functions
from modelHandler import TrainAndTest
from dataHandler import GenerateDatasets
from models import dnn3,dnn3NoRes
from IQRAX import runRAxML,runIQTREE
from plotter import line

#Run Sequence Length vs. Accuracy Test
NUM_EPOCHS = 3
for sL in [20]:
    #Generate Data
    datasets = GenerateDatasets({"train":1000,"test":100,"dev":10},sequenceLength=sL)

    #ML Tests
    testset = datasets['test']
    dataset_name = "Simple"
    IQTREE_accuracy = runIQTREE(dataset_name,testset)
    RAxML_accuracy = runRAxML(dataset_name,testset)

    #DL Models Train & Testing
    res_accuracy = TrainAndTest(dnn3(),datasets,NUM_EPOCHS,f"ResNet: sequenceLength={sL}",doPlot=False)
    conv_accuracy = TrainAndTest(dnn3NoRes(),datasets,NUM_EPOCHS,f"ConvNet: sequenceLength={sL}",doPlot=False)

    #Print results
    print(f"Accuracies for sequenceLength={sL}\n\tRes:{res_accuracy}\n\tConv:{conv_accuracy}\n\n\tIQTREE:{IQTREE_accuracy}\nRAxML:{RAxML_accuracy}")
    #Plot results
    # TODO: Actually plot this

    #Save results to CSV
    # TODO: Actually implement this
