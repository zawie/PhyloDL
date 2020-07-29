#Import necessary functions
from modelHandler import TrainAndTest
from dataHandler import GenerateDatasets
from models import dnn3,dnn3NoRes
from IQRAX import runRAxML,runIQTREE,runRAxMLClassification
from plotter import line

#Run Sequence Length vs. Accuracy Test
NUM_EPOCHS = 3
for sL in [20]:
    #Define results dictionary
    results = dict()

    #Generate Data
    datasets = GenerateDatasets({"train":1000,"test":100,"dev":10},sequenceLength=sL)

    #ML Tests
    testset = datasets['test']
    results['IQTREE'] = runIQTREE(testset)
    results['RAxML (Inference)'] = runRAxML(testset)
    results['RAxML (Classification)'] = runRAxMLClassification(testset)

    #DL Models Train & Testing
    results['ResNet (dnn3)']  = TrainAndTest(dnn3(),datasets,NUM_EPOCHS,f"ResNet: sequenceLength={sL}",doPlot=False)
    results['ConvNet (dnn3)']  = TrainAndTest(dnn3NoRes(),datasets,NUM_EPOCHS,f"ConvNet: sequenceLength={sL}",doPlot=False)

    #Print and plot results
    print(f"Accuracies for sequenceLength={sL}")
    for name,accuracy in results.items():
        #Print result
        print(f"\t{name}: {int(accuracy*100*100)/100}%")
        #Plot result
        line(name,[sL],[accuracy],window='Sequence Length vs. Accuracy',xlabel="Sequence Length")

    #Save results to CSV
    # TODO: Actually implement this
