#Import necessary functions
from modelHandler import TrainAndTest
from dataHandler import GenerateDatasets,GenerateMergedGTRDatasets
from models import dnn3,dnn3NoRes
from IQRAX import runRAxML,runIQTREE,runRAxMLClassification
from plotter import line
from evomodels import GTR as getRandomGTRValues

#Generate Random GTR models:
evoCount = 5
GTR_MODELS = []
for i in range(evoCount):
    _,base_freq,_,rate_mx = getRandomGTRValues()
    print(f"\nRandom GTR Model ({i+1}/{evoCount}):\n\tBase Frequency:{base_freq}\n\tRate Matrix:{rate_mx}\n")
    GTR_MODELS.append((base_freq,rate_mx))
    # TODO: Write models to a .txt

#Run Sequence Length vs. Accuracy Test
NUM_EPOCHS = 3
for sL in [20,40,80.160,320,640,1280,2560]:
    #Define results dictionary
    results = dict()

    #Generate Data
    amounts = {"train":10,"test":10,"dev":1}
    datasets = GenerateMergedGTRDatasets(amounts,GTR_MODELS,sequenceLength=sL)

    #ML Tests
    testset = datasets['test']
    results['RAxML (Classification)'] = runRAxMLClassification(testset)
    results['RAxML (Inference)'] = runRAxML(testset)
    results['IQTREE'] = runIQTREE(testset)

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

    # TODO: Save results to a .csv
