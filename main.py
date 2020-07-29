#Import necessary functions
from modelHandler import TrainAndTest
from dataHandler import GenerateDatasets,GenerateMergedGTRDatasets
from models import dnn3,dnn3NoRes
from IQRAX import runRAxML,runIQTREE,runRAxMLClassification
from plotter import line
from evomodels import GTR as getRandomGTRValues
from csv import writer
import time
TIME_STAMP = time.time()

#Generate Random GTR models:
evoCount = 10
GTR_MODELS = []
GTR_MODEL_TXT = f"results/frequencies{TIME_STAMP}.txt"
for i in range(evoCount):
    _,base_freq,_,rate_mx = getRandomGTRValues()
    print(f"\nRandom GTR Model ({i+1}/{evoCount}):\n\tBase Frequency:{base_freq}\n\tRate Matrix:{rate_mx}\n")
    GTR_MODELS.append((base_freq,rate_mx))
    #Write to a txt
    with open(GTR_MODEL_TXT, 'a') as file_obj:
        file_obj.write(f"\nRandom GTR Model ({i+1}/{evoCount}):\n\tBase Frequency:{base_freq}\n\tRate Matrix:{rate_mx}\n")

#Geneate a CSV file to save accuries:
CSV_FILE_PATH = f"results/accuracies{TIME_STAMP}.csv"
with open(CSV_FILE_PATH, 'w+', newline='') as write_obj:
    csv_writer = writer(write_obj)
    row = ['ResNet (dnn3','ConvNet (dnn3)','RAxML (Classification)','RAxML (Inference)','IQTREE']
    csv_writer.writerow(row)

#Run Sequence Length vs. Accuracy Test
NUM_EPOCHS = 3
for sL in [20,40,80.160,320,640,1280,2560]:
    #Define results dictionary
    results = dict()

    #Generate Data
    amounts = {"train":1000,"test":100}
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

    #Save results to a .csv
    with open(CSV_FILE_PATH, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        r = results
        row = [r['ResNet (dnn3)'],r['ConvNet (dnn3)'],r['RAxML (Classification)'],r['RAxML (Inference)'],r['IQTREE']]
        csv_writer.writerow(row)
