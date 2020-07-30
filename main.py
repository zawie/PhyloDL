#Import necessary functions
from modelHandler import TrainAndTest
from dataHandler import GenerateDatasets,GenerateMergedGTRDatasets,GenerateMergedSpecificDatasets
from models import dnn3,dnn3NoRes
from IQRAX import runRAxML,runIQTREE,runRAxMLClassification
from plotter import line
from evomodels import GTR as getRandomGTRValues
from csv import writer
import time
TIME_STAMP = time.time()

#Generate Random GTR models:
"""evoCount = 1
GTR_MODELS = []
GTR_MODEL_TXT = f"results/frequencies{TIME_STAMP}.txt"
for i in range(evoCount):
    _,base_freq,_,rate_mx = getRandomGTRValues()
    print(f"\nRandom GTR Model ({i+1}/{evoCount}):\n\tBase Frequency:{base_freq}\n\tRate Matrix:{rate_mx}\n")
    GTR_MODELS.append((base_freq,rate_mx))
    #Write to a txt
    with open(GTR_MODEL_TXT, 'a') as file_obj:
        file_obj.write(f"\nRandom GTR Model ({i+1}/{evoCount}):\n\tBase Frequency:{base_freq}\n\tRate Matrix:{rate_mx}\n")
"""

#Define specific models
SPECIFIC_MODELS = {#"Luay":{'m':"GTR",'r':[0.2173,0.9798,0.2575,0.1038,1,0.2070],'f':[0.2112,0.2888,0.2896,0.2104]},
         "Angiosperm":{'m':"GTR",'r':[1.61,3.82,0.27,1.56,4.99,1],'f':[0.34,0.15,0.18,0.33]},
         #"Simple":{'m':"JC"}
        }
#Geneate a CSV file to save accuries:
CSV_FILE_PATH = f"results/accuracies{TIME_STAMP}.csv"
with open(CSV_FILE_PATH, 'w+', newline='') as write_obj:
    csv_writer = writer(write_obj)
    row = ['ResNet (dnn3','ConvNet (dnn3)','RAxML (Classification)','RAxML (Inference)','IQTREE']
    csv_writer.writerow(row)

#Run Sequence Length vs. Accuracy Test
NUM_EPOCHS = 4
amounts = {"train":25000,"test":2500}

SUBTITLE = f"epochs={NUM_EPOCHS},amounts={amounts},simulation=Angiosperm"
for sL in [20,40,80,160,320,640,1280,2560]:
    #Define results dictionary
    results = dict()

    #Generate Data
    datasets = GenerateMergedSpecificDatasets(amounts,SPECIFIC_MODELS,sequenceLength=sL)
    #datasets = GenerateMergedGTRDatasets(amounts,GTR_MODELS,sequenceLength=sL)

    #ML Tests
    testset = datasets['test']
    results['RAxML (Classification)'] = runRAxMLClassification(testset)
    results['RAxML (Inference)'] = runRAxML(testset)
    results['IQTREE'] = runIQTREE(testset)
    print(results)

    #DL Models Train & Testing
    resnet = dnn3NoRes()
    convnet = dnn3()
    results['ResNet (dnn3)']  = TrainAndTest(resnet,datasets,NUM_EPOCHS,f"ResNet: sequenceLength={sL} | {SUBTITLE}",doPlot=True)
    results['ConvNet (dnn3)']  = TrainAndTest(convnet,datasets,NUM_EPOCHS,f"ConvNet: sequenceLength={sL} | {SUBTITLE}",doPlot=True)
    print(results)

    #Print and plot results
    print(f"Accuracies for sequenceLength={sL}")
    for name,accuracy in results.items():
        #Print result
        print(f"\t{name}: {int(accuracy*100*100)/100}%")
        #Plot result
        line(name,[sL],[accuracy],window=f'Sequence Length vs. Accuracy | {SUBTITLE}',xlabel="Sequence Length")

    #Save results to a .csv
    with open(CSV_FILE_PATH, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        r = results
        row = [r['ResNet (dnn3)'],r['ConvNet (dnn3)'],r['RAxML (Classification)'],r['RAxML (Inference)'],r['IQTREE']]
        csv_writer.writerow(row)
