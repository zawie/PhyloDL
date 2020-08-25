import modelHandler
import dataHandler
import models
import plotter
import time

#Settings
data_amounts = {"train":3000*3,"test":12000/4}
datas = {"Luay":{'m':"GTR",'r':[0.2173,0.9798,0.2575,0.1038,1,0.2070],'f':[0.2112,0.2888,0.2896,0.2104]},
         "Angiosperm":{'m':"GTR",'r':[1.61,3.82,0.27,1.56,4.99,1],'f':[0.34,0.15,0.18,0.33]},
         "Simple":{'m':"JC"}
         }

#Create and merge all data
for sequenceLength in [100]:
    mergedData = {}
    for name,settings in datas.items():
        print(name)
        #Generate data
        datasets = None
        m = settings['m']
        if m == 'JC':
            datasets = dataHandler.GenerateDatasets(data_amounts,sequenceLength=sequenceLength)
        else:
            datasets = dataHandler.GenerateDatasets(data_amounts,sequenceLength=sequenceLength,model=m,r_matrix=settings['r'],f_matrix=settings['f'])
        for key, dataset in datasets.items():
            if key in mergedData:
                mergedData[key] += dataset
            else:
                mergedData[key] = dataset
    #Train covnet
    convnet = models.dnn3NoRes()
    modelHandler.Train(convnet,mergedData["train"],None,3,name=f"Convnet length={sequenceLength} NEW",doLoad=False)
    conv_accuracy,_ = modelHandler.Test(convnet,mergedData["test"],"Test")
    #Train resnet
    resnet = models.dnn3()
    modelHandler.Train(resnet,mergedData["train"],None,3,name=f"Resnet length={sequenceLength} NEW",doLoad=False)
    res_accuracy,_ = modelHandler.Test(resnet,mergedData["test"],"Test")
    print(f"\nAccurcies\n\tML:{ML_accuracy}\n\tHC{HC_accuracy}\n\n\tConv:{conv_accuracy}\n\tRes:{res_accuracy}\n")
    plotter.line("Convnet (dnn3) NEW",[sequenceLength],[conv_accuracy],window='Accuracy v. Sequence Length',xlabel="Sequence Length")
    plotter.line("Resnet (dnn3) NEW",[sequenceLength],[res_accuracy],window='Accuracy v. Sequence Length',xlabel="Sequence Length")
