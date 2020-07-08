import modelHandler
import dataHandler
import MLHandler
import models
import plotter

#ML Accuracies
datas = {"Luay":{'m':"GTR",'r':[0.2173,0.9798,0.2575,0.1038,1,0.2070],'f':[0.2112,0.2888,0.2896,0.2104]},
         "Angiosperm":{'m':"GTR",'r':[1.61,3.82,0.27,1.56,4.99,1],'f':[0.34,0.15,0.18,0.33]},
         "Simple":{'m':"JC"}
         }
results = {}
#{'Luay': 0.799, 'Angiosperm': 0.8153333333333334, 'Simple': 0.7496666666666667}
#{'Luay': 0.778, 'Angiosperm': 0.8, 'Simple': 0.7336666666666667}

#Sequence Length 1000
#{'Luay': {'ML': 0.862, 'Model': 0.9713333333333334}, 'Angiosperm': {'ML': 0.926, 'Model': 0.974}, 'Simple': {'ML': 0.797, 'Model': 0.9773333333333334}}

#Sequence Length 200pyt
#{'Luay': {'ML': 0.798, 'Model': 0.9523333333333334}, 'Angiosperm': {'ML': 0.813, 'Model': 0.9406666666666667}, 'Simple': {'ML': 0.741, 'Model': 0.9536666666666667}}

#Sequence Length 500
#{'Luay': {'ML': 0.828, 'Model': 0.9566666666666667}, 'Angiosperm': {'ML': 0.872, 'Model': 0.9576666666666667}, 'Simple': {'ML': 0.776, 'Model': 0.9613333333333334}}

#Sl = 1000; bl min 0.1 JC
#{'Luay': {'ML': 0.884, 'Model': 0}, 'Angiosperm': {'ML': 0.945, 'Model': 0}, 'Simple': {'ML': 0.793, 'Model': 0}}

#Sl = 1000; bl min 0.1 GTR
#{'Luay': {'ML': 0.886, 'Model': 0}, 'Angiosperm': {'ML': 0.909, 'Model': 0}, 'Simple': {'ML': 0.788, 'Model': 0}}
sequenceLength = 200
for name,settings in datas.items():
    #Generate data
    datasets = None
    data_amounts = {"train":0,"dev":0,"test":200}
    m = settings['m']
    if m == 'JC':
        datasets = dataHandler.GenerateDatasets(data_amounts,sequenceLength=sequenceLength)
    else:
        datasets = dataHandler.GenerateDatasets(data_amounts,sequenceLength=sequenceLength,model=m,r_matrix=settings['r'],f_matrix=settings['f'])
    print(datasets)
    #Train model
    model_accuracy = 0
    """model = models.dnn3NoRes()
    modelHandler.Train(model,datasets["train"],datasets['dev'],5,name=f"{name}",doLoad=False)
    model_accuracy,_ = modelHandler.Test(model,datasets["test"],"Test")"""
    #Get ML Accuracy
    ML_accuracy = MLHandler.runML(name,datasets['test'])
    results[name] = {"ML":ML_accuracy,"Model":model_accuracy}
    print(results)
print(results)


#Pop Count Test
"""
pop_sizes = [x/4 for x in list(range(1,100))]
for p in pop_sizes:
    #Generate Data
    data_amounts = {"train":5000,"dev":100,"test":10000}
    datasets = dataHandler.GenerateDatasets(data_amounts,TreeConstructor=dataHandler.PureKingmanTreeConstructor,pop_size=p)
    #Create and train model
    model = models.dnn3NoRes()
    modelHandler.Train(model,datasets["train"],datasets['dev'],5,name=f"Pop Size = {p}",doLoad=False)
    #Get accuracy of model
    accuracy,_ = modelHandler.Test(model,datasets["test"],"Test")
    plotter.line("NoRes dnn3",[p],[accuracy],window='Accuracy v. Pop Size',xlabel="Pure Kingman Pop Size")
"""

#Accuracy v. Standard Deviation Plot
"""
#GTR Test traversion v, transition heatmap
#Define values to test
std_values = list(range(21))
std_values = [x/10 for x in std_values]
mean_values = list(range(21))
mean_values = [x/10 for x in mean_values]

#Create blank heat map
X = []
for y in range(len(std_values)):
    X.append(list())
    for x in range(len(mean_values)):
        X[y].append(0)
plotter.heatmap("Accuracy Heatmap: Standard Deviation (Y-axis) and Mean (X-axis)", X, xlabel=mean_values,ylabel=std_values)

#Run a bunch of models to fill heat map
for y in range(len(std_values)):
    for x in range(len(mean_values)):
        std = std_values[y]
        mean = mean_values[x]
        #Generate Data
        dataHandler.GenerateAll(2500,0,5000,model="HKY",std=std,mean=mean)
        trainset = dataHandler.NonpermutedDataset("train")
        testset = dataHandler.NonpermutedDataset("test")
        #Create and train a model
        model = models.dnn3NoRes()
        modelHandler.Train(model,trainset,None,3,name=f"STD={std} | Mean = {mean}",doLoad=False)
        #Get accuracy of model
        accuracy,_ = modelHandler.Test(model,testset,"Test")
        X[y][x] = accuracy
        print(f"Mean={mean}, STD={std}, Accuracy={accuracy}")
        plotter.heatmap("GTR Accuracy Heatmap: Traversion (Y-axis) and Transition (X-axis)", X, xlabel=mean_values,ylabel=std_values)
"""

#GTR Test traversion v, transition heatmap
"""
#Define values to test
traversion_values = [0.5,1,1.5,2,2.5,3,3.5,4]
transition_values = [0.5,1,1.5,2,2.5,3,3.5,4]
#Create blank heat map
X = []
for y in range(len(traversion_values)):
    X.append(list())
    for x in range(len(transition_values)):
        X[y].append(0)
plotter.heatmap("GTR Accuracy Heatmap: Traversion (Y-axis) and Transition (X-axis)", X, xlabel=transition_values,ylabel=traversion_values)

#Run a bunch of models to fill heat map
for y in range(len(traversion_values)):
    for x in range(len(transition_values)):
        traversion = traversion_values[y]
        transition = transition_values[x]
        #Generate Data
        dataHandler.GenerateAll(2500,0,1000,model="GTR",r_matrix=[traversion,transition]*3,mean=0.5)
        trainset = dataHandler.NonpermutedDataset("train")
        testset = dataHandler.NonpermutedDataset("test")
        #Create and train a model
        model = models.dnn3NoRes()
        modelHandler.Train(model,trainset,None,3,name=f"GTR Data:Traversion={traversion}, Transition={transition}",doLoad=False)
        #Get accuracy of model
        accuracy,_ = modelHandler.Test(model,testset,"Test")
        X[y][x] = accuracy
        print(f"Traversion={traversion}, Transition={transition}, Accuracy={accuracy}")
        plotter.heatmap("GTR Accuracy Heatmap: Traversion (Y-axis) and Transition (X-axis)", X, xlabel=transition_values,ylabel=traversion_values)
"""
