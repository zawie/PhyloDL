import modelHandler
import dataHandler
import models
import plotter

#Pop Count
modelDict = {"CovNet dnn3":models.dnn3NoRes, "ResNet dnn3":models.dnn3}
for p in [1,2,4,8,16,32,64,128]:
    #Generate Data
    data_amounts = {"train":10000,"dev":100,"test":10000}
    datasets = dataHandler.GenerateDatasets(data_amounts,TreeConstructor=dataHandler.PureKingmanTreeConstructor,pop_size=p)
    for key,modelTemplate in modelDict.items():
        #Create and train model
        model = modelTemplate()
        modelHandler.Train(model,datasets["train"],datasets['dev'],3,name=f"Model = {key} | Pop Size = {p}",doLoad=False)
        #Get accuracy of model
        accuracy,_ = modelHandler.Test(model,datasets["test"],"Test")
        plotter.line(key,[p],[accuracy],window='Accuracy v. Pop Size',xlabel="Pure Kingman Pop Size")










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
