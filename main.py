import modelHandler
import dataHandler
import dnn3
import plotter

"""
#Accuracy v. Standard Deviation Plot
for std in [0,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.9,.95,1]:
    #Generate Appropriate data
    dataHandler.Generate("train",2500,mean=0.5,std=std)
    dataHandler.Generate("test",1000,mean=0.5,std=std)
    dataHandler.Generate("dev",100,mean=0.5,std=std)
    trainset = dataHandler.NonpermutedDataset("train")
    testset = dataHandler.NonpermutedDataset("test")
    valset = dataHandler.NonpermutedDataset("dev")
    #Train model
    model = dnn3._Model()
    old_accuracy = None
    for i in range(5):
        modelHandler.Train(model,trainset,valset,3,name=f"Standard Deviation = {std}",doLoad=False)
        accuracy,_ = modelHandler.Test(model,testset,"Test")
        if accuracy - old_accuracy < .01:
            break
        old_accuracy = accuracy
    modelHandler.plot("Line1",[std],[accuracy],window='Accuracy v. Standard Deviation',xlabel="Standard Deviation")
"""

#Accuracy v. Sequence Length Plot
"""
for l in [20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]:
    #Generate Appropriate data
    dataHandler.Generate("train",2500,sequenceLength=l)
    dataHandler.Generate("test",1000,sequenceLength=l)
    dataHandler.Generate("dev",100,sequenceLength=l)
    trainset = dataHandler.NonpermutedDataset("train")
    testset = dataHandler.NonpermutedDataset("test")
    valset = dataHandler.NonpermutedDataset("dev")
    #Train model
    model = dnn3._Model()
    old_accuracy = None
    for i in range(5):
        modelHandler.Train(model,trainset,valset,3,name=f"Sequence Length = {l}",doLoad=False)
        accuracy,_ = modelHandler.Test(model,testset,"Test")
        if accuracy - old_accuracy < .01 or accuracy >= .99:
            break
        old_accuracy = accuracy
    modelHandler.plot("Line1",[std],[accuracy],window='Accuracy v. Sequence Length',xlabel="Sequence Length")
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
        model = dnn3._Model()
        modelHandler.Train(model,trainset,None,3,name=f"GTR Data:Traversion={traversion}, Transition={transition}",doLoad=False)
        #Get accuracy of model
        accuracy,_ = modelHandler.Test(model,testset,"Test")
        X[y][x] = accuracy
        print(f"Traversion={traversion}, Transition={transition}, Accuracy={accuracy}")
        plotter.heatmap("GTR Accuracy Heatmap: Traversion (Y-axis) and Transition (X-axis)", X, xlabel=transition_values,ylabel=traversion_values)
"""
