import modelHandler
import dataHandler
import dnn3

#Generating Testing Data
dataHandler.Generate("test",1000)
dataHandler.Generate("dev",100)
testset = dataHandler.UnpermutedDataset("test")
valset = dataHandler.UnpermutedDataset("dev")

#Binary Search
amount = 50000
while True:
    dataHandler.Generate("train",amount)
    trainset = dataHandler.UnpermutedDataset("train")
    model = dnn3._Model()
    modelHandler.Train(model,trainset,valset,3,name=f"Train Amount = {amount}",doLoad=False)
    accuracy,_ = modelHandler.Test(model,testset,"Test")
    if accuracy > .95:
        amount = amount//2
    else:
        amount = int(amount*1.5)
