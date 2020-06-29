import modelHandler
import dataHandler
import dnn3

dataHandler.GenerateTrees()

testset = dataHandler.UnpermutedDataset("test")
valset = dataHandler.UnpermutedDataset("dev")

#Unpermuted Train
trainset = dataHandler.UnpermutedDataset("train")
model = dnn3._Model()
modelHandler.Train(model,trainset,valset,24,name="Zawie's Non-permuted ",doLoad=False)
accuracy,_ = modelHandler.Test(model,testset,"Test")
