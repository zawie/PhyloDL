import modelHandler
import dataHandler
import dnn3

dataHandler.GenerateTrees()
print("Processing datasets...")
testset = dataHandler.UnpermutedDataset("test")
valset = dataHandler.UnpermutedDataset("dev")
trainset = dataHandler.PermutedDataset("train")
print("Datasets Processed")
model = dnn3._Model()
modelHandler.Train(model,trainset,valset,10,name=f"Random Lengths",doLoad=False)
accuracy,_ = modelHandler.Test(model,testset,"Test")
