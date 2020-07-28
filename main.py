import modelHandler
import dataHandler
import IQRAX
import models
import plotter

datasets = dataHandler.GenerateDatasets({"train":10000,"test":1000,"dev":100})
model = models.dnn3()
modelHandler.Train(model,datasets['train'],datasets['dev'],3)
accuracy,_ = modelHandler.Test(model,datasets['test'],"Final Test")
print("FINAL ACCURACY:",accuracy)
