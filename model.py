import torch
import datahandler as dataHandler
from torch.utils.data import DataLoader
import torch.nn as nn
import sys
import os
import pickle
import visdom
from datetime import datetime
import numpy as np
import dnn3

viz = visdom.Visdom(env='main')

def plot(name,X,Y,window='Main'):
    viz.line(
        X=np.array(X),
        Y=np.array(Y),
        win=window,
        name=name,
        update='append',
        opts = dict(title=f"Model Data | Sequence Length = {dataHandler.default_length}", xlabel="Epoch", showlegend=True)
    )

def compare(outputs,labels,doPrint = False):
    s = 0
    t = 0
    _, predicted = torch.max(outputs.data, 1)
    if doPrint:
        print("Results:\t{}\nLabels:\t{}\n".format(predicted.tolist(),labels.tolist()))
    for pair in zip(labels.tolist(),predicted.tolist()):
        t += 1
        if pair[0] == pair[1]:
            s += 1
    return s,t

def Test(loader,name,criterion):
    print("{} testing...".format(name))
    trials = 0
    successes = 0
    l_list = list()
    for i in range(len(loader)):
        #Run model
        inputs,labels = next(iter(loader))
        inputs,labels = dataHandler.expandBatch(inputs,labels)
        outputs = model(inputs)
        #Check accuracy
        s,t = compare(outputs,labels)
        successes += s
        trials += t
        #Measure loss
        l_list.append(criterion(outputs, labels).item())

    average_loss = sum(l_list)/len(l_list)
    print("\t{} Results: {}/{} = {}% | Average loss:{}".format(name,successes,trials,int(successes/(trials)*10000)/100,average_loss))
    return successes/trials,average_loss

def Save(name):
    print("Saving model...")
    torch.save(model.state_dict(), "models/{}".format(name))
    print("Model saved! [{}]".format(name))

#Define model
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(15936, 1000)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1000, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

#Setup data

print("Processing testset... [1/3]")
testset = dataHandler.test(preprocess=True)
print("Processing valset... [2/3]")
valset = dataHandler.dev(preprocess=True)
print("Processing trainset... [3/3]")
trainset = dataHandler.train(preprocess=True)

print("Creating data loaders...")
train_loader = DataLoader(dataset=trainset, batch_size=4, shuffle=True)
test_loader = DataLoader(dataset=testset, batch_size=1, shuffle=True)
val_loader = DataLoader(dataset=valset, batch_size=1, shuffle=True)
print("Data processed!")

#Get model
print("Loading model...")
model = dnn3._Model()
model.load_state_dict(torch.load('models/alpha'))
model.eval()
print("Model loaded!")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

# Train the model
total_step = len(train_loader)
loss_list = []
train_s = 0
train_t = 0
num_epochs = 100

#Test(val_loader,"Validation")

for epoch in range(num_epochs):
    print()
    #Dev test
    success_rate,average_loss = Test(val_loader,"Validation",criterion)
    plot("Validation Accuracy",[epoch],[success_rate])
    if epoch > 0:
        plot("Validation Loss",[epoch],[average_loss])
    #Train
    print("Training...")
    for i, (data, labels) in enumerate(train_loader):
        data, labels = dataHandler.expandBatch(data,labels)
        # Run the forward pass
        outputs = model(data)
   
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())
        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #Log accuracy
        s,t = compare(outputs,labels)
        train_s += s
        train_t += t
                
        #Log training loss   
        if i % 1000 == 0:
            success_rate,average_loss = Test(val_loader,"Validation",criterion)
            plot("Validation Accuracy",[epoch + i/total_step],[success_rate])
            Save("alpha")      
        if i % 100 == 0:
            print('\tEpoch [{}/{}], Batch [{}/{}], Latest Average Loss: {:.4f}'
                    .format(epoch + 1, num_epochs, i + 1, total_step, sum(loss_list)/len(loss_list)))
        if i % 100 == 0 and (epoch+i) > 0:
            plot("Training Loss",[i/total_step+epoch],[sum(loss_list)/len(loss_list)])
            loss_list = list()

    #Plot accuracy
    plot("Training Accuracy",[epoch+1],[train_s/train_t])
    train_s = 0
    train_t = 0

    #Save model
    Save("alpha")

Test(train_loader,"Final")