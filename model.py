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

def plot(name,X,Y,window='Main',length=dataHandler.default_length):
    viz.line(
        X=np.array(X),
        Y=np.array(Y),
        win=window,
        name=name,
        update='append',
        opts = dict(title=f"{window} Data | Sequence Length = {length}", xlabel="Epoch", showlegend=True)
    )

def Save(model,name):
    print("Saving model...")
    torch.save(model.state_dict(), "models/{}".format(name))
    print("Model saved! [{}]".format(name))

def Load(model,name):
    print("Loading model...")
    model.load_state_dict(torch.load(f'models/{name}'))
    model.eval()
    print("Model loaded!")

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

def Test(student,dataset,name,criterion=None):
    print("{} testing...".format(name))
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    trials = 0
    successes = 0
    l_list = list()
    for i in range(len(loader)):
        #Run model
        inputs,labels = next(iter(loader))
        inputs,labels = dataset.expand(inputs,labels)
        outputs = student(inputs)
        #Check accuracy
        s,t = compare(outputs,labels)
        successes += s
        trials += t
        #Measure loss
        if criterion:
            l_list.append(criterion(outputs, labels).item())
    if criterion:
        average_loss = sum(l_list)/len(l_list)
    else:
        average_loss = -1
    print("\t{} Results: {}/{} = {}% | Average loss:{}".format(name,successes,trials,int(successes/(trials)*10000000)/100000,average_loss))
    return successes/trials,average_loss

def Train(model,trainset,valset,num_epochs,name="Model",doLoad=False):
    print("Creating data loaders...")
    train_loader = DataLoader(dataset=trainset, batch_size=4, shuffle=True)
    test_loader = DataLoader(dataset=testset, batch_size=1, shuffle=True)
    print("Data processed!")
    #Get model
    if doLoad:
        Load(model,f"{name}")
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    # Train the model
    total_step = len(train_loader)
    loss_list = []
    train_s = 0
    train_t = 0
    for epoch in range(num_epochs):
        print()
        #Dev test
        success_rate,average_loss = Test(model,valset,"Validation",criterion=criterion)
        plot("Validation Accuracy",[epoch],[success_rate],window=name)
        if epoch > 0:
            plot("Validation Loss",[epoch],[average_loss],window=name)
        #Train
        print("Training...")
        for i, (inputs, labels) in enumerate(train_loader):
            inputs,labels = trainset.expand(inputs,labels)
            # Run the forward pass
            outputs = model(inputs)
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
            #Long epoch save/log step  
            if total_step > 3000 and i % 1000 == 0:
                #Log validation mid long epoch
                success_rate,average_loss = Test(model,valset,"Validation",criterion=criterion)
                plot("Validation Accuracy",[epoch + i/total_step],[success_rate],window=name)
                plot("Validation Loss",[epoch],[average_loss],window=name)
                #Log training accuracy mide long epoch
                plot("Training Accuracy",[epoch+ + i/total_step],[train_s/train_t],window=name)
                train_s = 0
                train_t = 0
                #Save
                Save(model,f"{name}")            
            #Print to terminal
            if i % 100 == 0:
                print('\tEpoch [{}/{}], Batch [{}/{}], Latest Average Loss: {:.4f}'
                        .format(epoch + 1, num_epochs, i + 1, total_step, sum(loss_list)/len(loss_list)))
            #Plot Training loss
            if i % 20 == 0 and (epoch+i) > 0:
                plot("Training Loss",[i/total_step+epoch],[sum(loss_list)/len(loss_list)],window=name)
                loss_list = list()
        #Plot accuracy
        plot("Training Accuracy",[epoch+1],[train_s/train_t],window=name)
        train_s = 0
        train_t = 0
        #Save model
        Save(model,f"{name}")

"""dataHandler.GenerateRandomBranchLengths()
print("Processing datasets...")
testset = dataHandler.UnpermutedDataset("test")
valset = dataHandler.UnpermutedDataset("dev")
trainset = dataHandler.PermutedDataset("train")
print("Datasets Processed")
model = dnn3._Model()
Train(model,trainset,valset,10,name=f"Random Lengths",doLoad=False)
accuracy,_ = Test(model,testset,"Test")"""

for std in [1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]:
    #Changing branch length and generating
    dataHandler.GenerateRandomBranchLengths(std=std,mean=0.5)
    #Setup data
    print("Processing datasets...")
    testset = dataHandler.UnpermutedDataset("test")
    valset = dataHandler.UnpermutedDataset("dev")
    trainset = dataHandler.PermutedDataset("train")
    print("Datasets Processed")

    model = dnn3._Model()
    Train(model,trainset,valset,3,name=f"STD={std}",doLoad=False)
    accuracy,_ = Test(model,testset,"Test")
    viz.line(
        X=np.array([std]),
        Y=np.array([accuracy]),
        win="Final",
        name="Accuracy",
        update='append',
        opts = dict(title="Accuracy vs. Standard Deviaton", xlabel="Branch Length", showlegend=False)
    )
