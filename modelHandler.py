import torch
import dataHandler
from torch.utils.data import DataLoader
import torch.nn as nn
import sys
import os
import visdom
import numpy as np

#Connect to Visdom
viz = visdom.Visdom(env='main')

def plot(name,X,Y,window='Main',title='Graph',xlabel="Epoch"):
    """
    Plots onto a visdom plot
    """
    viz.line(
        X=np.array(X),
        Y=np.array(Y),
        win=window,
        name=name,
        update='append',
        opts = dict(title=title, xlabel=xlabel, showlegend=True)
    )

def Save(model,name,doPrint=True):
    """
    Saves the model to models/name
    """
    if doPrint:
        print("Saving model...")
    torch.save(model.state_dict(), "saved_models/{}".format(name))
    if doPrint:
        print("Model saved! [{}]".format(name))

def Load(model,name,doPrint=True):
    """
    Loads the model from models/name
    """
    #Need to check if model data exists before loading it
    if doPrint:
        print("Loading model...")
    model.load_state_dict(torch.load(f'saved_models/{name}'))
    model.eval()
    if doPrint:
        print("Model loaded!")

def compare(outputs,labels,doPrint = False):
    """
    Returns the the number of successes and the number of trials.
    Inputs: outputs from the model
            labels of what it should be
    Returns: success (int)
             trials (int)
             success_rate (float, 0-1)
    """
    s = 0
    t = 0
    #Convert output to guessed labels
    _, predicted = torch.max(outputs.data, 1)
    labels = labels.tolist()
    predicted = predicted.tolist()
    for pair in zip(labels,predicted):
        t += 1
        if pair[0] == pair[1]:
            s += 1
    if doPrint:
        accuracy = int(s/t*10000)/100
        print("Results:\t{predicted}\nLabels:\t{labels}\nAccuracy:\t{s}/{t}={accuracy}%")
    return s,t,s/t

def Test(model,dataset,name=None,criterion=None):
    """
    Runs a test on the dataset given a certain model.
    Inputs: model - the model to be tested on
            dataset - the dataset to be tested with
            name (optional) - what to print, if nothing passed, no print
            criterion (optional) - criterion to be used to calculate loss
    Outputs success_rate (float),
            average_loss (defaults to -1 if no criterion passed)

    """
    if name:
        print(f"{name} testing...")
    trials = 0
    successes = 0
    if criterion:
        l_list = list()
    #Run through loader
    for datapoint in dataset:
        #Break up datapoount
        (inputs,labels) = datapoint
        #Add batch axis & expand
        inputs.unsqueeze_(0)
        labels.unsqueeze_(0)
        inputs,labels = dataset.expand(inputs,labels)
        #Run model
        outputs = model(inputs)
        #Check accuracy
        s,t,_ = compare(outputs,labels)
        successes += s
        trials += t
        #Measure loss
        if criterion:
            l_list.append(criterion(outputs, labels).item())
    #Average loss
    if criterion:
        average_loss = sum(l_list)/len(l_list)
    else:
        average_loss = -1
    #Print & output
    if name:
        accuracy = int(successes/(trials)*10000000)/100000
        print(f"\t{name} Results: {successes}/{trials} = {accuracy}%\n\tAverage loss:{average_loss}")
    return successes/trials,average_loss

def Train(model,trainset,valset,num_epochs,name="Model",doLoad=False):
    """
    Trains a given model
    Inputs model - the model to be trained
           trainset - the dataset to be trianed with
           valet - a validation set
           num_epochs - number of times to go over trainset
           name (optional) - what to save/label model as
           doLoad (optional) - whether or not to load the model
    """
    #Create data loaders
    train_loader = DataLoader(dataset=trainset, batch_size=4, shuffle=True)
    test_loader = DataLoader(dataset=testset, batch_size=1, shuffle=True)
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
        success_rate,average_loss = Test(model,valset,name="Validation",criterion=criterion)
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
            s,t,_ = compare(outputs,labels)
            train_s += s
            train_t += t
            #Long epoch save/log step
            if i % 200 == 0:
                #Log validation mid long epoch
                success_rate,average_loss = Test(model,valset,name="Mid-Validation",criterion=criterion)
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
