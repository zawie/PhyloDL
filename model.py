import torch
import dataHandler
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
#os.system("visdom")

def Test(loader,name):
    print("{} testing...".format(name))
    trials = len(loader)
    successes = 0
    for i in range(trials):
        inputs,label = next(iter(loader))
        output = model(inputs).tolist()
        label = label.tolist()
        #Check if max index is same
        if output.index(max(output)) == label[0]:
            successes +=1
    print("{} Test: {}/{} = {}%".format(name,successes,i+1,int(successes/(trials)*10000)/100))
    return successes/trials

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
testset = dataHandler.test(preprocess=False)
print("Processing valset... [2/3]")
valset = dataHandler.dev(preprocess=False)
print("Processing trainset... [3/3]")
trainset = dataHandler.train(preprocess=False)

print("Creating data loaders...")
train_loader = DataLoader(dataset=trainset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=testset, batch_size=1, shuffle=True)
val_loader = DataLoader(dataset=valset, batch_size=1, shuffle=True)
print("Data processed!")

#Get model
print("Loading model...")
model = dnn3._Model()
#model.load_state_dict(torch.load('models/alpha'))
#model.eval()
print("Model loaded!")

# Loss and optimizer
criterion = nn.CrossEntropyLoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Train the model
total_step = len(train_loader)
loss_list = []
num_epochs = 100

success_rate = Test(val_loader,"Validation")

for epoch in range(num_epochs):
    print()
    #Train
    print("Training...")
    for i, (images, labels) in enumerate(train_loader):
        # Run the forward pass
        #print("\t\tIMAGE:",images.shape)
        outputs = model(images)
        #print("\t\tOUTPUT:",outputs.shape)
        #print("\t\tLABELS:",labels.shape)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())
        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            viz.line(
                X=np.array([i+total_step*epoch]),
                Y=np.array(sum(loss_list)/10),
                win="Loss",
                name='Line1',
                update='append',
            )
            loss_list = list()
        if i % 100 == 0:
            print('\tEpoch [{}/{}], Batch [{}/{}], Last Loss: {:.4f}'
                    .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
            #Dev test
    success_rate = Test(val_loader,"Validation")
    viz.line(
            X=np.array([epoch]),
            Y=np.array([success_rate]),
            win="Validation",
            name='Line2',
            update='append',
        )
    #Save model
    Save("beta")

Test(train_loader,"Final")