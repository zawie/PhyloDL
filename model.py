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

viz = visdom.Visdom(env='main')
#os.system("visdom")

def Flush(txt):
    print(" "+txt+(" "*100), end="\r", flush=True)

def Write(txt):
    print("\n "+txt+(" "*100), end="\r", flush=True)

def Test(loader,name):
    Write("{} testing...".format(name))
    trials = len(loader)
    successes = 0
    for i in range(trials):
        inputs,label = next(iter(loader))
        output = model(inputs).tolist()
        a = label[0][0] - 0.5
        b = output[0][0] - 0.5
        if (a<0 and b<0) or (a>0 and b>0):
            successes +=1
        Flush("{} Test: {}/{} = {}%".format(name,successes,i+1,int(successes/(i+1)*10000)/100))
    return successes/trials

def Save(name):
    #Write("Saving model...")
    torch.save(model.state_dict(), "models/{}".format(name))
    #Flush("Model saved! [{}]".format(name))

#Define model
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(40000, 1000)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1000, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out

#Setup data

Write("Processing testset... [0/3]")
testset = dataHandler.test(preprocess=False)
Flush("Processing valset... [1/3]")
valset = dataHandler.dev(preprocess=False)
Flush("Processing trainset... [2/3]")
trainset = dataHandler.train(preprocess=False)

Flush("Creating data loaders... [3/3]")
train_loader = DataLoader(dataset=trainset, batch_size=16, shuffle=True)
test_loader = DataLoader(dataset=testset, batch_size=1, shuffle=True)
val_loader = DataLoader(dataset=valset, batch_size=1, shuffle=True)
Flush("Data processed!")

#Get model
Write("Loading model...")
model = ConvNet()
#model.load_state_dict(torch.load('models/alpha'))
#model.eval()
Flush("Model loaded!")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

# Train the model
total_step = len(train_loader)
loss_list = []
acc_list = []
num_epochs = 1000

for epoch in range(num_epochs):
    print()
    #Train
    Write("Training...")
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
        viz.line(
            X=np.array([i+total_step*epoch]),
            Y=np.array([loss.item()]),
            win="Loss",
            name='Line1',
            update='append',
        )
        Flush('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}'
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
    Save("alpha")
    #Save loss list
    pickle.dump(loss_list,open("loss_list.pickle","wb"))

Test(train_loader,"Final")
