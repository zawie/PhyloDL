import torch
import torch.nn
import dataHandler
import modelHandler
import plotter
from torch.utils.data import DataLoader

#Networks
class Encoder(torch.nn.Module):
    def __init__(self,latent_size=200):
        self.latent_size = latent_size
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(16, 80, 1,groups=4),
            torch.nn.BatchNorm1d(80),
            torch.nn.ReLU(),
            torch.nn.Conv1d(80, 32, 1),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1),
        )
        self.linear = torch.nn.Linear(32, latent_size)

    def forward(self, x):
        x = torch.reshape(x,[x.size()[0],16,-1])
        x = self.conv(x).squeeze(dim=2)
        return self.linear(x)

class Decoder():
    def __init__(self,latent_size=200):
        self.latent_size = latent_size
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(1, 80, 1),
            torch.nn.BatchNorm1d(80),
            torch.nn.ReLU(),
            torch.nn.Conv1d(80, 16, 1),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1),
        )
        self.linear = torch.nn.Linear(32, latent_size)

    def forward(self, x):
        assert(x.size()[1] == self.latent_size)
        x = self.conv(x).squeeze(dim=2)
        return self.linear(x)


#Working funcitons
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
    #Get model
    if doLoad:
        modelHandler.Load(model,f"{name}")
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
        if valset:
            success_rate,average_loss = Test(model,valset,name="Validation",criterion=criterion)
            plotter.line("Validation Accuracy",[epoch],[success_rate],window=name)
            if epoch > 0:
                plotter.line("Validation Loss",[epoch],[average_loss],window=name)
        #Train
        print("Training...")
        for i, (inputs, _) in enumerate(train_loader):
            inputs,_ = trainset.expand(inputs,_)
            # Run the forward pass
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss_list.append(loss.item())
            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #Long epoch save/log step
            if i % 100 == 0 and i > 0:
                print(f'\tEpoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{total_step}]')
                #Plot Training Loss
                plotter.line("Training Loss",[i/total_step+epoch],[sum(loss_list)/len(loss_list)],window=name)
                loss_list = list()
                #Log validation mid long epoch
                if valset:
                    success_rate,average_loss = Test(model,valset,criterion=criterion)
                    print_succes_rate = int(success_rate*10000000)/100000
                    print(f"\tValidation Accuracy: {print_succes_rate}%")
                    plotter.line("Validation Accuracy",[epoch + i/total_step],[success_rate],window=name)
                    plotter.line("Validation Loss",[epoch + i/total_step],[average_loss],window=name)
                #Log training accuracy mide long epoch
                if train_t > 0:
                    plotter.line("Training Accuracy",[epoch+ + i/total_step],[train_s/train_t],window=name)
                    train_s = 0
                    train_t = 0
                #Plot Training loss
                if len(loss_list) > 0:
                    plotter.line("Training Loss",[i/total_step+epoch],[sum(loss_list)/len(loss_list)],window=name)
                    loss_list = list()
                #Mid-Save
                modelHandler.Save(model,f"{name}",doPrint=False)
        #Save model at end of epoch
        modelHandler.Save(model,f"{name}")

#Generate Data
dataHandler.Generate("encoder_train",1000,sequenceLength=200,mean=0.1,std=0,model="HKY",symmetricOnly=False,r_matrix=None)
dataHandler.Generate("encoder_dev",1000,sequenceLength=200,mean=0.1,std=0,model="HKY",symmetricOnly=False,r_matrix=None)
dataHandler.Generate("encoder_test",1000,sequenceLength=200,mean=0.1,std=0,model="HKY",symmetricOnly=False,r_matrix=None)
trainset = dataHandler.NonpermutedDataset("encoder_train")
devset = dataHandler.NonpermutedDataset("encoder_dev")
testset = dataHandler.NonpermutedDataset("encoder_test")
