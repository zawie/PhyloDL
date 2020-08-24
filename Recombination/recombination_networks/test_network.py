#!/usr/bin/env python3

"""Quartet tree classification

* Model: Convolutional neural network with basic residual connections
  and batch normalization.
* Training data:
    * 100000 pre-simulated trees using training1.
    * Each epoch uses randomly sampled 2000 trees.
    * The batch size is 16.
* Validation data: 2000 pre-simulated trees using training1.
* Optimizer: Adam with an initial learning rate of 0.001.
"""
print("imports")
import visdom
import pathlib
import pickle
import random

import numpy as np
import torch.autograd
import torch.nn
import torch.optim
import torch.utils.data

graph_title = "Full Test Recombination Model recomb_test2"
graph_win = "recomb_test2.8" #"recomb2"
data_test = "recomb_test2"
model_number = "8"

n = 4 #number of nucleotides (amount needed for hot encoding)
s = 4 #number of sequences
c = None #sequence length

class _Model(torch.nn.Module):
    """A neural network model to predict phylogenetic trees."""

    def __init__(self):
        """Create a neural network model."""
        print("making model")
        super().__init__()

        k = 32
        l = 16

        # self.conv = torch.nn.Sequential(
        #     torch.nn.Conv2d(16, 80, 1, groups=4),
        #     torch.nn.BatchNorm1d(80),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv1d(80, 32, 1),
        #     torch.nn.BatchNorm1d(32),
        #     torch.nn.ReLU(),
        #
        #     torch.nn.AdaptiveAvgPool1d(1),
        # )

        self.transition = torch.nn.Sequential(
            torch.nn.Conv2d(s, k, (1,n)),
            torch.nn.BatchNorm2d(k),
            torch.nn.ReLU(),

            _ResidueModule(k),
            _ResidueModule(k),
            torch.nn.AvgPool2d(kernel_size=(2,1)),
            _ResidueModule(k),
            _ResidueModule(k),
            torch.nn.AvgPool2d(kernel_size=(2,1)),
            _ResidueModule(k),
            _ResidueModule(k),
            torch.nn.AvgPool2d(kernel_size=(2,1)),

            torch.nn.Conv2d(k, l, (c//8,1)),
            torch.nn.BatchNorm2d(l),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            )

        self.classifier = torch.nn.Linear(l, 3)

    def forward(self, x):
        """Predict phylogenetic trees for the given sequences.

        Parameters
        ----------
        x : torch.Tensor
            One-hot encoded sequences.

        Returns
        -------
        torch.Tensor
            The predicted adjacency trees.
        """

        # x = x.view(x.size()[0], 16, -1)
        # x = self.conv(x).squeeze(dim=2)

        x = self.transition(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class _ResidueModule(torch.nn.Module):

    def __init__(self, channel_count):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(channel_count, channel_count, (1, 1)),
            torch.nn.BatchNorm2d(channel_count),
            torch.nn.ReLU(),
            torch.nn.Conv2d(channel_count, channel_count, (1, 1)),
            torch.nn.BatchNorm2d(channel_count),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        return x + self.layers(x)

training_data = np.load(f"/Users/rhuck/Downloads/DL_Phylo/Recombination/data_generation/test_data/{data_test}_train.npy", allow_pickle = True)
dev_data = np.load(f"/Users/rhuck/Downloads/DL_Phylo/Recombination/data_generation/test_data/{data_test}_dev.npy", allow_pickle = True)
train_data = training_data.tolist()
validation_data = dev_data.tolist()
print("Train Set Size: ", len(train_data))
print("Development Set Size: ", len(validation_data))
c = len(dev_data[0][0][0])

#plotting
vis = visdom.Visdom()

#model Hyperparameters
model = _Model()

# #Load Model
# load_path = f"/Users/rhuck/Downloads/DL_Phylo/Recombination/models/{data_test}." + str(epoch)
# model = torch.load(load_path)



optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
loss_function = torch.nn.CrossEntropyLoss(reduction='sum')

BATCH_SIZE = 16
TRAIN_SIZE = 10000#len(train_data)#3600
epoch = 1

#Train
while epoch < 300:

    #TRAIN
    model.train()

    #randomly sample TRAIN_SIZE number of datapoints
    epoch_train = random.sample(train_data, TRAIN_SIZE)
    sample_count, correct, score = 0, 0, 0.0

    for i in range(TRAIN_SIZE // BATCH_SIZE):
        data = epoch_train[i * BATCH_SIZE : (i+1) * BATCH_SIZE]

        x_list = []
        y_list = []

        for datapoint in data: #transformed_data:
            sequences = datapoint[0]
            label = datapoint[1]
            x_list.append(sequences)
            y_list.append(label)


        x = torch.tensor(x_list, dtype=torch.float)
        x = x.view(BATCH_SIZE, s, -1, n)
        y = torch.tensor(y_list)
        sample_count += x.size()[0]

        optimizer.zero_grad()
        output = model(x)
        loss = loss_function(output, y)
        loss.backward()
        optimizer.step()

        score += float(loss)
        _, predicted = torch.max(output.data, 1)
        correct += (predicted == y).sum().item()

        print("\n", predicted, y, "\n")

    score /= sample_count
    accuracy = correct / sample_count

    print("\n\n", "Epoch: \n", epoch, "Train Acc: ", accuracy, "Train Score: ", score)

    vis.line(
        X = [epoch],
        Y = [accuracy],
        opts= dict(title=graph_title,
               xlabel="Epochs",
               showlegend=True),
        win= graph_win,
        name = "Train Accuracy",
        update="append"
        )
    vis.line(
        X = [epoch],
        Y = [score],
        win= graph_win,
        name = "Train Score",
        update="append"
        )

    ##VALIDATE
    optimizer.zero_grad()
    model.train(False)
    sample_count, correct, score = 0, 0, 0.0

    tree_0_len, tree_1_len, tree_2_len = 0, 0, 0
    guess_0, guess_1, guess_2 = 0,0,0
    real_0, real_1, real_2 = 0,0,0

    #NO PERMUTE -- batch size of 1
    for x, y in validation_data:

        x = torch.tensor(x, dtype=torch.float)
        x = x.view(1, s, -1, n)
        y = torch.tensor([y])
        sample_count += x.size()[0]

        output = model(x)
        loss = loss_function(output, y)

        score += float(loss)
        _, predicted = torch.max(output.data, 1)
        correct += (predicted == y).sum().item()

        print("\n", predicted, y, "\n")


    score /= sample_count
    accuracy = correct / sample_count

    print("\n", "Val Acc: ", accuracy, "Val Score: ", score)

    vis.line(
        X = [epoch],
        Y = [accuracy],
        win= graph_win,
        name = "Dev Accuracy",
        update="append"
        )
    vis.line(
        X = [epoch],
        Y = [score],
        win= graph_win,
        name = "Dev Score",
        update="append"
        )

    #save MODEL
    save_path = f"/Users/rhuck/Downloads/DL_Phylo/Recombination/recombination_networks/models/{data_test}_{model_number}." + str(epoch)
    torch.save(model.state_dict(), save_path)

    epoch += 1
