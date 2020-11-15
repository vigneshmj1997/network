import torch
import torch.nn as nn
import argparse
import pandas as pd
from torch.utils.data import Dataset, SubsetRandomSampler
import torch.optim as optim
import time
import numpy as np
from Model import Project
from data import Data


def train(Model, criterion, epouch, optimiser, train_loader, valid_loader):
    loss_graph = []
    for _ in range(epouch):
        for features, labels in train_loader:
            out = Model(features)
            loss = criterion(out, labels.view(len(labels)))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            loss_graph.append(loss)
            if len(loss_graph) % 1 == 0:
                print("traning loss", loss)
                for j in valid_loader:
                    print(
                        "Validation loss", criterion(Model(j[0]), j[1].view(len(j[1])))
                    )
    print("Traning completed ")


def main():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--file")
    parser.add_argument("--epouch")
    parser.add_argument("--batch_size")
    parser.add_argument("--num_workers")
    args = parser.parse_args()
    batch_size = int(args.batch_size)
    data = Data(args.file)
    validation_split = 0.2
    shuffle_dataset = True
    random_seed = 42
    dataset_size = len(data)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, sampler=train_sampler, num_workers=24
    )
    validation_loader = torch.utils.data.DataLoader(
        data, batch_size=len(valid_sampler), sampler=valid_sampler, num_workers=24
    )

    net = Project()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    train(net, criterion, int(args.epouch), optimizer, train_loader, validation_loader)


if __name__ == "__main__":
    main()
