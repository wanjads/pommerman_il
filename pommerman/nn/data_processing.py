"""
In this module inherits all data processing classes for 
"""

import numpy as np
import pandas as pd
import time, random, math

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pommerman.nn.utils

import sklearn
from sklearn.model_selection import train_test_split

import os


class Training():
    def __init__(self, epochs, batch_size, optimizer, criterion, model, device):
        self.optimizer, self.criterion, self.model = optimizer, criterion, model
        self.epochs, self.batch_size = epochs, batch_size

        self.best_train_loss = 5

        self.device = device

    def evaluate(self, iterator):
        self.model.eval()

        epoch_loss = 0

        with torch.no_grad():
            for batch, (inp, target) in enumerate(iterator):
                inp = inp.to(self.device)
                target = target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(inp)[1]
               # output = nn.Softmax(output, dim=1)
                if batch == len(iterator) - 1 or batch == 1:
                    print(f'Output: {output}, Target: {target}')

                loss = self.criterion(output, target)
                epoch_loss += loss.item()

        test_loss = epoch_loss / len(iterator)
        print(f'| Test Loss: {test_loss:.3f}')

    def train(self, iterator, epoch, path):
        self.model.train()

        epoch_loss = 0

        for batch, (inp, target) in enumerate(iterator):
            self.optimizer.zero_grad()
            inp = inp.to(self.device)
            target = target.to(self.device)
            output = self.model(inp)[1]
            soft = nn.Softmax(dim=1)
            output = soft(output)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            if batch == len(iterator) - 1:
                print(f'Output: {output[:3]}, Target: {target[:3]}, Sum_out: {torch.sum(output, dim=1)}')

            if epoch_loss / len(iterator) < self.best_train_loss:
                self.best_train_loss = epoch_loss / len(iterator)
                path = os.path.join(path, "torch_state.tar")
                torch.save({'model_state_dict': self.model.state_dict(),
                            # TODO: not sure if self.optimizer.state_dict is the same thing
                            # that timour also saves (Copied from timour)
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            # TODO: not sure if 'iterator_state' is available here
                            # (Copied from timour)
                            #'iterator_state': iterator.sampler.get_state()
                            }, path)

        return epoch_loss / len(iterator)

    def train_setup(self, iterator, path):
        for epoch in range(self.epochs):

            start_time = time.time()

            train_loss = self.train(iterator, epoch, path)

            end_time = time.time()

            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)

            
            print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f}')

    @staticmethod
    def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs


class Dataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def transform(batch):
        pass


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data, label = self.data[idx], self.labels[idx]
        return torch.tensor(data).float(), torch.tensor(label).float()

