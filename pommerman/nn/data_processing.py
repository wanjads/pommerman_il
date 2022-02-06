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

                loss = self.criterion(output, target)
                epoch_loss += loss.item()

        test_loss = epoch_loss / len(iterator)
        print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

    def train(self, iterator, epoch, path):
        self.model.train()

        epoch_loss = 0

        for batch, (inp, target) in enumerate(iterator):
            self.optimizer.zero_grad()
            inp = inp.to(self.device)
            target = target.to(self.device)
            output = self.model(inp)[1]

            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

            if epoch_loss / len(iterator) < self.best_train_loss and batch == len(iterator) - 1:
                print(f'Output: {output[:3]}, Target: {target[:3]}')
                self.best_train_loss = epoch_loss / len(iterator)
                path = os.path.join(path, "iter_checkpoint.pt")
                torch.save({'checkpoint_epoch': epoch,
                            # 'checkpoint_early_stopping': early_stopping.counter,
                            'checkpoint_idx': batch,
                            'model_state_dict': self.model.state_dict(),
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
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')

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


"""def main():
    param = {
        "epochs": 100,
        "batch_size": 10,
        "l_rate": 0.001,
        "path": "./saved_models/model_weights.pt",
        "test_size": 0.5
    }

    transformator = PositionDefinition()
    translator = CommunicationProtocol()

    elements = [1, 2, 5, 10, 11, 12, 13]

    data = pd.read_csv("msg_pred_data.csv")
    data = data.drop(columns=["Unnamed: 0"], axis=1)
    data = [np.array(data["current_obs"]), ]
    current_obs = []
    last_obs = []
    data["last_obs"]
    message = []
    data["message"]

    input = data[['current_obs', 'last_obs', 'message']].copy()
    labels = data[["current_true_obs"]]

    X_train, X_test, y_train, y_test = train_test_split(input, labels, test_size=param["test_size"])

    train_loader, test_loader = DataLoader(Dataset(X_train, y_train, elements), batch_size=param["batch_size"],
                                           shuffle=True), DataLoader(Dataset(X_test, y_test, elements),
                                                                     batch_size=param["batch_size"], shuffle=True)

    model = PositionPrediction(len(elements) * 3)
    optimizer = optim.Adam(model.parameters(), lr=param["l_rate"])
    criterion = nn.MSELoss()

    training = Training(param["epochs"], param["batch_size"], optimizer, criterion, model)

    training.train_setup(data, param["path"])
    training.evaluate(data)


if __name__ == main():
    main()"""