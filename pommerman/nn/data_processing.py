import numpy as np
import pandas as pd
import time, random, math

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import Dataset, DataLoader

import sklearn
from sklearn.model_selection import train_test_split


class Training():
    def __init__(self, epochs, batch_size, optimizer, criterion, model):
        self.optimizer, self.criterion, self.model = optimizer, criterion, model
        self.epochs, self.batch_size = epochs, batch_size

        self.best_train_loss = 0

    def evaluate(self, iterator):
        self.model.eval()

        epoch_loss = 0

        with torch.no_grad():
            for batch, (inp, target) in enumerate(iterator):
                self.optimizer.zero_grad()
                output = self.model(inp)[1]

                loss = self.criterion(output, target)
                epoch_loss += loss.item()

        test_loss = epoch_loss / len(iterator)
        print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

    def train(self, iterator):
        self.model.train()

        epoch_loss = 0

        for batch, (inp, target) in enumerate(iterator):
            # print(batch, inp)
            self.optimizer.zero_grad()
            output = self.model(inp)[1]

            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(iterator)

    def train_setup(self, iterator, path):
        for epoch in range(self.epochs):

            start_time = time.time()

            train_loss = self.train(iterator)

            end_time = time.time()

            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)

            if train_loss < self.best_train_loss:
                self.best_train_loss = train_loss
                torch.save({'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            }, path)

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