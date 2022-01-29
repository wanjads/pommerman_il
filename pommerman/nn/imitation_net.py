import numpy as np
from pommerman.nn import data_processing
import torch
from torch import optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


def get_nn_input(state):

    return np.random.rand(18, 11, 11)


def get_nn_target(actions, imitated_agent_nr):

    action = actions[imitated_agent_nr][0]
    ret = [0, 0, 0, 0, 0, 0]
    ret[action] = 1

    return ret


def train_net(model, nn_inputs, nn_targets):
    param = {
        "epochs": 1,
        "batch_size": 32,
        "l_rate": 0.001,
        "path": "./saved_models/model_weights.pt",
        "test_size": 0.3
    }

    X_train, X_test, y_train, y_test = train_test_split(nn_inputs, nn_targets, test_size=param["test_size"])

    train_loader, test_loader = DataLoader(data_processing.Dataset(X_train, y_train), batch_size=param["batch_size"],
                                           shuffle=True), DataLoader(data_processing.Dataset(X_test, y_test),
                                                                     batch_size=param["batch_size"], shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=param["l_rate"])
    criterion = torch.nn.MSELoss()

    training = data_processing.Training(param["epochs"], param["batch_size"], optimizer, criterion, model)

    training.train_setup(train_loader, param["path"])
    training.evaluate(test_loader)

    return model
