import numpy as np
from pommerman.nn import data_processing
import torch
from torch import optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import pommerman.nn.utils

import os


def get_nn_input(state, trans_obj):
    return trans_obj.planeFilling(state, trans_obj.planes)


def get_nn_target(actions, imitated_agent_nr):
    #print(actions, imitated_agent_nr)
    action = actions[imitated_agent_nr][0]
    ret = [0, 0, 0, 0, 0, 0]
    ret[action] = 1

    return ret


def train_net(model, nn_inputs, nn_targets):
    param = {
        "epochs": 100,
        "batch_size": 32,
        "l_rate": 0.01,
        "path": "./saved_models/checkpoints",
        "test_size": 0.3
    }

    if not os.path.exists(param["path"]):
        os.makedirs(param["path"])


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device: ",device, " cuda av: ", torch.cuda.is_available(), " cuda device: ", torch.cuda.device(0), torch.cuda.device_count(), torch.cuda.get_device_name(0))

    X_train, X_test, y_train, y_test = train_test_split(nn_inputs, nn_targets, test_size=param["test_size"])

    train_loader, test_loader = DataLoader(data_processing.Dataset(X_train, y_train), batch_size=param["batch_size"],
                                           shuffle=True), DataLoader(data_processing.Dataset(X_test, y_test),
                                                                     batch_size=param["batch_size"], shuffle=True)
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=param["l_rate"])
    criterion = torch.nn.MSELoss()

    training = data_processing.Training(param["epochs"], param["batch_size"], optimizer, criterion, model, device)

    training.train_setup(train_loader, param["path"])



    training.evaluate(test_loader)

    return model
