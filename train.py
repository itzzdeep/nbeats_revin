import argparse

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
#from torch import optim
from torch.nn import functional as F

from nbeats import NBeatsNet
from RevIN import RevIN


warnings.filterwarnings(action='ignore', message='Setting attributes')

def plot_scatter(*args, **kwargs):
    plt.plot(*args, **kwargs)


def data_generator(x, y, size):
    assert len(x) == len(y)
    batches = []
    for ii in range(0, len(x), size):
        batches.append((x[ii:ii + size], y[ii:ii + size]))
    for batch in batches:
        yield batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',type=str, help='path of data')
    parser.add_argument('--column',type=str, help='Column name of the TS data')
    parser.add_argument('--fcast_length',type=int,
              help='forecast length')
    parser.add_argument('--bcast_multipler',type=int,default=3,
              help='multiplier to forecast to get backcast length')
    parser.add_argument('--hl_units',type=int,default=128,
              help='hidden_layer_units')
    parser.add_argument('--split_ratio',type=float,default=.75,
              help='split ratio of train & test')
    parser.add_argument('--lr', type=float, default=5*1e-4, metavar='LR',
                          help='learning rate (default: 5*1e-4)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                          help='random seed (default: 1)')
    parser.add_argument('--n_features',type=int,default=1, help='univarite or multivariate')
    parser.add_argument('--bs',type=int,default=10,
              help='batch size')
    parser.add_argument('--n_epoch',type=int,default=50,
              help='no of epoch')


    
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = pd.read_csv(args.dataset, index_col=0, parse_dates=True)
    data = data[args.column].values.flatten() 

    num_features = args.n_features
    forecast_length = args.fcast_length
    backcast_length =  args.bcast_multipler * forecast_length
    batch_size = args.bs

    x, y = [], []
    for epoch in range(backcast_length, len(data) - forecast_length):
        x.append(data[epoch - backcast_length:epoch])
        y.append(data[epoch:epoch + forecast_length])
    x = np.array(x)
    y = np.array(y)

    
    #split train/test.
    c = int(len(x) * args.split_ratio)
    x_train, y_train = x[:c], y[:c]
    x_test, y_test = x[c:], y[c:]



        # model
    net = NBeatsNet(
        stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK),
        forecast_length=forecast_length,
        backcast_length=backcast_length,
        hidden_layer_units=args.hl_units,
    )


    optimiser = torch.optim.Adam(lr=args.lr, params=net.parameters())


    # RevIN

    revin_train = RevIN(args.n_features)


    grad_step = 0
    for epoch in range(args.n_epoch):
        # train.
        net.train()
        train_loss = []
        for x_train_batch, y_train_batch in data_generator(x, y, batch_size):
            grad_step += 1
            optimiser.zero_grad()
            x_train_batch = torch.tensor(x_train_batch, dtype=torch.float)
            x_train_batch = revin_train(x_train_batch,'norm')
            _, forecast = net(x_train_batch.to(net.device))
            forecast = revin_train(forecast,'denorm')
            loss = F.mse_loss(forecast, torch.tensor(y_train_batch, dtype=torch.float))
            train_loss.append(loss.item())
            loss.backward()
            optimiser.step()
        train_loss = np.mean(train_loss)


        if epoch % 20 == 0:
            net.eval()
            x_test = torch.tensor(x_test, dtype=torch.float)
            x_test = revin_train(x_test,'norm')
            _,forecast_test = net(x_test)
            forecast_test = revin_train(forecast_test,'denorm')
            test_loss = F.mse_loss(forecast_test, torch.tensor(y_test, dtype=torch.float)).item()

            print(f'epoch = {str(epoch).zfill(4)}, '
                  f'tr_loss  = {train_loss}, '
                  f'te_loss = {test_loss}')
