# N-BEATS: Neural basis expansion analysis for interpretable time series forecasting and Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift (RevIN)

Pytorch Implementation of N-BEATS and RevIN

## Introduction
N-BEATS focuses on the univariate times series point forecasting problem using deep learning. This a deep neural architecture based on backward and
forward residual links and a very deep stack of fully-connected layers. It is interpretable, applicable without modification to a wide array of target domains, and fast to train, makes it widely useable.

Although different kinds of architechture is widely used for time series forecasting namely Staistical Model, Hybrid Model, Purely Deep Learning models, all the models ignores the problem of distribution shift. Statistical properties such as mean and variance often change over time in time series, i.e., time-series data suffer from a distribution shift problem. To overcome this issue, RevIN provides a simple yet effective normalization method called reversible instance normalization (RevIN), a generally-applicable normalization-and-denormalization method with learnable affine transformation added on symetric layers.

## Installation

1. Install Pytorch >= v.1.8.0
2. Clone this Repository.
   ```linux
   git clone https://github.com/itzzdeep/nbeats_revin.git
   ```

## Usage
N-NEATS has two configurations of the architecture. One of them is generic DL, the other one is augmented with certain inductive biases to be interpretable.The generic architecture does not rely on Time Series-specific knowledge. The interpretable architecture can be constructed by reusing the overall architectural approach and by adding structure to basis layers at stack level.

Here we are showing the usage of Generic model using RevIN.

```python
from nbeats import NBeatsNet
from RevIN import RevIN

net = NBeatsNet(
        stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK),
        forecast_length=forecast_length,
        backcast_length=backcast_length,
        hidden_layer_units=128,
    )
    
revin_layer = RevIN(num_features)
x_in = revin_layer(x_in, 'norm')
x_out = net(x_in) # your model or subnetwork within the model
x_out = revin_layer(x_out, 'denorm')
```
