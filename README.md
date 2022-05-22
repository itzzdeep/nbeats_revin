# N-BEATS: Neural basis expansion analysis for interpretable time series forecasting and Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift

Pytorch Implementation of N-BEATS and RevIN

## Introduction
N-BEATS focuses on the univariate times series point forecasting problem using deep learning. This a deep neural architecture based on backward and
forward residual links and a very deep stack of fully-connected layers. It is interpretable, applicable without modification to a wide array of target domains, and fast to train, makes it widely useable.

Although different kinds of architechture is widely used for time series forecasting namely Staistical Model, Hybrid Model, Purely Deep Learning models, all the models ignores the problem of distribution shift. Statistical properties such as mean and variance often change over time in time series, i.e., time-series data suffer from a distribution shift problem. To overcome this issue, RevIN provides a simple yet effective normalization method called reversible instance normalization (RevIN), a generally-applicable normalization-and-denormalization method with learnable affine transformation added on symetric layers.
