# CProp for Tensorflow

This is experimental.

I personally don't have much experience with tensorflow. Pull-requests are so much appreciated.

# What's included 

I have implemented two version of optimizes: `tf.train.Optimizer`-based and `tf.keras.optimizers.Optimzer`-based. 

CProp optimizer wrapper which is slow but works with any optimizer. It is beyond my skill to implement each for every popular optimizer.

# Installation 

```
# clone 
# cd to this directory
# install with editable mode
pip install -e . 
```

# Usage

## `train.Optimizer` style:

```
from cprop_tf.cprop_tf1 import CProp

opt = tf.train.AdamOptimizer(lr=1e-3)
opt = CProp(opt, beta=0.999, c=1, cdf='bft')
```

## `keras.optimizers` style: 

```
from cprop_tf.cprop import CProp

opt = tf.keras.optimizers.Adam(lr=1e-3)
opt = CProp(opt, beta=0.999, c=1, cdf='bft')
```

Options are the same as that of [Pytorch](https://github.com/phizaz/cprop/tree/master)

You can see an example (keras style) in `cprop_cifar10.ipynb`

# Plots

## Cifar10
![alt text](https://raw.githubusercontent.com/phizaz/cprop/master/tf/cifar10.png)