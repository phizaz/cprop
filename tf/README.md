# CProp for Tensorflow

This is experimental.

I personally don't have much experience with tensorflow. Pull-requests are so much appreciated.

# What's included 

CProp optimizer wrapper which is slow but works with any optimizer. It is beyond my skill to implement each for every popular optimizer.

# Usage

Everything is in `cprop.py` just copy and use it. 

```
opt = tf.keras.optimizers.Adam(lr=1e-3)
if use_cprop:
    opt = CProp(opt, beta=0.999, c=1, cdf='bft')
```

Options are the same as that of [Pytorch](https://github.com/phizaz/cprop/tree/master)

You can see example in `cprop_cifar10.ipynb`

# Plots

## Cifar10
![alt text](https://raw.githubusercontent.com/phizaz/cprop/master/tf/cifar10.png)