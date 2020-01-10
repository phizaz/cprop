# CProp: Adaptive Learning Rate Scaling from Past Gradient Conformity

Implementation of CProp in Pytorch.

[Looking for Tensorflow version?](https://github.com/phizaz/cprop/tree/master/tf)

A preprint Arxiv version can be found at https://arxiv.org/abs/1912.11493.

Paper is being reviewed.

## Installation

Requires Python with type-hint support (I guess 3.6+).

It seems to require Pytorch 1.2 due to its use of JIT.

```
# clone 
...
# install
pip install -e .
```

## What's included 

1. CPropSGD. CProp-augmented SGD.
2. CPropAdam. CProp-augmented Adam.
3. CProp universal wrapper. Slower but could be used with any optimizer.

## Usage

```
import cprop

opt = cprop.CPropSGD(net.parameters(), lr=0.1, cprop_beta=0.999, cprop_c=1, cprop_cdf='normal')
...
# use it as usual
opt.step()
```

`cprop_beta` is the gradint horizon. Default 0.999 works well. 

`cprop_c` is overconfidence coefficient. Default 1. Try it first. If you want better generalization, use larger `c`, e.g. 3, 5.

`cprop_cdf` how to compute CDF. `normal` is the most correct way to do it (but slowest). `bft` using Logistic approximation which is ~10% faster with no observable performance loss.

With any optimizer:

```
opt = any optimizer
opt = cprop.CProp(opt, beta=0.999, c=1, cdf='normal')
...
# use it as usual
opt.step()
```

## Plots

### Quick test on Cifar10
![alt text](https://raw.githubusercontent.com/phizaz/cprop/master/cifar10_small.png)

### Image classification:
![alt text](https://raw.githubusercontent.com/phizaz/cprop/master/plots/cifar100-vgg-bn-github.png)

### Language modeling:
![alt text](https://raw.githubusercontent.com/phizaz/cprop/master/plots/ptb-lstm-dropout-github.png)



