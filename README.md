Paper is being reviewed.

# Installation

```
# clone 
...
# install
pip install -e .
```

# What's included 

1. CPropSGD.
2. CPropAdam.
3. CProp universal wrapper. Slower but could be used with any optimizer.

# Usage

```
import cprop

cprop.CPropSGD(net.parameters(), lr=0.1, cprop_beta=0.999, cprop_c=1, cprop_cdf='normal')
```

`cprop_beta` is the gradint horizon. Default 0.999 works well. 

`cprop_c` is overconfidence coefficient. Default 1. Try it first. If you want better generalization, use larger `c`, e.g. 3, 5.

`cprop_cdf` how to compute CDF. `normal` is the most correct way to do it (but slowest). `bft` using Logistic approximation which is ~10% faster with no observable performance loss.




