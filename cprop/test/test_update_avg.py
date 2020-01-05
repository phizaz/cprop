from cprop.cprop_lib import _update_avg_x
import torch

m = torch.tensor([1., 2.])
x = torch.tensor([1., 1.])
beta = 0.9

_update_avg_x(m, x, beta)

print(m)
