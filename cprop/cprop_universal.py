from collections import defaultdict

import torch
from torch.optim import Optimizer

from .cprop_lib import *


class CProp(Optimizer):
    """CProp optimizer wrapper.
    To use CProp with any optimizer. 
    However this is significantly SLOWER than its specialized counterparts.
    """
    def __init__(
            self,
            optimizer: Optimizer,
            beta=0.999,
            c=1,
            eps=1e-8,
            cdf='bft',
    ):
        assert isinstance(optimizer, Optimizer)
        if not 0.0 <= eps:
            raise ValueError("Invalid eps value: {}".format(eps))
        if not 0.0 <= c:
            raise ValueError("Invalid c value: {}".format(c))
        if not 0.0 <= beta < 1.0:
            raise ValueError("Invalid beta parameter: {}".format(beta))
        self.opt = optimizer
        self.defaults = dict(cprop_beta=beta,
                             cprop_c=c,
                             cprop_eps=eps,
                             cprop_cdf=cdf)

        self.param_groups = self.opt.param_groups
        self.state = defaultdict(dict)

        # set hyperparameters of cprop in param groups
        for group in self.param_groups:
            for k, v in self.defaults.items():
                group[k] = v

    def state_dict(self):
        cprop_state = super(CProp, self).state_dict()
        opt_state = self.opt.state_dict()
        return {
            'cprop_state': cprop_state['state'],
            **opt_state,
        }

    def load_state_dict(self, state_dict):
        self.opt.load_state_dict(
            {k: v
             for k, v in state_dict.items() if k != 'cprop_state'})
        super(CProp, self).load_state_dict({
            'param_groups':
            state_dict['param_groups'],
            'state':
            state_dict['cprop_state'],
        })

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        # get params before update
        before_params = []
        for group in self.param_groups:
            before_params.append([p.clone() for p in group['params']])

        # step the original optimizer
        loss = self.opt.step(closure)

        # alter the updates
        for before, group in zip(before_params, self.param_groups):
            beta = group['cprop_beta']
            c = group['cprop_c']
            eps = group['cprop_eps']
            cdf = group['cprop_cdf']
            for b, p in zip(before, group['params']):
                if p.grad is None:
                    continue
                g = p.grad.data
                state = self.state[p]

                # cprop scales
                s = cprop(state, g, beta, c, eps, cdf)

                # alter the updates
                p.data = scale_update(b, p, s)

        return loss


@torch.jit.script
def scale_update(before, after, scale):
    return before + (after - before) * scale
