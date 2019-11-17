import torch
from torch.optim import Optimizer

from .cprop_lib import *


class CPropSGD(Optimizer):
    """
    Based on Pytorch's SGD.
    """
    def __init__(
            self,
            params,
            lr,
            cprop_beta=0.999,
            cprop_c=1,
            cprop_eps=1e-8,
            cprop_cdf='bft',
            momentum=0,
            dampening=0,
            weight_decay=0,
            nesterov=False,
    ):
        if not 0.0 <= cprop_eps:
            raise ValueError("Invalid cprop_eps value: {}".format(cprop_eps))
        if not 0.0 <= cprop_c:
            raise ValueError("Invalid cprop_c value: {}".format(cprop_c))
        if not 0.0 <= cprop_beta < 1.0:
            raise ValueError(
                "Invalid cprop_beta parameter: {}".format(cprop_beta))

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(
            cprop_beta=cprop_beta,
            cprop_c=cprop_c,
            cprop_eps=cprop_eps,
            cprop_cdf=cprop_cdf,
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                "Nesterov momentum requires a momentum and zero dampening")
        super(CPropSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(CPropSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            beta = group['cprop_beta']
            c = group['cprop_c']
            eps = group['cprop_eps']
            cdf = group['cprop_cdf']

            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                state = self.state[p]

                s = cprop(state, d_p, beta, c, eps, cdf)

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(
                            d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.addcmul_(-group['lr'], d_p, s)

        return loss
