import torch
from torch.optim import Optimizer

from .cprop_lib import *


class CPropAdam(Optimizer):
    """
    CProp-augmented Adam.
    Based on Pytorch's Adam.
    """
    def __init__(self,
                 params,
                 lr=1e-3,
                 cprop_beta=0.999,
                 cprop_c=1,
                 cprop_eps=1e-8,
                 cprop_cdf='bft',
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0,
                 amsgrad=False):
        if not 0.0 <= cprop_eps:
            raise ValueError("Invalid cprop_eps value: {}".format(cprop_eps))
        if not 0.0 <= cprop_c:
            raise ValueError("Invalid cprop_c value: {}".format(cprop_c))
        if not 0.0 <= cprop_beta < 1.0:
            raise ValueError(
                "Invalid cprop_beta parameter: {}".format(cprop_beta))

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(
                betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(
                betas[1]))
        defaults = dict(
            cprop_beta=cprop_beta,
            cprop_c=cprop_c,
            cprop_eps=cprop_eps,
            cprop_cdf=cprop_cdf,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
        super(CPropAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(CPropAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

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
            cprop_beta = group['cprop_beta']
            cprop_c = group['cprop_c']
            cprop_eps = group['cprop_eps']
            cprop_cdf = group['cprop_cdf']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead'
                    )
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1**state['step']
                bias_correction2 = 1 - beta2**state['step']

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                # cprop
                # note: need to wait for moving averages to be calculated
                # in case of sharing the averages
                scale = cprop(
                    state,
                    grad=grad,
                    beta=cprop_beta,
                    c=cprop_c,
                    eps=cprop_eps,
                    cdf=cprop_cdf,
                    # share the moving averages if possible
                    m=(exp_avg if beta1 == cprop_beta else None),
                    v=(exp_avg_sq if beta2 == cprop_beta else None),
                )

                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() /
                             math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() /
                             math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                # scale the gradient update
                p.data.add_(update(step_size, exp_avg, denom, scale))

        return loss


@torch.jit.script
def update(step_size: float, m, denom, scale):
    return -step_size * m / denom * scale
