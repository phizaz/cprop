import math

import torch
from torch.distributions import Normal


def cprop(state, grad, beta, c, eps, cdf, m=None, v=None):
    """
    Boilerplate for cprop in any optimizer

    Args:
        cdf: 'normal', 'bft', 'bfo', 'logistic'
        m, v: supplied custom moving averages (not bias corrected)
    """
    if 'cprop_t' not in state:
        state['cprop_t'] = 0
    if m is None and 'cprop_m' not in state:
        # Exponential moving average of gradient values
        state['cprop_m'] = torch.zeros_like(grad)
    if v is None and 'cprop_v' not in state:
        # Exponential moving average of squared gradient values
        state['cprop_v'] = torch.zeros_like(grad)

    if m is None:
        m = state['cprop_m']
        m.mul_(beta).add_(1 - beta, grad)
    if v is None:
        v = state['cprop_v']
        v.mul_(beta).addcmul_(1 - beta, grad, grad)

    state['cprop_t'] += 1
    t = state['cprop_t']

    if cdf == 'normal':
        s = cprop_scale_normal(m, v, t, beta, c, eps)
    elif cdf == 'bft':
        s = cprop_scale_bft(m, v, t, beta, c, eps)
    elif cdf == 'bfo':
        s = cprop_scale_bfo(m, v, t, beta, c, eps)
    elif cdf == 'logistic':
        s = cprop_scale_logistic(m, v, t, beta, c, eps)
    else:
        raise NotImplementedError()
    return s


def normal_cdf(loc, sd):
    """normal cdf(0)"""
    # it is not jit-able
    d = Normal(loc, sd)
    return d.cdf(0)


@torch.jit.script
def logistic_cdf(loc, sd):
    """using the same variance logistic distribution as an approximation; cdf(0)"""
    s = sd.mul(math.sqrt(3) / math.pi)
    return 1 / ((loc / s).exp() + 1)


@torch.jit.script
def best_fit_two(loc, sd):
    """
    return an approximation of cdf(0) with method proposed in:
    Bowling, Shannon R., Mohammad T. Khasawneh, Sittichai Kaewkuekool, and Byung Rae Cho. 2009. 
    “A Logistic Approximation to the Cumulative Normal Distribution.” 
    Journal of Industrial Engineering and Management 2 (1). 
    https://doi.org/10.3926/jiem.2009.v2n1.p114-127.
    """
    z = -loc / sd
    return 1 / (1 + (-0.07056 * z**3 - 1.5976 * z).exp())


@torch.jit.script
def best_fit_one(loc, sd):
    """
    return an approximation of cdf(0) with method proposed in:
    Bowling, Shannon R., Mohammad T. Khasawneh, Sittichai Kaewkuekool, and Byung Rae Cho. 2009. 
    “A Logistic Approximation to the Cumulative Normal Distribution.” 
    Journal of Industrial Engineering and Management 2 (1). 
    https://doi.org/10.3926/jiem.2009.v2n1.p114-127.
    """
    z = -loc / sd
    return 1 / (1 + (-1.702 * z).exp())


@torch.jit.script
def cprop_se(m, v, t: float, beta: float, eps: float):
    """calculate the standard error using moving statistics"""
    bias_correct = (1 - beta**t)
    m = m / bias_correct
    v = v / bias_correct
    n = bias_correct / (1 - beta)
    se = ((v - m**2).clamp(min=0) / (n - 1 + eps)).sqrt() + eps
    return m, se


@torch.jit.script
def _cprop_scale(cdf, c: float):
    """scale given confidence (probability < 0)"""
    return (2 * c * (cdf - 0.5).abs()).clamp(max=1)


def cprop_scale_normal(m, v, t: float, beta: float, c: float, eps: float):
    m, se = cprop_se(m, v, t, beta, eps)
    # normal is not jit-able
    cdf = normal_cdf(m, se)
    return _cprop_scale(cdf, c)


@torch.jit.script
def cprop_scale_bft(m, v, t: float, beta: float, c: float, eps: float):
    m, se = cprop_se(m, v, t, beta, eps)
    cdf = best_fit_two(m, se)
    return _cprop_scale(cdf, c)


@torch.jit.script
def cprop_scale_bfo(m, v, t: float, beta: float, c: float, eps: float):
    m, se = cprop_se(m, v, t, beta, eps)
    cdf = best_fit_one(m, se)
    return _cprop_scale(cdf, c)


@torch.jit.script
def cprop_scale_logistic(m, v, t: float, beta: float, c: float, eps: float):
    m, se = cprop_se(m, v, t, beta, eps)
    cdf = logistic_cdf(m, se)
    return _cprop_scale(cdf, c)
