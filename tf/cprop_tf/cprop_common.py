import tensorflow as tf


def cprop(grad, m, v, t, beta, c, eps, cdf: str, use_locking):
    # moving averages
    m = m.assign(beta * m + (1 - beta) * grad, use_locking=use_locking)
    v = v.assign(beta * v + (1 - beta) * grad * grad, use_locking=use_locking)

    # get standard error
    bias_correct = (1 - beta**t)
    m_hat = m / bias_correct
    v_hat = v / bias_correct
    n = bias_correct / (1 - beta)
    variance = v_hat - m_hat**2
    se = tf.math.sqrt(tf.math.maximum(variance, 0) / (n - 1 + eps)) + eps

    # get the scale
    if cdf == 'normal':
        p = normal_cdf(m_hat, se)
    elif cdf == 'bft':
        p = best_fit_two(m_hat, se)
    elif cdf == 'bfo':
        p = best_fit_one(m_hat, se)
    else:
        raise NotImplementedError()
    scale = tf.clip_by_value(2 * c * tf.math.abs(p - 0.5), 0, 1)
    return scale


def normal_cdf(loc, sd):
    """normal cdf(0)"""
    from tensorflow_probability import distributions
    # it is not jit-able
    d = distributions.Normal(loc, sd)
    return d.cdf(0)


def best_fit_two(loc, sd):
    """
    return an approximation of cdf(0) with method proposed in:
    Bowling, Shannon R., Mohammad T. Khasawneh, Sittichai Kaewkuekool, and Byung Rae Cho. 2009. 
    “A Logistic Approximation to the Cumulative Normal Distribution.” 
    Journal of Industrial Engineering and Management 2 (1). 
    https://doi.org/10.3926/jiem.2009.v2n1.p114-127.
    """
    z = -loc / sd
    return 1 / (1 + tf.math.exp(-0.07056 * z**3 - 1.5976 * z))


def best_fit_one(loc, sd):
    """
    return an approximation of cdf(0) with method proposed in:
    Bowling, Shannon R., Mohammad T. Khasawneh, Sittichai Kaewkuekool, and Byung Rae Cho. 2009. 
    “A Logistic Approximation to the Cumulative Normal Distribution.” 
    Journal of Industrial Engineering and Management 2 (1). 
    https://doi.org/10.3926/jiem.2009.v2n1.p114-127.
    """
    z = -loc / sd
    return 1 / (1 + tf.math.exp(-1.702 * z))
