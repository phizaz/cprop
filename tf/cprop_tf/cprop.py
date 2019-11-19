import math

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from .cprop_common import *


class CProp(tf.keras.optimizers.Optimizer):
    """
    Implementation of CProp as an optimizer wrapper.
    """
    def __init__(self,
                 optimizer,
                 beta: float = 0.999,
                 c: float = 1,
                 eps: float = 1e-8,
                 cdf: str = 'bft',
                 name="cprop",
                 **kwargs):
        super(CProp, self).__init__(name, **kwargs)

        if isinstance(optimizer, str):
            optimizer = tf.keras.optimizers.get(optimizer)
        if not isinstance(optimizer, tf.keras.optimizers.Optimizer):
            raise TypeError(
                "optimizer is not an object of tf.keras.optimizers.Optimizer")

        self._optimizer = optimizer
        self._set_hyper('cprop_beta', beta)
        self._set_hyper('cprop_c', c)
        self._set_hyper('cprop_eps', eps)
        # we want to use python control flow
        self.cdf = cdf

    def _create_slots(self, var_list):
        self._optimizer._create_slots(var_list=var_list)
        # set slots for first and second moments
        for var in var_list:
            self.add_slot(var, 'cprop_m')
            self.add_slot(var, 'cprop_v')

    def _create_hypers(self):
        self._optimizer._create_hypers()

    def _prepare(self, var_list):
        return self._optimizer._prepare(var_list=var_list)

    def apply_gradients(self, grads_and_vars, name=None):
        self._optimizer._iterations = self.iterations
        return super(CProp, self).apply_gradients(grads_and_vars, name)

    def _resource_apply_dense(self, grad, var):
        """main method"""
        beta = self._get_hyper('cprop_beta')
        c = self._get_hyper('cprop_c')
        eps = self._get_hyper('cprop_eps')
        t = tf.cast(self.iterations + 1, tf.dtypes.float32)
        m = self.get_slot(var, 'cprop_m')
        v = self.get_slot(var, 'cprop_v')

        # cprop scale
        scale = cprop(grad, m, v, t, beta, c, eps, self.cdf, self._use_locking)

        # back a copy of var
        var_old = tf.identity(var)

        # step the main optimizer
        train_op = self._optimizer._resource_apply_dense(grad, var)

        # wait for it to finish
        with tf.control_dependencies([train_op]):
            # alter the update
            cprop_op = var.assign(var_old + scale * (var - var_old),
                                  use_locking=self._use_locking)

        return tf.group(
            train_op,
            cprop_op,
        )

    def get_config(self):
        config = {
            'optimizer': tf.keras.optimizers.serialize(self._optimizer),
            'cprop_beta': self._serialize_hyperparameter('cprop_beta'),
            'cprop_c': self._serialize_hyperparameter('cprop_c'),
            'cprop_eps': self._serialize_hyperparameter('cprop_eps'),
            'cdf': self.cdf,
        }
        base_config = super(CProp, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        optimizer = tf.keras.optimizers.deserialize(
            config.pop('optimizer'),
            custom_objects=custom_objects,
        )
        return cls(optimizer, **config)
