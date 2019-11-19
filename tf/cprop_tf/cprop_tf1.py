import math

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.eager import context

from .cprop_common import *
from collections import defaultdict


class CProp(tf.compat.v1.train.Optimizer):
    """
    Implementation of CProp as an optimizer wrapper.
    Supports Tensorflow V1
    """
    def __init__(self,
                 optimizer,
                 beta: float = 0.999,
                 c: float = 1,
                 eps: float = 1e-8,
                 cdf: str = 'bft',
                 use_locking=False,
                 name="cprop"):
        super(CProp, self).__init__(use_locking, name)

        assert isinstance(optimizer, tf.compat.v1.train.Optimizer)

        self._optimizer = optimizer
        self._beta = beta
        self._c = c
        self._eps = eps
        self.cdf = cdf

        # Tensor versions of the constructor arguments, created in _prepare().
        self._beta_t = None
        self._c_t = None
        self._eps_t = None

    def _get_step(self):
        with tf.init_scope():
            if context.executing_eagerly():
                graph = None
            else:
                graph = tf.get_default_graph()
            return self._get_non_slot_variable('cprop_step', graph=graph)

    def _create_slots(self, var_list):
        self._optimizer._create_slots(var_list=var_list)

        # based from adam
        first_var = min(var_list, key=lambda x: x.name)
        self._create_non_slot_variable(initial_value=1,
                                       name='cprop_step',
                                       colocate_with=first_var)

        # set slots for first and second moments
        for v in var_list:
            self._zeros_slot(v, 'cprop_m', self._name)
            self._zeros_slot(v, 'cprop_v', self._name)
            self._zeros_slot(v, 'cprop_buf', self._name)

    def _prepare(self):
        self._optimizer._prepare()

        beta = self._call_if_callable(self._beta)
        c = self._call_if_callable(self._c)
        eps = self._call_if_callable(self._eps)

        self._beta_t = ops.convert_to_tensor(beta, name='cprop_beta')
        self._c_t = ops.convert_to_tensor(c, name='cprop_c')
        self._eps_t = ops.convert_to_tensor(eps, name='cprop_eps')

    def _apply_dense(self, grad, var):
        """used with tf.layers"""
        beta = tf.cast(self._beta_t, var.dtype.base_dtype)
        c = tf.cast(self._c_t, var.dtype.base_dtype)
        eps = tf.cast(self._eps_t, var.dtype.base_dtype)
        t = tf.cast(self._get_step(), var.dtype.base_dtype)
        m = self.get_slot(var, 'cprop_m')
        v = self.get_slot(var, 'cprop_v')

        # cprop scale
        scale = cprop(grad, m, v, t, beta, c, eps, self.cdf, self._use_locking)

        # back a copy of var
        # note: identity trick doesn't work, the change propagates to var_old still
        var_old = self.get_slot(var, 'cprop_buf')
        var_old = var_old.assign(var, use_locking=self._use_locking)

        # step the main optimizer
        with tf.control_dependencies([var_old]):
            train_op = self._optimizer._apply_dense(grad, var)

        # wait for it to finish
        with tf.control_dependencies([train_op]):
            # alter the update
            cprop_op = var.assign(var_old + scale * (var - var_old),
                                  use_locking=self._use_locking)

        return tf.group(
            train_op,
            cprop_op,
        )

    def _resource_apply_dense(self, grad, var):
        """used with tf.keras.layers, almost a clone from self._apply_dense"""
        beta = tf.cast(self._beta_t, var.dtype.base_dtype)
        c = tf.cast(self._c_t, var.dtype.base_dtype)
        eps = tf.cast(self._eps_t, var.dtype.base_dtype)
        t = tf.cast(self._get_step(), var.dtype.base_dtype)
        m = self.get_slot(var, 'cprop_m')
        v = self.get_slot(var, 'cprop_v')

        # cprop scale
        scale = cprop(grad, m, v, t, beta, c, eps, self.cdf, self._use_locking)

        # back a copy of var
        # note: identity trick doesn't work, the change propagates to var_old still
        var_old = self.get_slot(var, 'cprop_buf')
        var_old = var_old.assign(var, use_locking=self._use_locking)

        # step the main optimizer
        with tf.control_dependencies([var_old]):
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

    def _finish(self, update_ops, name_scope):
        with tf.control_dependencies(update_ops):
            step = self._get_step()
            update_step = tf.assign_add(step, 1, use_locking=self._use_locking)

        return tf.group(*update_ops + [update_step], name=name_scope)
