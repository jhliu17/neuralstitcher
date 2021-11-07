"""
Implementation of the autodifferentiation Functions for Tensor.
"""
import numpy as np

from .autodiff import FunctionBase
from .tensor import Tensor


class Function(FunctionBase):
    """Constructors for Batchify Function

    Each function takes `data_type` as tensor inputs and represents a batch of data.
        * To finish forward propogation, we should retain the batch dim.
        * We don't need to reduce the batch dimension in batch backward propoagation. And the gradient
          tensor will be automactically reduced or broadcast to the shape of input tensors by calling
          `reshape_grad` method in our autodiff engine.
    """
    data_type = Tensor

    @staticmethod
    def variable(data: data_type.DATA_TYPE, back=None):
        return Tensor(data, back)

    @staticmethod
    def data(val: data_type):
        return val._tensor


class Add(Function):
    @staticmethod
    def forward(ctx, t1, t2):
        return t1 + t2

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output


class Sum(Function):
    @staticmethod
    def forward(ctx, a):
        val = a.sum()
        return np.array([val])

    @staticmethod
    def backward(ctx, grad_output):
        return np.ones_like(grad_output)


class Mean(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        val = a.mean()
        return np.array([val])

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_values
        return np.ones_like(grad_output) / a.size()


class Neg(Function):
    @staticmethod
    def forward(ctx, a):
        return -a

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output


class Inv(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return 1 / a

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_values
        return -grad_output * (1 / np.square(a))


class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward((a, b))
        return a * b

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_values
        return b * grad_output, a * grad_output


class Exp(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return np.exp(a)

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_values
        return np.exp(a) * grad_output


class Log(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return np.log(a)

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_values
        return 1 / a * grad_output


class ReLU(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return np.where(a > 0, a, 0.)

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_values
        return np.where(a > 0, grad_output, 0.)


class Sigmoid(Function):
    @staticmethod
    def _sigmoid(a):
        return 1 / (1 + np.exp(-a))

    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return Sigmoid._sigmoid(a)

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_values
        b = Sigmoid._sigmoid(a)
        return b * (1 - b)


class Tanh(Function):
    '''
    Ref: https://blogs.cuit.columbia.edu/zp2130/derivative_of_tanh_function/
    '''
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return np.tanh(a)

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_values
        return (1 - np.square(np.tanh(a))) * grad_output


class LinearLayerMatMul(Function):
    @staticmethod
    def forward(ctx, feats, weights):
        ctx.save_for_backward(feats)
        return feats @ weights

    @staticmethod
    def backward(ctx, grad_output):
        feats = ctx.saved_values
        return None, np.expand_dims(feats, axis=-1) * np.expand_dims(grad_output, axis=1)
