"""
Implementation of the core Tensor object for autodifferentiation.
"""
import numpy as np

from .autodiff import Variable


class Tensor(Variable):
    """
    Tensor is a generalization of Scalar in that it is a Variable that
    handles multidimensional arrays.

    Attributes:

        _tensor (:class:`DATA_TYPE`) : the tensor data storage
    """
    DATA_TYPE = np.ndarray
    FLOAT = np.float64  # default float precision

    def __init__(self, v, back=None, name=None):
        assert isinstance(v, Tensor.DATA_TYPE)
        super().__init__(back, name=name)
        self._tensor = v.astype(Tensor.FLOAT)

    def to_numpy(self):
        """
        Returns:
             narray : converted to numpy array
        """
        return self.contiguous()._tensor.reshape(self.shape)

    # Properties
    @property
    def shape(self):
        """
        Returns:
             tuple : shape of the tensor
        """
        return self._tensor.shape

    @property
    def size(self):
        """
        Returns:
             int : size of the tensor
        """
        return np.prod(self.shape)

    @property
    def dims(self):
        """
        Returns:
             int : dimensionality of the tensor
        """
        return len(self._tensor.shape)

    def _ensure_tensor(self, b):
        "Turns a python number into a tensor with the same precision."
        if isinstance(b, (int, float)):
            b = Tensor.make([b], (1,))
        else:
            b._type_(Tensor.FLOAT)
        return b

    # Functions
    def __add__(self, b):
        from .function import Add
        return Add.apply(self, self._ensure_tensor(b))

    def __sub__(self, b):
        from .function import Add
        return Add.apply(self, -self._ensure_tensor(b))

    def __mul__(self, b):
        from .function import Mul
        return Mul.apply(self, self._ensure_tensor(b))

    def __truediv__(self, b):
        from .function import Mul, Inv
        return Mul.apply(
            self, Inv.apply(self._ensure_tensor(b))
        )

    def __neg__(self):
        from .function import Neg
        return Neg.apply(self)

    def sigmoid(self):
        from .function import Sigmoid
        return Sigmoid.apply(self)

    def relu(self):
        from .function import ReLU
        return ReLU.apply(self)

    def log(self):
        from .function import Log
        return Log.apply(self)

    def exp(self):
        from .function import Exp
        return Exp.apply(self)

    def sum(self):
        "Compute the sum over all elements"
        from .function import Sum
        return Sum.apply(self)

    def mean(self):
        "Compute the mean over all elements"
        from .function import Mean
        return Mean.apply(self)

    # def sum(self, dim=None):
    #     "Compute the sum over dimension `dim`"
    #     return self.backend.Sum.apply(self, dim)

    # def mean(self, dim=None):
    #     "Compute the mean over dimension `dim`"
    #     return self.backend.Mean.apply(self, dim)

    # def permute(self, *order):
    #     "Permute tensor dimensions to *order"
    #     return self.backend.Permute.apply(self, order)

    # def view(self, *shape):
    #     "Change the shape of the tensor to a new shape with the same size"
    #     return self.backend.View.apply(self, shape)

    def contiguous(self):
        "Return a contiguous tensor with the same data"
        return self.backend.Copy.apply(self)

    def __repr__(self):
        return repr(self._tensor)

    def __getitem__(self, key):
        return self._tensor.get(key)

    def __setitem__(self, key, val):
        self._tensor.set(key, val)

    @property
    def grad(self):
        return self.derivative

    # Internal methods used for autodiff.
    def _type_(self, precision):
        Tensor.FLOAT = precision
        self._tensor.astype(Tensor.FLOAT)

    def _new(self, tensor_data):
        return Tensor(tensor_data)

    @staticmethod
    def make(storage, shape):
        "Create a new tensor from data"
        return Tensor(np.array(storage, dtype=Tensor.FLOAT).reshape(shape))

    def reshape_grad(self, other):
        "Method used to allow for backprop over reduce."
        if self.shape == other.shape:
            return other
        shape = np.broadcast_shapes(self.shape, other.shape)
        buf = Tensor(np.broadcast_to(other._tensor, shape))
        if self.shape == shape:
            return buf

        # batch first
        buf2 = Tensor(np.sum(buf._tensor, axis=0))
        return buf2

    def zeros(self, shape=None):
        def zero(shape):
            return Tensor.make(
                [0] * np.prod(shape), shape
            )

        if shape is None:
            out = zero(self.shape)
        else:
            out = zero(shape)
        out._type_(Tensor.FLOAT)
        return out

    def tuple(self):
        return (self._tensor.flatten().tolist(), self.shape)

    def get_data(self):
        return Tensor(self._tensor)

    def get_storage(self) -> DATA_TYPE:
        return self._tensor

    def backward(self, grad_output=None):
        if grad_output is None:
            assert self.shape == (1,), "Must provide grad_output if non-scalar"
            grad_output = Tensor.make([1.0], (1,))
        super().backward(grad_output)
