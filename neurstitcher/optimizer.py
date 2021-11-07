from abc import ABC, abstractmethod
from typing import List

from .autodiff import Parameter
from .tensor import Tensor


class Optimizer(ABC):
    """
    Base class for implementing optimize algorithm.
    """
    data_type = Tensor

    def __init__(self, parameters: List[Parameter]) -> None:
        self.parameters = parameters

    def zero_grad(self):
        for p in self.parameters:
            p.value.zero_grad_()

    def step(self):
        for p in self.parameters:
            assert p.value.grad is not None, "The gradient of parameter {p} is None."
            _para = self.optimize_step(p.value.get_storage(), p.value.grad.get_storage())
            p.update(Tensor(_para))

    @abstractmethod
    def optimize_step(self, para, grad):
        pass


class MiniBatchSGD(Optimizer):
    def __init__(self, parameters: List[Parameter], lr: float) -> None:
        super().__init__(parameters)
        self.lr = lr

    def optimize_step(self, para, grad):
        return para - self.lr * grad
