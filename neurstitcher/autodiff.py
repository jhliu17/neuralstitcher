import uuid

from queue import Queue
from abc import ABC, abstractstaticmethod


def wrap_tuple(x):
    if isinstance(x, tuple):
        return x
    return (x,)


def unwrap_tuple(x):
    if len(x) == 1:
        return x[0]
    return x


def unwrap_parameter(x):
    if isinstance(x, Parameter):
        return x.value
    return x


class Variable:
    """
    An abstract class for gradient tape variable class.
    The core mathematical operation must be implemented by the children class!

    Attributes:
        history (:class:`History`) : the Function calls that created this variable or None if constant
        derivative (number): the derivative with respect to this variable
        name (string) : an optional name for debugging
    """

    def __init__(self, history, name=None):
        assert history is None or isinstance(history, History), history

        self.history = history
        self._derivative = None

        # For debugging can have a name.
        if name is not None:
            self.name = name
        else:
            self.name = str(uuid.uuid4())

    def requires_grad_(self, val=True):
        if val:
            self.history = History(None, None, None)
            self.zero_grad_()

    def backward(self, d_output=None):
        """
        Calls autodiff to fill in the derivatives for the history of this object.
        """
        if d_output is None:
            d_output = 1.0
        backpropagate(VariableWithDeriv(self, d_output))

    @property
    def derivative(self):
        return self._derivative

    def __hash__(self):
        return hash(self._name)

    def _add_deriv(self, val):
        assert self.history.is_leaf(), "Only leaf variables can have derivatives."
        if self._derivative is None:
            self._derivative = self.zeros()
        self._derivative += val

    def zero_grad_(self):
        self._derivative = self.zeros()

    def zeros(self):
        return 0.0

    def reshape_grad(self, x):
        """
        Make batchify gradient.
        """
        return x


class Context:
    """
    Context class is used by.
    """

    def __init__(self, no_grad=False):
        self._saved_values = None
        self.no_grad = no_grad

    def save_for_backward(self, *values):
        if self.no_grad:
            return
        self._saved_values = values

    @property
    def saved_values(self):
        assert not self.no_grad, "Doesn't require grad"
        assert self._saved_values is not None, "Did you forget to save values?"
        return unwrap_tuple(self._saved_values)


class History:
    """
    `History` stores all of the `Function` operations that were used to
    construct an autodiff object.

    Attributes:
        last_fn (:class:`FunctionBase`) : The last function that was called.
        ctx (:class:`Context`): The context for that function.
        inputs (list of inputs) : The inputs that were given when `last_fn.forward` was called.
    """

    def __init__(self, last_fn=None, ctx=None, inputs=None):
        self.last_fn = last_fn
        self.ctx = ctx
        self.inputs = inputs

    def is_leaf(self):
        return self.last_fn is None

    def backprop_step(self, d_output):
        return self.last_fn.chain_rule(self.ctx, self.inputs, d_output)


class VariableWithDeriv:
    "Holder for a variable with it derivative."

    def __init__(self, variable, deriv):
        self.variable = variable
        self.deriv = variable.reshape_grad(deriv)


class Parameter:
    """
    A Parameter is a special container stored in a :class:`Module`.

    It is designed to hold a :class:`Variable`, but we allow it to hold
    any value for testing.
    """

    def __init__(self, x=None):
        self.value: Variable = x
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)

    def update(self, x):
        "Update the parameter value."
        self.value = x
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)

    def __repr__(self):
        return repr(self.value)


class FunctionBase(ABC):
    """
    A function that can act on :class:`Variable` arguments to
    produce a :class:`Variable` output, while tracking the internal history.

    Call by :func:`FunctionBase.apply`.

    """
    tape_grad: bool = True

    @abstractstaticmethod
    def variable(raw, history):
        pass

    @abstractstaticmethod
    def data(val):
        pass

    @classmethod
    def unwrap_variable(cls, *args):
        return tuple([cls.data(v) if v is not None else None for v in args])

    @classmethod
    def wrap_variable(cls, *args):
        return tuple([cls.variable(v, None) if v is not None else None for v in args])

    @classmethod
    def _forward(cls, ctx, *args):
        """
        Unwrap the computing data and feed them to forward.
        """
        data = cls.unwrap_variable(*args)
        res = wrap_tuple(cls.forward(ctx, *data))
        var = cls.wrap_variable(*res)
        return var[0]

    @classmethod
    def _backward(cls, ctx, grad_output):
        """
        Unwrap the gradient data and feed them to backward.
        """
        grad = cls.unwrap_variable(grad_output)
        res = wrap_tuple(cls.backward(ctx, grad[0]))
        var = cls.wrap_variable(*res)
        return var

    @classmethod
    def apply(cls, *vals):
        raw_vals = []
        need_grad = False

        # first unwrap module paremeter
        vals = [unwrap_parameter(v) for v in vals]

        # whether any input variables need gradient
        for v in vals:
            if isinstance(v, Variable):
                if v.history is not None:
                    need_grad = True
                raw_vals.append(v.get_data())
            else:
                raw_vals.append(v)

        # forward
        ctx = Context(not need_grad)
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, cls.data_type), "Expected return typ %s got %s" % (
            cls.data_type,
            type(c)
        )
        back = None

        # save backward ctx
        if need_grad and cls.tape_grad:
            back = History(cls, ctx, vals)
        return cls.variable(cls.data(c), back)

    @classmethod
    def chain_rule(cls, ctx, inputs, d_output):
        """
        Implement the derivative chain-rule.

        Args:
            cls (:class:`FunctionBase`): The Function
            ctx (:class:`Context`) : The context from running forward
            inputs (list of args) : The args that were passed to :func:`FunctionBase.apply` (e.g. :math:`x, y`)
            d_output (number) : The `d_output` value in the chain rule.

        Returns:
            list of :class:`VariableWithDeriv`: A list of non-constant variables with their derivatives
            (see `is_constant` to remove unneeded variables)

        """
        all_derivatives = list(wrap_tuple(cls._backward(ctx, d_output)))
        needed_derivatives = []

        for var, der in zip(inputs, all_derivatives):
            if not is_constant(var):
                if der is None:
                    raise Exception(
                        f"Invalid gradient for variable {var.name} presents at backward pass.")
                needed_derivatives.append(VariableWithDeriv(var, der))

        return needed_derivatives

    @classmethod
    def is_tape_enabled(cls):
        return cls.tape_grad

    @classmethod
    def set_tape_enabled(cls, val=True):
        cls.tape_grad = val


def is_leaf(val):
    return isinstance(val, Variable) and val.history.is_leaf()


def is_constant(val):
    return not isinstance(val, Variable) or val.history is None


def backpropagate(final_variable_with_deriv):
    """
    Runs a breadth-first search on the computation graph in order to
    backpropagate derivatives to the leaves.

    See :doc:`backpropagate` for details on the algorithm

    Args:
       final_variable_with_deriv (:class:`VariableWithDeriv`): The final variable
           and its derivative that we want to propagate backward to the leaves.
    """
    back_queue = Queue()
    back_queue.put(final_variable_with_deriv)

    while not back_queue.empty():
        var_with_der: VariableWithDeriv = back_queue.get()
        var: Variable = var_with_der.variable
        der = var_with_der.deriv

        if is_leaf(var):
            var._add_deriv(der)
        else:
            history: History = var.history
            fn: FunctionBase = history.last_fn
            inputs = history.inputs
            ctx = history.ctx
            back_var_with_der = fn.chain_rule(ctx, inputs, der)

            # add to queue
            for i in back_var_with_der:
                back_queue.put(i)


class no_grad:
    def __enter__(self):
        self.prev = FunctionBase.is_tape_enabled()
        FunctionBase.set_tape_enabled(False)

    def __exit__(self):
        FunctionBase.set_tape_enabled(self.prev)
