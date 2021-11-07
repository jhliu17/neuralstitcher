from .utils import rand
from . import function as F
from .autodiff import Parameter


class Module:
    """
    Attributes:
        _modules (dict of name x :class:`Module`): Storage of the child modules
        _parameters (dict of name x :class:`Parameter`): Storage of the module's parameters
        mode (string): Mode of operation, can be {"train", "eval"}.

    """

    def __init__(self):
        self._modules = {}
        self._parameters = {}
        self.mode = "train"

    def modules(self):
        "Return the child modules of this module."
        return self.__dict__["_modules"].values()

    def train(self):
        "Set the mode of this module and all descendent modules to `train`."
        self.__set_mode('train')

    def eval(self):
        "Set the mode of this module and all descendent modules to `eval`."
        self.__set_mode('eval')

    def __set_mode(self, mode):
        """Set the module and its descendent with mode

        Args:
            mode (str): the setting mode ('train' or 'eval')
        """
        # set self mode
        self.mode = mode

        # set descendent
        for m in self._modules.values():
            m.__set_mode(mode)

    def named_parameters(self):
        """
        Collect all the parameters of this module and its descendents.


        Returns:
            dict: Each name (key) and :class:`Parameter` (value) under this module.
        """
        named_dict = {}

        # add self parameters
        named_dict.update(self._parameters)

        # add descendent parematers
        for k, m in self._modules.items():
            m_named_parameters = m.named_parameters()
            for name, paras in m_named_parameters.items():
                named_dict['%s.%s' % (k, name)] = paras

        return named_dict

    def parameters(self):
        return self.named_parameters().values()

    def add_parameter(self, k, v):
        """
        Manually add a parameter. Useful helper for scalar parameters.

        Args:
            k (str): Local name of the parameter.
            v (value): Value for the parameter.

        Returns:
            Parameter: Newly created parameter.
        """
        val = Parameter(v)
        self.__dict__["_parameters"][k] = val
        return val

    def __setattr__(self, key, val):
        if isinstance(val, Parameter):
            self.__dict__["_parameters"][key] = val
        elif isinstance(val, Module):
            self.__dict__["_modules"][key] = val
        else:
            super().__setattr__(key, val)

    def __getattr__(self, key):
        if key in self.__dict__["_parameters"]:
            return self.__dict__["_parameters"][key]

        if key in self.__dict__["_modules"]:
            return self.__dict__["_modules"][key]

        return self.__getattribute__(key)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self):
        assert False, "Not Implemented"

    def __repr__(self):
        def _addindent(s_, numSpaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(numSpaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        child_lines = []

        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = child_lines

        main_str = self.__class__.__name__ + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str


class LinearLayer(Module):
    def __init__(self, input_dim: int, output_dim: int, bias: bool = True):
        super().__init__()
        self.W = Parameter(rand((input_dim, output_dim)))

        self.bias = bias
        if bias:
            self.b = Parameter(rand((1, output_dim)))

    def forward(self, x):
        if self.bias:
            return F.Add.apply(F.LinearLayerMatMul.apply(x, self.W), self.b)
        else:
            return F.LinearLayerMatMul.apply(x, self.W)


class Sigmod(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.Sigmoid.apply(x)


class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.ReLU.apply(x)


class Tanh(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.Tanh.apply(x)
