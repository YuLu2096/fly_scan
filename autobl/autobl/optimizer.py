import numpy as np
import torch


class GradientBasedOptimizer:

    def __init__(self, x_init, *args, **kwargs):
        self.x = np.array(x_init)


class LevenbergMarquadtOptimizer:

    def __init__(self, x_init, lmbda=1e-3, *args, **kwargs):
        super().__init__(x_init, *args, **kwargs)
        self.lmbda = lmbda

    def step(self, grad):
        raise NotImplementedError


class PyTorchOptimizer:

    def __init__(self, x_init, opt_class, opt_args, *args, **kwargs):
        assert issubclass(opt_class, torch.optim.Optimizer)
        assert isinstance(opt_args, dict)
        self.x = x_init
        self.x_tensor = torch.tensor(x_init)
        self.opt = opt_class([self.x_tensor], **opt_args)

    def step(self, grad):
        self.x_tensor.grad = torch.tensor(grad)
        self.opt.step()
        self.x = self.x_tensor.cpu().numpy()
