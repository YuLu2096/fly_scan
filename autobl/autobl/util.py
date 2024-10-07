import random

import numpy as np
try:
    import torch
    from torch import Tensor
except:
    print('Unable to import PyTorch.')


def to_tensor(x):
    if isinstance(x, np.ndarray):
        # If CUDA is available, convert array to tensor using torch.tensor, which honors the set default device.
        # Otherwise, use from_numpy which creates a reference to the data in memory instead of creating a copy.
        if torch.cuda.is_available():
            return torch.tensor(x)
        else:
            return torch.from_numpy(x)
    else:
        return x


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, (list, tuple)):
        return np.asarray(x)
    else:
        return x


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
    except:
        pass


def interp1d_tensor(x0, y0, x):
    """
    1D interpolation for tensors.

    :param x0: Tensor. 1D tensor giving the coordinates.
    :param y0: Tensor. 1D tensor giving the values of the array to be interpolated.
    :param x: Tensor. 1D tensor giving the locations where new values are to be caluclated.
    :return: Tensor.
    """
    ibins = torch.bucketize(x, x0)
    y_floor = y0[torch.clamp(ibins - 1, 0, len(y0) - 1)]
    y_ceil = y0[torch.clamp(ibins, 0, len(y0) - 1)]
    x_floor = x0[torch.clamp(ibins - 1, 0, len(y0) - 1)]
    x_ceil = x0[torch.clamp(ibins, 0, len(y0) - 1)]
    w = ((x - x_floor) / (x_ceil - x_floor + 1e-8)).clamp(0, 1)
    vals = y_ceil * w + y_floor * (1 - w)
    return vals


def elementwise_derivative(f, x, order=1):
    """
    Elementwise 1st or 2nd order derivative.

    :param f: Callable.
    :param x: Tensor.
    :param order: int. Can be 1 or 2.
    :return: Tensor if order == 1; (Tensor, Tensor) if order == 2, where the returned tensors are respectively
             the first order and second order derivatives.
    """
    def differentiate(x):
        jac = torch.autograd.functional.jacobian(f, x, create_graph=True)
        g = jac[torch.tensor(range(jac.shape[0])), torch.tensor(range(jac.shape[0]))]
        return g
    g = differentiate(x)
    if order == 1:
        return g
    if order == 2:
        jac2 = torch.autograd.functional.jacobian(differentiate, x, create_graph=True)
        gg = jac2[torch.tensor(range(jac2.shape[0])), torch.tensor(range(jac2.shape[0]))]
        return g, gg


def sigmoid(x, r=1.0, d=0.0):
    if isinstance(x, torch.Tensor):
        mod = torch
    else:
        mod = np
    exponent = mod.clip(-r * (x - d), None, 1e2)
    return 1.0 / (1.0 + mod.exp(exponent))

def gaussian(x, a, mu, sigma, c):
    if isinstance(x, torch.Tensor):
        return a * torch.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + c
    else:
        return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + c


def fit(f, data_x, data_y, init_params, n_iters=20, opt_class=torch.optim.Adam, opt_params=None):
    if opt_params is None:
        opt_params = {}
    params = init_params
    for p in params:
        p.requires_grad_(True)
    opt = opt_class(params=params, **opt_params)
    for i_iter in range(n_iters):
        y = f(data_x, *params)
        l = torch.mean((y - data_y) ** 2)
        l.backward(retain_graph=True)
        opt.step()
    for p in params:
        p.requires_grad_(False)
    return params


def rms(actual, true):
    return np.sqrt(np.mean((actual - true) ** 2))


def estimate_sparse_gradient(x, y):
    """
    Given a list of data sampled on a non-uniform grid, find the 2-point numerical gradient by first
    calculating the gradient at the mid-point of point i - 1 and i + 1, then linearly interpolate for
    the gradient at i.

    :param x: np.ndarray.
    :param y: np.ndarray.
    """
    grad_x = x[1:-1]
    grad_y_mid = (y[2:] - y[:-2]) / (x[2:] - x[:-2])
    grad_x_mid = (x[2:] + x[:-2]) / 2.0
    grad_y = np.interp(grad_x, grad_x_mid, grad_y_mid)
    return grad_x, grad_y
