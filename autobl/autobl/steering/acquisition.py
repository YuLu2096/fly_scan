import logging
from typing import Optional, Callable, Any, Tuple

import botorch
import matplotlib.pyplot as plt
import torch
from botorch.models.model import Model
from botorch.acquisition import *
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.transforms.input import Normalize
from botorch.utils import t_batch_mode_transform
from torch import Tensor

from autobl.util import *


class DummyAcquisition:
    
    def __init__(self, *args, **kwargs) -> None:
        pass
    
    def __getattr__(self, name):
        def dummy_func(*args, **kwargs):
            return None
        return dummy_func
    
    def forward(self, x: Tensor, *args, **kwargs):
        return torch.tensor(0.0, device=x.device)
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class PosteriorStandardDeviationDerivedAcquisition(PosteriorStandardDeviation):
    def __init__(
            self, model: Model,
            input_transform: Optional[Normalize] = None,
            posterior_transform: Optional[PosteriorTransform] = None,
            maximize: bool = True,
            beta: float = 0.95,
            gamma: float = 0.95,
            add_posterior_stddev: bool = True,
            estimate_posterior_mean_by_interpolation: bool = False,
            guide_obj=None,
            debug=False
    ) -> None:
        """
        The constructor.

        :param beta: float. Decay factor of the weights of add-on terms in the acquisition function.
        :param gamma: float. Decay factor of the mixing coefficient between the weighting function and the original
            value.
        :param input_transform: Optional[Normalize]. The transform object used for normalizing x.
        :param add_posterior_stddev: bool. If False, the posterior standard deviation will not be added
            to the returned value, and the acquisition function will be solely
            contributed by fitting residue.
        :param guide_obj: Optional[GPExperimentGuide]. If posterior mean is to be estimated from spline interpolation,
            this argument is required.
        """
        super().__init__(model, posterior_transform, maximize)
        self.input_transform = input_transform
        self.add_posterior_stddev = add_posterior_stddev
        self.debug = debug
        self.weight_func = None
        self.alpha = 1.0
        self.phi = 0
        self.beta = beta
        self.gamma = gamma
        self.intermediate_data = {}
        self.guide_obj = guide_obj
        self.estimate_posterior_mean_by_interpolation = estimate_posterior_mean_by_interpolation

    def set_weight_func(self, f: Callable):
        self.weight_func = f

    def apply_weight_func(self, x, a):
        if self.weight_func is not None:
            m = self.weight_func(x).squeeze(-2).squeeze(-1)
            a = (self.alpha * m + 1 - self.alpha) * a
        return a

    def update_hyperparams_following_schedule(self):
        self.phi = self.phi * self.beta
        self.alpha = self.alpha * self.gamma

    def _mean_and_sigma(
        self, X: Tensor, compute_sigma: bool = True, min_var: float = 1e-12
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if not self.estimate_posterior_mean_by_interpolation:
            return super()._mean_and_sigma(X, compute_sigma, min_var)
        else:
            if compute_sigma:
                _, sigma = super()._mean_and_sigma(X, True, min_var)
            else:
                sigma = None
            mu, _ = self.guide_obj.get_posterior_mean_and_std(X, transform=False, untransform=True, compute_sigma=False)
            mu = mu.squeeze(-1)
            return mu, sigma


class FittingResiduePosteriorStandardDeviation(PosteriorStandardDeviationDerivedAcquisition):
    r"""Posterior standard deviation enhanced by fitting residue.
    The current posterior mean is linearly fit using a series of reference spectra, and
    the fitting residue is combined with the posterior mean to give the acquisiion function's
    value.
    """
    def __init__(
            self, model: Model,
            input_transform: Optional[Normalize] = None,
            posterior_transform: Optional[PosteriorTransform] = None,
            maximize: bool = True,
            beta: float = 0.99,
            gamma: float = 0.95,
            reference_spectra_x: Tensor = None,
            reference_spectra_y: Tensor = None,
            phi: float = 0.1,
            add_posterior_stddev: bool = True,
            estimate_posterior_mean_by_interpolation: bool = False,
            guide_obj: Any = None,
            debug: bool = False
    ) -> None:
        """
        The constructor.

        :param reference_spectra_x: Tensor. A Tensor with shape [m,] containing the coordinates of the
            n refernce spectra.
        :param reference_spectra_y: Tensor. A Tensor with shape [n, m] containing the values of the
            n refernce spectra.
        :param phi: Optional[float]. Weight of the residue term.
        :param add_posterior_stddev: bool. If False, the posterior standard deviation will not be added
            to the returned value, and the acquisition function will be solely
            contributed by fitting residue.
        """
        super().__init__(model, input_transform, posterior_transform, maximize, beta, gamma,
                         add_posterior_stddev, estimate_posterior_mean_by_interpolation, guide_obj, debug)
        self.reference_spectra_x = self.input_transform.transform(reference_spectra_x.reshape(-1, 1)).squeeze()
        self.reference_spectra_y = reference_spectra_y
        self.phi = phi
        if self.phi is None:
            self.estimate_weights()

    def estimate_weights(self):
        self.phi = 1.0

        x = torch.linspace(0, 1, 100).view([-1, 1, 1])
        add_posterior_stddev_orig = self.add_posterior_stddev
        self.add_posterior_stddev = True
        a = self.forward(x)
        self.add_posterior_stddev = add_posterior_stddev_orig

        sigma = self.intermediate_data['sigma']
        r = self.intermediate_data['r']
        self.phi = sigma.max() / r.max() * 5.0
        logging.info('Automatically determined fitting residue weights: phi = {}.'.format(self.phi))

    @t_batch_mode_transform()
    def forward(self, x: Tensor, sigma_x=None, **kwargs) -> Tensor:
        mu, _ = self._mean_and_sigma(self.reference_spectra_x, compute_sigma=False)
        amat = self.reference_spectra_y.T
        bvec = mu.reshape(-1, 1)
        xvec = torch.matmul(torch.linalg.pinv(amat), bvec)
        y_fit = torch.matmul(amat, xvec).view(-1)
        res = (y_fit - mu) ** 2

        if self.add_posterior_stddev:
            if sigma_x is None:
                _, sigma = self._mean_and_sigma(x)
            else:
                sigma = sigma_x
        else:
            sigma = 0
        r = interp1d_tensor(self.reference_spectra_x.squeeze(),
                            res,
                            x.squeeze())
        a = sigma + self.phi * r
        a = self.apply_weight_func(x, a)

        self.intermediate_data = {'sigma': sigma, 'r': r}

        if self.debug:
            self._plot_acquisition_values(locals())
        return a

    def _plot_acquisition_values(self, func_locals):
        x = func_locals['x']
        if len(x) > 100:
            mu = func_locals['mu']
            y_fit = func_locals['y_fit']
            res = func_locals['res']
            fig, ax = plt.subplots(1, 1)
            ax.plot(to_numpy(self.reference_spectra_x.squeeze()), to_numpy(mu.squeeze()), label='mu')
            ax.plot(to_numpy(self.reference_spectra_x.squeeze()), to_numpy(y_fit.squeeze()), label='fit')
            ax.plot(to_numpy(self.reference_spectra_x.squeeze()), to_numpy(self.reference_spectra_y[0].squeeze()),
                    label='ref1')
            ax.plot(to_numpy(self.reference_spectra_x.squeeze()), to_numpy(self.reference_spectra_y[1].squeeze()),
                    label='ref2')
            ax2 = ax.twinx()
            ax2.plot(to_numpy(self.reference_spectra_x.squeeze()), to_numpy(res.squeeze()), color='black',
                     label='residue', alpha=0.4)
            ax.legend(loc='upper left')
            ax2.legend(loc='center left')
            plt.show()


class GradientAwarePosteriorStandardDeviation(PosteriorStandardDeviationDerivedAcquisition):
    r"""Gradient-aware posterior standard deviation.
    The acquisition function is given as sigma + phi * grad, where grad
    is the norm of the gradient of the posterior mean along given axes, and
    phi is the weight.
    """
    def __init__(
            self, model: Model,
            input_transform: Optional[Normalize] = None,
            posterior_transform: Optional[PosteriorTransform] = None,
            maximize: bool = True,
            beta: float = 0.99,
            gamma: float = 0.95,
            gradient_dims: Optional[list[int, ...]] = None,
            phi: Optional[float] = 0.1,
            phi2: Optional[float] = 0.001,
            method: str = 'analytical',
            order: int = 1,
            finite_difference_interval: float = 1e-2,
            add_posterior_stddev: bool = True,
            estimate_posterior_mean_by_interpolation: bool = False,
            guide_obj: Any = None,
            debug: bool = False
    ) -> None:
        """
        The constructor.

        :param gradient_dims: Optional[list[int, ...]]. The dimensions along which the magnitude of gradient should
            be computed. If None, it will be assumed to be the last dimension.
        :param phi: float. The weight of the gradient term. If None, it will be automatically determined.
        :param phi2: float. The weight of the second or higher-order derivative term. Disregarded if `order == 1`.
        :param method: str. Can be "analytical" or "numerical". If "analytical", gradients are calculated using
            automatic differentiation. Due to the limitation of BoTorch, only first-order derivative is
            available using this method. If "numerical", gradients are calculated using finite difference
            based on the evaluations of the posterior mean on `numerical_gradient_points` points.
        :param order: int. The order of the derivative. If `method` is `"analytical"`, it can only be 1.
        :param finite_difference_interval: float. The interval used for finite-difference differentiation. Only used
            when `method` is `"numerical"`.  This value is specified in the normalized
            scale (between 0 and 1).
        :param add_posterior_stddev: bool. If True, posterior standard deviation is added to the function value.
        """
        super().__init__(model, input_transform, posterior_transform, maximize, beta, gamma,
                         add_posterior_stddev, estimate_posterior_mean_by_interpolation, guide_obj, debug)
        self.gradient_dims = gradient_dims
        self.phi = phi
        self.phi2 = phi2
        self.method = method
        self.order = order
        self.finite_difference_interval = finite_difference_interval
        self.add_posterior_stddev = add_posterior_stddev
        if method == 'analytical' and order > 1:
            raise ValueError("When method is 'analytical', order can only be 1.")
        if method not in ['analytical', 'numerical']:
            raise ValueError("'method' can only be 'analytical' or 'numerical'.")
        if self.phi is None or self.phi2 is None:
            self.estimate_weights()

    def update_hyperparams_following_schedule(self):
        self.phi = self.phi * self.beta
        self.phi2 = self.phi2 * self.beta
        self.alpha = self.alpha * self.gamma

    def estimate_weights(self):
        self.phi = 1.0
        self.phi2 = 1.0

        x = torch.linspace(0, 1, 100).view([-1, 1, 1])
        add_posterior_stddev_orig = self.add_posterior_stddev
        self.add_posterior_stddev = True
        a = self.forward(x)
        self.add_posterior_stddev = add_posterior_stddev_orig

        sigma = self.intermediate_data['sigma']
        gradients_all_orders = self.intermediate_data['gradients_all_orders']
        self.phi = sigma.max() / gradients_all_orders[0].max() * 0.5
        if len(gradients_all_orders) > 1:
            self.phi2 = sigma.max() / gradients_all_orders[1].max() * 0.5
        logging.info('Automatically determined gradient weights: phi = {}, phi2 = {}.'.format(self.phi, self.phi2))

    @t_batch_mode_transform()
    def forward(self, x: Tensor, mu_x=None, sigma_x=None) -> Tensor:
        if mu_x is None or sigma_x is None:
            mu, sigma = self._mean_and_sigma(x)
        else:
            mu, sigma = mu_x, sigma_x
        gradients_all_orders = [0.0] * self.order
        if self.method == 'analytical':
            g = self.calculate_gradients_analytical(x)
            gradients_all_orders[0] = torch.linalg.norm(g, dim=-1)
        elif self.method == 'numerical':
            for ord in range(1, self.order + 1):
                g = self.calculate_gradients_numerical(x, order=ord)
                g = torch.linalg.norm(g, dim=-1)
                gradients_all_orders[ord - 1] = g
        else:
            raise ValueError
        if not self.add_posterior_stddev:
            sigma = 0
        a = sigma + self.phi * gradients_all_orders[0]
        if len(gradients_all_orders) > 1:
            for gg in gradients_all_orders[1:]:
                a = a + self.phi2 * gg
        a = self.apply_weight_func(x, a)
        self.intermediate_data = {'sigma': sigma, 'gradients_all_orders': gradients_all_orders}
        return a

    def calculate_gradients_analytical(self, x: Tensor):
        """
        Calculate gradients using AD.

        :param x: Tensor.
        :return: Tensor. Gradient of shape (n, d).
        """
        f_posterior_mean = lambda x: self.model.posterior(x).mean.squeeze()
        with torch.enable_grad():
            jac = torch.autograd.functional.jacobian(f_posterior_mean, x)
            g = jac[torch.tensor(range(jac.shape[0])), torch.tensor(range(jac.shape[0]))]
        if self.gradient_dims is not None:
            g = torch.index_select(g, -1, self.gradient_dims)
        return g.squeeze(1)

    def calculate_gradients_numerical(self, x: Tensor, order: int = 1):
        def differentiate(x, h, f, order=1):
            if order == 1:
                return (f(x + h) - f(x - h)) / (2 * h)
            else:
                d = (differentiate(x + h, h, f, order=order - 1) - differentiate(x - h, h, f, order=order - 1))
                d = d / (2 * h)
                return d
        f = lambda x: self.model.posterior(x).mean
        g = []
        gradient_dims = self.gradient_dims
        if gradient_dims is None:
            gradient_dims = list(range(x.shape[-1]))
        for grad_dim in gradient_dims:
            h = torch.zeros(x.shape[-1])
            h[grad_dim] = self.finite_difference_interval
            gi = differentiate(x, h, f, order=order)
            g.append(gi)
        g = torch.cat(g, dim=-1)
        return g.squeeze(1)


class ComprehensiveAugmentedAcquisitionFunction(PosteriorStandardDeviationDerivedAcquisition):
    r"""Acquisition function that combines gradient and reference spectrum augmentations."""
    def __init__(
            self,
            model: Model,
            input_transform: Optional[Normalize] = None,
            posterior_transform: Optional[PosteriorTransform] = None,
            maximize: bool = True,
            beta: float = 0.999,
            gamma: float = 0.95,
            gradient_dims: Optional[list[int, ...]] = None,
            gradient_order: int = 1,
            differentiation_method: str = 'analytical',
            reference_spectra_x: Optional[Tensor] = None,
            reference_spectra_y: Optional[Tensor] = None,
            phi_g: float = 0.1,
            phi_g2: float = 0.001,
            phi_r: float = 100,
            addon_term_lower_bound: float = 1e-2,
            add_posterior_stddev: bool = True,
            estimate_posterior_mean_by_interpolation: bool = False,
            guide_obj: Any = None,
            debug: bool = False
    ) -> None:
        """
        The constructor.

        :param phi_g: float.
        :param phi_g2: float.
        :param phi_r: float.
        :param addon_term_lower_bound: float. The lower bound of add-on terms like gradient and fitting residue. The
            weighted sum of these terms is clipped to this value before multiplied with the posterior standard
            deviation. Choose this value carefully: a too large value turns the add-on terms to a constant and the
            acquisition function will behave like a simple posterior standard deviation, while a too small value
            prevents the algorithm from exploring regions with high uncertainty yet low add-on term values.
        """
        super().__init__(model, input_transform, posterior_transform, maximize, beta, gamma,
                         add_posterior_stddev, estimate_posterior_mean_by_interpolation, guide_obj, debug)
        self.acqf_g = GradientAwarePosteriorStandardDeviation(
            model,
            input_transform=input_transform,
            posterior_transform=posterior_transform,
            maximize=maximize,
            beta=beta,
            gamma=gamma,
            gradient_dims=gradient_dims,
            method=differentiation_method,
            order=gradient_order,
            phi=phi_g,
            phi2=phi_g2,
            add_posterior_stddev=False,
            estimate_posterior_mean_by_interpolation=estimate_posterior_mean_by_interpolation,
            guide_obj=guide_obj,
        )
        if reference_spectra_x is not None and reference_spectra_y is not None:
            self.acqf_r = FittingResiduePosteriorStandardDeviation(
                model,
                input_transform=input_transform,
                posterior_transform=posterior_transform,
                maximize=maximize,
                beta=beta,
                gamma=gamma,
                reference_spectra_x=reference_spectra_x,
                reference_spectra_y=reference_spectra_y,
                phi=phi_r,
                add_posterior_stddev=False,
                estimate_posterior_mean_by_interpolation=estimate_posterior_mean_by_interpolation,
                guide_obj=guide_obj,
                debug=debug
            )
            self.phi_r = self.acqf_r.phi
        else:
            logging.warning('No reference spectra provided. Using dummy acquisition function instead.')
            self.acqf_r = DummyAcquisition()
            self.phi_r = 0
        self.gradient_order = gradient_order
        self.phi_g = self.acqf_g.phi
        self.phi_g2 = self.acqf_g.phi2
        self.reference_spectra_x = reference_spectra_x
        self.reference_spectra_y = reference_spectra_y
        self.add_posterior_stddev = add_posterior_stddev
        self.addon_term_lower_bound = addon_term_lower_bound
        self.debug = debug

    def update_hyperparams_following_schedule(self):
        self.acqf_r.update_hyperparams_following_schedule()
        self.acqf_g.update_hyperparams_following_schedule()
        self.alpha = self.alpha * self.gamma

    @t_batch_mode_transform()
    def forward(self, x: Tensor) -> Tensor:
        mu, sigma = self._mean_and_sigma(x)
        if self.add_posterior_stddev:
            # a = torch.clip(sigma, 1e-3, None)#
            a = sigma - sigma.min()
        else:
            a = 1.0
        if self.phi_g > 0:
            a_g = self.acqf_g(x, mu_x=mu, sigma_x=sigma)
        else:
            a_g = torch.tensor(0.0, device=x.device)
        if self.phi_r > 0:
            a_r = self.acqf_r(x, mu_x=mu, sigma_x=sigma)
        else:
            a_r = torch.tensor(0.0, device=x.device)
        a = a * torch.clip(a_g + a_r, self.addon_term_lower_bound, None)
        a = self.apply_weight_func(x, a)
        if self.debug:
            self._plot_acquisition_values(locals())
        return a

    def _plot_acquisition_values(self, func_locals):
        mu = func_locals['mu']
        sigma = func_locals['sigma']
        a_g = func_locals['a_g']
        a_r = func_locals['a_r']
        a = func_locals['a']
        x = func_locals['x']
        x_squeezed = to_numpy(x.squeeze())
        if x.max() - x.min() > 0.99:
            fig, ax = plt.subplots(5, 1, figsize=(3, 12))
            ax[0].plot(x_squeezed, to_numpy(mu.squeeze()), label='mu')
            ax[0].fill_between(
                x_squeezed, to_numpy((mu - sigma).squeeze()), to_numpy((mu + sigma).squeeze()), alpha=0.5)
            ax[0].legend(bbox_to_anchor=(1, 0.5))
            ax[1].plot(x_squeezed, to_numpy(sigma.squeeze()), label='sigma')
            ax[1].legend(bbox_to_anchor=(1, 0.5))
            if self.phi_g > 0:
                ax[2].plot(x_squeezed, to_numpy(a_g.squeeze()), label='a_g')
            if self.phi_r > 0:
                ax[2].plot(x_squeezed, to_numpy(a_r.squeeze()), label='a_r')
            ax[2].legend(bbox_to_anchor=(1, 0.5))
            if self.phi_g > 0 or self.phi_r:
                ax[3].plot(x_squeezed, to_numpy((a_g + a_r).squeeze()), label='a_g + a_r')
                ax[3].legend(bbox_to_anchor=(1, 0.5))
            ax[4].plot(x_squeezed, to_numpy(a.squeeze()), label='a')
            ax[4].legend(bbox_to_anchor=(1, 0.5))
            if self.weight_func is not None:
                import matplotlib
                matplotlib.rc('font', family='Times New Roman')
                matplotlib.rcParams['font.size'] = 14
                matplotlib.rcParams['pdf.fonttype'] = 42
                fig, ax = plt.subplots(1, 1, figsize=(5, 4))
                ax.plot(self.input_transform.untransform(x).squeeze().detach(), mu.detach().squeeze(),
                        label='Posterior mean', color='gray', linestyle='--')
                ax.set_xlabel('Energy (eV)')
                ax.set_ylabel('Posterior mean')
                ax2 = ax.twinx()
                ax2.plot(self.input_transform.untransform(x).squeeze().detach(), to_numpy(self.weight_func(x.squeeze())),
                         label='Reweighting function')
                ax2.set_ylabel('Reweighting function')
                plt.legend(frameon=False)
                plt.tight_layout()
            plt.show()
