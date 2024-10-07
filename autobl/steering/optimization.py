from typing import Callable, Optional
import warnings

import torch
from torch import Tensor
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.optim.optimize import *
from botorch.generation.gen import *


class Optimizer:
    """
    Acquisition function optimizer. ExperimentGuide object automatically fills bounds and num_candidates.
    """
    def __init__(
            self,
            *args,
            bounds: Optional[Tensor] = None,
            num_candidates: int = 1,
            **kwargs
    ) -> None:
        self.bounds = bounds
        self.num_candidates = num_candidates
        self.measured_points = torch.tensor([], requires_grad=False)
        self.duplicate_distance_threshold = self.get_duplicate_threshold()

    def maximize(self, acquisition_function: AcquisitionFunction):
        pass

    def get_required_params(self):
        return []

    def get_duplicate_threshold(self):
        assert self.bounds is not None
        perc = 1e-4
        duplicate_distance_threshold = (self.bounds[1] - self.bounds[0]) * perc
        duplicate_distance_threshold[torch.isinf(duplicate_distance_threshold)] = 0
        return duplicate_distance_threshold

    def update_sampled_points(self, pts):
        self.measured_points = torch.cat([self.measured_points, pts], dim=0)

    def find_duplicate_point_mask(self, pts):
        """
        Returns a mask that has the shape of `pts.shape[:-1]`, where False elements represent points that have aLready
        been measured in the past and should not be measured again.

        :param pts: Tensor. Tensor of suggested points with shape `[n, q, d]`.
        :return: Tensor. A mask indicating non-duplicating points.
        """
        if len(self.measured_points) == 0:
            return torch.ones(pts.shape[:-1]).bool()
        diff_mat = torch.abs(pts[..., None] - self.measured_points)
        diff_mat = (diff_mat < self.duplicate_distance_threshold).int()
        # The first sum is over feaature dim and the second is over measured_points.
        diff_mat = diff_mat.sum(-1).sum(-1).bool()
        mask = ~diff_mat
        if torch.count_nonzero(mask) == 0:
            warnings.warn('All suggested points have been measured in the past! ({})'.format(pts))
            mask = torch.ones(pts.shape[:-1]).bool()
        return mask


class ContinuousOptimizer(Optimizer):
    """
    Wrapper of BoTorch optimization functions that use continuous optimization (e.g., not based on
    optimize_acqf_discrete).
    """
    default_params = {
            "num_restarts": 5,
            "raw_samples": 10
    }

    required_params = ['bounds', 'q', 'num_restarts']

    def __init__(
            self,
            *args,
            bounds: Optional[Tensor] = None,
            num_candidates: int = 1,
            optim_func: Callable = optimize_acqf,
            optim_func_params: Optional[dict] = None,
            **kwargs
    ) -> None:
        """
        The constructor.

        :param bounds: Optional[Tensor[Tensor[float, ...], Tensor[float, ...]]]. The lower and upper bound of the
                       search space.
        :param num_candidates: int. The number of candidates to suggest. When working with analytical acquisition
                               functions, this must be 1.
        :param optim_func: Callable. An optimization function from botorch.optim.optimize, such as optimize_acqf.
        :param optim_func_params: Optional[dict]. The parameters of the BoTorch optimization function other than
                                  `acq_function`, `bounds`, and `q`.
        """
        super().__init__(*args, bounds=bounds, num_candidates=num_candidates, **kwargs)
        self.optim_func = optim_func
        self.optim_func_params = optim_func_params
        if self.optim_func_params is None:
            self.optim_func_params = {}

    def get_argument_dict(self):
        arg_dict = {**self.optim_func_params}
        for arg in self.default_params.keys():
            if arg not in arg_dict.keys():
                arg_dict[arg] = self.default_params[arg]

        if 'bounds' in self.required_params and 'bounds' not in arg_dict.keys():
            arg_dict['bounds'] = self.bounds
        if 'q' in self.required_params and 'q' not in arg_dict.keys():
            arg_dict['q'] = self.num_candidates

        for arg in self.required_params:
            if arg not in arg_dict.keys():
                raise ValueError("{} is required by {}, but is not provided.".format(arg, self.optim_func.__name__))
        return arg_dict

    def maximize(self, acquisition_function: AcquisitionFunction):
        """
        Maximize the acquisition function. Returns the points of optima, and the corresponding acquisition
        function value. The returned points have shape `[num_candidates = q, d]`. Regardless of `q`, the returned
        acquisition value is always a scalar that gives the highest acquisition value among the `q` points.

        The specified BoTorch optimization function is used to get the optima. By default, num_restarts is set to
        a number greater than 1, so that the function returns a Tensor of [num_restarts, q, d]. Then, points
        that have been measured in the past are identified, and the q-batches containing such points are excluded.
        The suggested point(s) are then the q-batch that has the largest acquisition value among what remain.
        The stored list of measured points is updated with the selected point(s).

        :param acquisition_function: AcquisitionFunction. The acquisition function to optimize.
        :return: Tensor, Tensor[float]. The locations and value of the optima.
        """
        arg_dict = self.get_argument_dict()
        # Returns optimal points in [num_restarts, q = num_candidates, d] and their
        # acquisition values in [num_restarts] (there is no q dimension).
        pts, acq_vals = self.optim_func(acquisition_function, return_best_only=False, **arg_dict)
        # nonduplicating_mask.shape = [num_restarts, q].
        nonduplicating_mask = self.find_duplicate_point_mask(pts)

        # Find the q-batch that has the highest acquisition value after exlcuding those containing already measured
        # points.
        q_selection_mask = nonduplicating_mask.int().sum(-1).bool()
        acq_vals[~q_selection_mask] = -torch.inf
        restart_ind = torch.argmax(acq_vals)
        pts = pts[restart_ind]
        acq_vals = acq_vals[restart_ind]

        self.update_sampled_points(pts)
        return pts, acq_vals

    def get_required_params(self):
        return self.required_params


class DiscreteOptimizer(Optimizer):

    default_params = {}

    required_params = ['q']

    def __init__(
            self,
            *args,
            bounds: Optional[Tensor] = None,
            num_candidates: int = 1,
            optim_func: Callable = optimize_acqf,
            optim_func_params: Optional[dict] = None,
            **kwargs
    ) -> None:
        """
        The constructor.

        :param bounds: Optional[Tensor[Tensor[float, ...], Tensor[float, ...]]]. The lower and upper bound of the
                       search space.
        :param num_candidates: int. The number of candidates to suggest. When working with analytical acquisition
                               functions, this must be 1.
        :param optim_func: Callable. An optimization function from botorch.optim.optimize, such as optimize_acqf.
        :param optim_func_params: Optional[dict]. The parameters of the BoTorch optimization function other than
                                  `acq_function`, `bounds`, and `q`.
        """
        super().__init__(*args, bounds=bounds, num_candidates=num_candidates, **kwargs)
        self.optim_func = optim_func
        self.optim_func_params = optim_func_params
        if self.optim_func_params is None:
            self.optim_func_params = {}

    def get_argument_dict(self):
        arg_dict = {**self.optim_func_params}
        for arg in self.default_params.keys():
            if arg not in arg_dict.keys():
                arg_dict[arg] = self.default_params[arg]

        if 'bounds' in self.required_params and 'bounds' not in arg_dict.keys():
            arg_dict['bounds'] = self.bounds
        if 'q' in self.required_params and 'q' not in arg_dict.keys():
            arg_dict['q'] = self.num_candidates

        for arg in self.required_params:
            if arg not in arg_dict.keys():
                raise ValueError("{} is required by {}, but is not provided.".format(arg, self.optim_func.__name__))
        return arg_dict

    def maximize(self, acquisition_function: AcquisitionFunction):
        """
        Maximize the acquisition function. Returns the points of optima, and the corresponding acquisition
        function value. The returned points have shape `[num_candidates = q, d]`. Regardless of `q`, the returned
        acquisition value is always a scalar that gives the highest acquisition value among the `q` points.

        The specified BoTorch optimization function is used to get the optima. By default, num_restarts is set to
        a number greater than 1, so that the function returns a Tensor of [num_restarts, q, d]. Then, points
        that have been measured in the past are identified, and the q-batches containing such points are excluded.
        The suggested point(s) are then the q-batch that has the largest acquisition value among what remain.
        The stored list of measured points is updated with the selected point(s).

        :param acquisition_function: AcquisitionFunction. The acquisition function to optimize.
        :return: Tensor, Tensor[float]. The locations and value of the optima.
        """
        arg_dict = self.get_argument_dict()
        # If using optimize_discrete, the shape of returned points will be [q, d].
        pts, acq_vals = self.optim_func(acquisition_function, return_best_only=False, **arg_dict)
        pts = pts[None, ...]
        if acq_vals.ndim == 0:
            acq_vals = torch.tensor([[acq_vals]])
        elif acq_vals.ndim == 1:
            acq_vals = acq_vals[None, ...]
        # nonduplicating_mask.shape = [num_restarts, q].
        nonduplicating_mask = self.find_duplicate_point_mask(pts)

        # Find the q-batch that has the highest acquisition value after exlcuding those containing already measured
        # points.
        q_selection_mask = nonduplicating_mask.int().sum(-1).bool()
        acq_vals[~q_selection_mask] = -torch.inf
        restart_ind = torch.argmax(acq_vals)
        pts = pts[restart_ind]
        acq_vals = acq_vals[restart_ind]

        self.update_sampled_points(pts)
        return pts, acq_vals

    def get_required_params(self):
        return self.required_params


class TorchOptimizer(Optimizer):

    default_params = {
            "num_restarts": 5,
            "raw_samples": 10
    }

    def __init__(self,
                 *args,
                 bounds: Optional[Tensor] = None,
                 num_candidates: int = 1,
                 torch_optimizer: torch.optim.Optimizer = torch.optim.Adam,
                 torch_optimizer_options: Optional[dict] = None,
                 **kwargs):
        super().__init__(bounds=bounds, num_candidates=num_candidates)
        self.optimizer_class = torch_optimizer
        self.optimizer_options = torch_optimizer_options
        if self.optimizer_options is None:
            self.optimizer_options = {}
        self.kwargs = kwargs

    def get_argument_dict(self):
        arg_dict = {**self.kwargs}
        for arg in self.default_params.keys():
            if arg not in arg_dict.keys():
                arg_dict[arg] = self.default_params[arg]
        return arg_dict

    def maximize(self, acquisition_function: AcquisitionFunction):
        arg_dict = self.get_argument_dict()
        batch_initial_conditions = gen_batch_initial_conditions(
            acq_function=acquisition_function,
            bounds=self.bounds,
            q=self.num_candidates,
            **arg_dict
        )
        # Returns optimal points in [num_restarts, q = num_candidates, d] and their
        # acquisition values in [num_restarts] (there is no q dimension).
        pts, acq_vals = gen_candidates_torch(
            initial_conditions=batch_initial_conditions,
            acquisition_function=acquisition_function,
            lower_bounds=self.bounds[0],
            upper_bounds=self.bounds[1],
            optimizer=self.optimizer_class,
            options=self.optimizer_options
        )

        # nonduplicating_mask.shape = [num_restarts, q].
        nonduplicating_mask = self.find_duplicate_point_mask(pts)

        # Find the q-batch that has the highest acquisition value after exlcuding those containing already measured
        # points.
        q_selection_mask = nonduplicating_mask.int().sum(-1).bool()
        acq_vals[~q_selection_mask] = -torch.inf
        restart_ind = torch.argmax(acq_vals)
        pts = pts[restart_ind]
        acq_vals = acq_vals[restart_ind]

        self.update_sampled_points(pts)
        return pts, acq_vals
