import logging

import botorch
import torch
import numpy as np
from gpytorch.distributions import MultivariateNormal
from torch import Tensor

from autobl.util import *


class ProjectedSpaceSingleTaskGP(botorch.models.SingleTaskGP):

    def __init__(self, *args, projection_function, **kwargs):
        super().__init__(*args, **kwargs)
        self.project_func = projection_function

    def set_projection_func(self, f):
        self.project_func = f

    def forward(self, x: Tensor) -> MultivariateNormal:
        if self.training:
            x = self.transform_inputs(x)
        x = self.project_func(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
