"""
Configuration dataclasses
"""

import dataclasses
import json
from collections.abc import Sequence  # , Callable
from typing import Any, Optional, Tuple, Type

import botorch
import numpy as np
# import gpytorch

from autobl.steering.optimization import ContinuousOptimizer, Optimizer


@dataclasses.dataclass
class Config:
    """
    Base class for configurations
    """

    random_seed: Any = 123
    """Random seed."""

    def dictionarize(self):
        """
        Transform configuration into a (nested) dictionary structure
        """
        d = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                d[key] = value.dictionarize()
            else:
                d[key] = value
        return d

    def to_json(self, fname):
        """
        Convert the dictionary version of the configuration to a .json file
        """
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(self.dictionarize(), f)


@dataclasses.dataclass
class StoppingCriterionConfig(Config):
    """
    Configuration that specifies the stopping criteria
    """

    method: str = "max_uncertainty"
    """
    Method of early stopping determination. Can be:
    - 'max_uncertainty': stops once max(acqf_weight_func * posterior_stdev) drops below a threshold.
        Parameters:
            - 'threshold': The threshold of max uncertainty.
    """

    params: Optional[dict] = dataclasses.field(default_factory=dict)
    """Parameters of the chosen method."""

    n_updates_to_begin: int = 10
    """Start early stopping checks after this number of model updates have been
    done."""

    n_check_interval: int = 5
    """Check stopping criterion after every this number of model updates."""


@dataclasses.dataclass
class ExperimentAnalyzerConfig(Config):
    """
    Configuration for how the experiment is analyzed
    """

    name: str = "Experiment"
    """Name of the experiment."""

    output_dir: str = "outputs"
    """Output directory of images and data."""

    n_plot_interval: int = 5
    """Generate intermediate plot after every this number of measurements."""

    save: bool = True
    """If True, analysis results will be saved to `output_dir`."""

    show: bool = False
    """If True, analysis figures will be displayed."""


@dataclasses.dataclass
class ExperimentGuideConfig(Config):
    """
    Configuration for experiment guide
    """

    dim_measurement_space: int = None
    """Number of dimensions of the space where measurements are performed (i.e., the feature space)."""

    lower_bounds: Sequence[float, ...] = None
    """Lower bound of sampling points to be suggested."""

    upper_bounds: Sequence[float, ...] = None
    """Upper bound of sampling points to be suggested."""

    num_candidates: int = 1
    """
    Number of sampling points to suggest. If an analytical acquisition function
    built in BoTorch is used, this must be 1; to get multiple candidates, either
    use Monte-Carlo-based acquisition functions (e.g., those whose names start
    with "q" like `qUpperConfidenceBound`), or define a custom acquisition
    function.
    """

    debug: bool = False

    def __post_init__(self):
        if self.dim_measurement_space is None:
            raise TypeError("Missing required argument: 'dim_measurement_space'")


@dataclasses.dataclass
class GPExperimentGuideConfig(ExperimentGuideConfig):

    model_class: Type[botorch.models.model.Model] = botorch.models.SingleTaskGP
    """Class handle of the model. Should be a subclass of botorch.Model."""

    model_params: dict = dataclasses.field(default_factory=dict)
    """
    Parameters of the model class. 

    The covariance kernel and its parameters also go here and should be given
    through "covariance_module" as a `gpytorch.kernels.Kernel` object. One may
    also specify a prior for the lengthscale when instantiating the kernel. The
    distribution parameters should be in the space normalized between 0 and 1.
    For example:
    ```
    model_params={'covar_module': gpytorch.kernels.MaternKernel(
        nu=2.5, 
        lengthscale_prior=gpytorch.priors.NormalPrior(0.08, 0.05))}
    ```
    Instead of using a prior, one may also use the `override_kernel_lengthscale`
    parameter to specify a fixed value for the lengthscale. 
    """

    acquisition_function_class: Type[
        botorch.acquisition.acquisition.AcquisitionFunction
    ] = botorch.acquisition.UpperConfidenceBound
    """Class handle of the acquisition function. Should be an instance of
    botorch.AcquisitionFunction."""

    acquisition_function_params: dict = dataclasses.field(default_factory=dict)
    """Parameters of the acquisition function."""

    optimizer_class: Type[Optimizer] = ContinuousOptimizer
    """
    The function handle of the optimization function. The function should have
    the acquisition function, q (number of candidates) and other parameters as
    arguments.
    """

    optimizer_params: dict = dataclasses.field(default_factory=dict)
    """
    Parameters of the optimizer constructor, not including `bounds`,
    `num_candidates` (these arguments are filled in in the ExperimentGuide class
    based on other config settings).
    """

    override_kernel_lengthscale: Optional[float] = None
    """
    If given, the lengthscale parameter of the kernel will be overridden with
    this number, but its prior distribution will not be changed. This value
    should be given in the original scale of the data, i.e., without any
    normalization.
    """

    noise_variance: Optional[float] = None
    """Noise variance of the observations."""

    beta: float = 0.99
    """Decay factor of the weights of add-on terms in the acquisition
    function."""

    stopping_criterion_configs: Optional[StoppingCriterionConfig] = None
    """Early stopping criterion configurations."""

    def __post_init__(self):
        super().__post_init__()
        if "input_transform" in self.model_params.keys():
            raise ValueError(
                "I see you specified input_transform in model_params. Don't do it! Data are normalized/"
                "standardized in GPExperimentGuide automatically."
            )
        if "outcome_transform" in self.model_params.keys():
            raise ValueError(
                "I see you specified outcome_transform in model_params. Don't do it! Data are normalized/"
                "standardized in GPExperimentGuide automatically."
            )


@dataclasses.dataclass
class XANESExperimentGuideConfig(GPExperimentGuideConfig):

    n_updates_create_acqf_weight_func: Optional[int] = None
    """
    If provided, the guide builds a weighting function that attenuates
    acquisition function values in pre-edge regions and amplifies post-edge
    regions after this number of model updates.
    """

    acqf_weight_func_floor_value: float = 0.1
    """Floor value of the sigmoid function used as acquisition weighting
    function."""

    acqf_weight_func_post_edge_gain: float = 5.0
    """Post edge gain in acquisition weighting function."""

    acqf_weight_func_post_edge_offset: float = 1.0
    """Location of post edge gain in acquisition weighting function as a
    multiple of edge width."""

    acqf_weight_func_post_edge_width: float = 0.5
    """Width of post edge gain in acquisition weighting function as a multiple
    of edge width."""

    acqf_weight_func_post_edge_decay_location: float = 50.0
    """Location where the weighting function starts to decay after the
    absorption edge."""

    project_func_sparseness_lower_bound: float = 0.5
    """
    Lower bound of the sparseness function used to calculate input feature
    projection mapping. A lower value means points in flat regions before and
    after the absorption edge are mapped more densely, which reduces their
    covariance distance and leads to smoother posterior mean in those regions.
    The input value should be between 0 and 1. 

    This parameter is only used when model is ProjectedSpaceSingleTaskGP.
    """

    project_func_sparseness_plateau_bounds: Tuple[float, float] = (-5, 50)
    """
    Lower and upper bounds of the sparseness function's plateau used in feature
    projection. The actual bounds are peak_location -/+
    project_func_sparseness_plateau_bounds[0/1] * peak_width, where peak_location
    and peak_width are automatically detected by the algorithm. Ideally, the
    bound should exactly include the region with fast variation (i.e., those
    requiring a smaller lengthscale). 
    
    This parameter is only used when model is ProjectedSpaceSingleTaskGP.
    """

    use_spline_interpolation_for_posterior_mean: bool = False
    """
    When True, posterior mean will be estimated using the cubic spline interpolation of past measurements instead
    of being calculated using Gaussian process. This could avoid oscillation that occurs in flat regions when 
    the lengthscale is too small. Note that this only affects the final reconstructed data but does not affect
    the posterior mean in acquisition function. To use interpolation also in acquisition function, choose
    a subclass of `PosteriorStandardDeviationDerivedAcquisition`, and set `estimate_posterior_mean_by_interpolation`
    to True in `acquisition_function_params`.
    """


@dataclasses.dataclass
class SampleParams(Config):
    pass


@dataclasses.dataclass
class SpatialSampleParams(SampleParams):

    image: np.ndarray = None
    """Sample image."""

    psize_nm: float = 0.0
    """Pixel size in nm."""


@dataclasses.dataclass
class ExperimentSetupParams(Config):
    pass


@dataclasses.dataclass
class FlyScanExperimentSetupParams(ExperimentSetupParams):

    psize_nm: float = 0.0
    """Pixel size in nm."""

    scan_speed_nm_sec: float = 1.0
    """Probe speed in nm/sec."""

    exposure_sec: float = 0.9
    """The time that the sample is exposed per measurement."""

    deadtime_sec: float = 1.0
    """The time that the detector is inactive and therefore not collecting data
    between measurements."""

    probe: Optional[np.ndarray] = None
    """
    The probe function. If None, it will be assumed to be a delta function.
    Otherwise, it should be a 2D array containing the probe that has the same
    pixel size as the sample image.
    """

    @property
    def exposure_length_nm(self):
        """Exposure length in nm"""
        return self.exposure_sec * self.scan_speed_nm_sec

    @property
    def exposure_length_pixel(self):
        """Exposure length in pixel equivalent"""
        return self.exposure_length_nm / self.psize_nm

    @property
    def dead_length_nm(self):
        """Dead scan length in nm"""
        return self.deadtime_sec * self.scan_speed_nm_sec

    @property
    def dead_length_pixel(self):
        """Dead scan length in pixel equivalent"""
        return self.dead_length_nm / self.psize_nm


@dataclasses.dataclass
class SimulationConfig(Config):

    sample_params: SampleParams = None
    """Sample parameters."""

    setup_params: ExperimentSetupParams = None
    """Setup parameters."""

    eps: float = 1e-6
    """Machine precision tolerance."""


@dataclasses.dataclass
class FlyScanSimulationConfig(SimulationConfig):
    """
    Configuration for a flyscan simulation
    """

    sample_params: SpatialSampleParams = None
    """Sample parameters."""

    setup_params: FlyScanExperimentSetupParams = None
    """Setup parameters."""

    num_pts_for_integration_per_measurement: Optional[int] = None
    """
    The number of points whose values are to be integrated to yield the fly scan
    measurement. Only used for simulation. Either this or
    `step_size_for_integration_nm` must be given.
    """

    step_size_for_integration_nm: Optional[float] = None
    """
    The separation between points integrated to estimate the measured intensity
    in a fly-scan measurement. Either this or
    `num_pts_for_integration_per_measurement` must be given.
    """

    def __post_init__(self):
        if (
            self.num_pts_for_integration_per_measurement is not None
            and self.step_size_for_integration_nm is not None
        ):
            raise ValueError(
                'Only one of "step_size_for_integration_nm" and "num_pts_for_integration_per_measurement" '
                "should be given."
            )

    @property
    def step_size_for_integration_pixel(self):
        """
        The step size for integration in pixels
        """
        if self.step_size_for_integration_nm:
            return self.step_size_for_integration_nm / self.setup_params.psize_nm
        else:
            assert self.num_pts_for_integration_per_measurement is not None
            return (
                self.setup_params.exposure_length_pixel
                / self.num_pts_for_integration_per_measurement
            )
