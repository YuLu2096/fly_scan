"""Measurement methods"""

import copy
from typing import Callable, List, Optional  # , Sequence

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import scipy.ndimage as ndi

import autobl.steering.configs as cfg
from autobl import util


class Measurement:
    """
    Measurement class
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the
        """

    def measure(self, *args, **kwargs):
        """
        Measure the sample
        """


class SimulatedMeasurement(Measurement):
    """
    Simulate a measurement
    """

    def __init__(self, *args, f: Callable = None, data: np.ndarray = None, **kwargs):
        super().__init__(*args, **kwargs)
        assert (f is not None) or (data is not None)
        self.f = f
        self.data = data
        self.interp = None
        if self.data is not None:
            if isinstance(self.data, (list, tuple)):
                x_data, y_data = self.data
                x_data = util.to_numpy(x_data)
                y_data = util.to_numpy(y_data)
            else:
                x_data = np.arange(len(data))
                y_data = util.to_numpy(self.data)
            self.interp = lambda pts: scipy.interpolate.interpn(
                x_data, y_data, pts, bounds_error=False, fill_value=None
            )

    def measure(self, x, *args, **kwargs):
        """
        Take measurement.

        :param x: Tensor. A list of points where the values are measured. Tensor
            shape should be [n_pts, n_dims].
        :return:
        """
        if self.f is not None:
            return util.to_tensor(self.f(x))
        elif self.data is not None:
            x = util.to_numpy(x)
            x = util.to_tensor(self.interp(x))
            return x
        else:
            raise ValueError("f or data cannot both be None.")


class XANESExperimentalMeasurement(Measurement):

    def __init__(self, *args, **kwargs):
        pass

    def measure(self, x, *args, **kwargs):
        """
        Take measurement.

        :param x: Tensor. A list of points where the values are measured. Tensor shape should be [n_pts, n_dims].
        :return: Tensor. Measured values at `x`. Values are returned in shape [n_pts,].
        """
        raise NotImplementedError


class FlyScanSingleValueSimulationMeasurement(Measurement):
    """
    Fly scan simulator for techniques where each exposure measures only a single
    intensity value. The simulator's perform measurement method takes in a
    continuous scan path defined by a list of vertices; the scan path is formed
    by linearly connecting the vertices sequentially. The simulator
    automatically split the scan path into exposures based on the setting of
    exposure time in `sample_params`. Between exposures, there can be dead times
    where no data is acquired based on the setting of dead time. For each
    exposure, several points are sampled along the path, whose interval is
    determined by the setting of `step_size_for_integration_nm` in
    `sample_params`. The intensities sampled at all sampling points are averaged
    as the measurement for that exposure. The positions of all sampling points
    are averaged as the position for that exposure.
    """

    def __init__(self, configs: cfg.FlyScanSimulationConfig):
        """
        The constructor.
        """
        super().__init__()
        self.configs = configs
        self.measured_values = None
        self.measured_positions = None
        self.eps = 1e-6
        self.points_to_sample_all_exposures = []
        self.probe_image = None

    def measure(self, vertex_list, *args, vertex_unit="pixel", **kwargs):
        """
        Perform measurement given a fly scan path defined by a list of vertices.

        :param vertex_list: list[list[float, float]]. A list of vertex positions
            that define the scan path, ordered in (y, x). The total number of
            segments is `len(vertex_list) - 1`; for segment i, probe moves from
            `vertex_list[i]` to `vertex_list[i + 1]`.
        :param vertex_unit: str. Can be 'pixel' or 'nm'.
        :return list[float]: measured values at all exposures. The positions of
            the exposures can be retrieved from the `measured_positions`
            attribute.
        """
        vertex_list = np.asarray(vertex_list)
        if vertex_unit == "nm":
            vertex_list = vertex_list / self.configs.setup_params.psize_nm

        self.build_sampling_points(vertex_list)

        meas_value_list = []
        meas_pos_list = []
        for pts_to_sample in self.points_to_sample_all_exposures:
            if len(pts_to_sample) == 0:
                continue
            sampled_vals = self.get_interpolated_values_from_image(pts_to_sample)
            # averages of measurements and positions
            meas_value_list.append(np.mean(sampled_vals))
            meas_pos_list.append(np.mean(pts_to_sample, axis=0))
        self.measured_values = np.array(meas_value_list)
        self.measured_positions = np.stack(meas_pos_list, axis=0)
        return self.measured_values

    def build_sampling_points(self, vertex_list: np.ndarray | List[List[float]]):
        """
        Given vertices, compute the points that are sampled along the scan path
        (between vertices)
        """
        if not isinstance(vertex_list, np.ndarray):
            vertex_list = np.array(vertex_list)
        # distances between vertices
        lengths = np.zeros((vertex_list.shape[0],))
        lengths[1:] = np.linalg.norm(np.diff(vertex_list, axis=0), axis=1)

        # cumulative distance along points
        arc_lengths = np.cumsum(lengths)

        # arc length interpolator
        interpolator = scipy.interpolate.make_interp_spline(
            arc_lengths, vertex_list, k=1
        )

        # generate samples along arc length interpolator
        step_size = self.configs.step_size_for_integration_nm
        steps = np.arange(0, arc_lengths[-1], step_size)

        # determine if the samples are in an exposure +- eps
        exposure = (
            np.fmod(
                steps / self.configs.setup_params.scan_speed_nm_sec + self.configs.eps,
                self.configs.setup_params.exposure_sec
                + self.configs.setup_params.deadtime_sec,
            )
            <= self.configs.setup_params.exposure_sec + 2 * self.configs.eps
        )

        # get the associated exposure number for each sample point
        exposure_number = (
            (steps + self.configs.eps)
            // (
                self.configs.setup_params.exposure_sec
                + self.configs.setup_params.deadtime_sec
            )
        ).astype(int)

        # get positions for each exposure
        self.points_to_sample_all_exposures = []
        for i in range(exposure_number[-1]):
            self.points_to_sample_all_exposures.append(
                interpolator(steps[np.logical_and(exposure, exposure_number == i)])
            )

    def build_sampling_points_old(self, vertex_list):
        """
        Given vertices, compute the points that are sampled along the scan path
        (between vertices)
        """
        points_to_sample_all_exposures = []
        i_input_segment = 0
        length_covered_in_current_segment = 0
        pt_coords = vertex_list[0]
        while i_input_segment < len(vertex_list) - 1:
            length_exposed = 0
            length_dead = 0
            length_sampling = 0
            # Add live segment.
            points_to_sample_current_exposure = [pt_coords]
            while (
                length_exposed
                < self.configs.setup_params.exposure_length_pixel - self.eps
            ):
                if i_input_segment + 1 >= len(vertex_list):
                    break
                current_direction = (
                    vertex_list[i_input_segment + 1] - vertex_list[i_input_segment]
                )
                current_seg_length = np.linalg.norm(current_direction)
                current_direction = current_direction / current_seg_length
                if (
                    length_covered_in_current_segment
                    + self.configs.step_size_for_integration_pixel
                    - length_sampling
                    <= current_seg_length
                ):
                    pt_coords = pt_coords + current_direction * (
                        self.configs.step_size_for_integration_pixel - length_sampling
                    )
                    points_to_sample_current_exposure.append(pt_coords)
                    length_covered_in_current_segment += (
                        self.configs.step_size_for_integration_pixel - length_sampling
                    )
                    length_exposed += (
                        self.configs.step_size_for_integration_pixel - length_sampling
                    )
                    length_sampling = 0
                else:
                    if i_input_segment + 1 >= len(vertex_list):
                        break
                    pt_coords = pt_coords + current_direction * (
                        current_seg_length - length_covered_in_current_segment
                    )
                    i_input_segment += 1
                    length_exposed += (
                        current_seg_length - length_covered_in_current_segment
                    )
                    length_sampling += (
                        current_seg_length - length_covered_in_current_segment
                    )
                    length_covered_in_current_segment = 0
            # Update variables for dead segment.
            while length_dead < self.configs.setup_params.dead_length_pixel - self.eps:
                if i_input_segment + 1 >= len(vertex_list):
                    break
                current_direction = (
                    vertex_list[i_input_segment + 1] - vertex_list[i_input_segment]
                )
                current_seg_length = np.linalg.norm(current_direction)
                current_direction = current_direction / current_seg_length
                if (
                    length_covered_in_current_segment
                    + self.configs.setup_params.dead_length_pixel
                    - length_dead
                    <= current_seg_length
                ):
                    pt_coords = pt_coords + current_direction * (
                        self.configs.setup_params.dead_length_pixel - length_dead
                    )
                    length_covered_in_current_segment += (
                        self.configs.setup_params.dead_length_pixel - length_dead
                    )
                    break
                else:
                    if i_input_segment + 1 >= len(vertex_list):
                        break
                    pt_coords = vertex_list[i_input_segment + 1]
                    i_input_segment += 1
                    length_dead += (
                        current_seg_length - length_covered_in_current_segment
                    )
                    length_covered_in_current_segment = 0
            points_to_sample_all_exposures.append(points_to_sample_current_exposure)
        self.points_to_sample_all_exposures = points_to_sample_all_exposures

    def plot_sampled_points(self):
        """
        Plot the points that will be sampled along the path
        """
        pts = np.concatenate(self.points_to_sample_all_exposures, axis=0)
        # pts = self.points_to_sample_all_exposures

        fig, ax = plt.subplots(1, 1)
        ax.scatter(pts[:, 1], pts[:, 0])
        plt.show()
        return fig, ax

    def get_interpolated_values_from_image_old(self, point_list, normalize_probe=True):
        """
        Obtain interpolated values from the image at given locations.

        :param point_list: list[list[float, float]]. List of point positions.
        :return: list[float].
        """
        if not isinstance(point_list, np.ndarray):
            point_list = np.array(point_list)
        y = point_list[:, 0]
        x = point_list[:, 1]
        if self.configs.setup_params.probe is None:
            # If probe function is not given, assume delta function.
            return ndi.map_coordinates(
                self.configs.sample_params.image, [y, x], order=1, mode="nearest"
            )
        else:
            # Prepare a list of coordinates that include n by m region around
            # each sampled point, where (n, m) is the probe shape.
            sampled_vals = []
            probe = self.configs.setup_params.probe
            if normalize_probe:
                probe = probe / np.sum(probe)
            for this_y, this_x in point_list:
                # Expand around each point to capture probe area
                this_y_all = np.linspace(
                    this_y - probe.shape[0] / 2.0 + 0.5,
                    this_y + probe.shape[0] / 2.0 - 0.5,
                    probe.shape[0],
                )
                this_x_all = np.linspace(
                    this_x - probe.shape[1] / 2.0 + 0.5,
                    this_x + probe.shape[1] / 2.0 - 0.5,
                    probe.shape[1],
                )

                # Combine expansions in x and y to get total expanded area
                xx, yy = np.meshgrid(this_x_all, this_y_all)
                yy = yy.reshape(-1)
                xx = xx.reshape(-1)

                # Get image interpolated values from the image
                vals = ndi.map_coordinates(
                    self.configs.sample_params.image, [yy, xx], order=1, mode="nearest"
                )
                # ... and multiply them with the probe weights
                val = np.sum(vals * probe.reshape(-1))
                sampled_vals.append(val)
            return sampled_vals

    def get_interpolated_values_from_image(
        self, point_list: np.ndarray, normalize_probe: bool = True
    ):
        """
        Obtain interpolated values from the image at given locations.

        Apply the probe PSF as a convolution and then sample (with
        interpolation) the convolved image.

        :param point_list: list[list[float, float]]. List of point positions.
        :return: list[float].
        """
        probe = self.configs.setup_params.probe
        if probe is None:
            probe = np.ones((1, 1))
        if normalize_probe:
            probe /= np.sum(probe)
        if self.probe_image is None:
            self.probe_image = convolve_probe_image(
                self.configs.sample_params.image, probe
            )
        if not isinstance(point_list, np.ndarray):
            point_list = np.array(point_list)  # [i_sample, (y,x)]
        return get_interpolated_values_from_image(
            point_list,
            self.configs.sample_params.image,
            self.probe_image,
            probe,
            normalize_probe,
        )


class FlyScanPathGenerator:
    """
    Fly scan path generator for raster scans
    """

    def __init__(self, shape, psize_nm=None, return_coordinates_type="pixel"):
        """
        Fly scan path generator.

        :param shape: list[int, int]. Shape of the support in pixel.
        :param psize_nm: float. Pixel size in nm.
        :param return_coordinates_type: str. Can be 'pixel' or 'nm'. Sets the
            unit of the returned coordinates.
        """
        self.shape = shape
        self.psize_nm = psize_nm
        self.return_coordinates_type = return_coordinates_type
        self.generated_path = []
        self.dead_segment_mask = []

    def plot_path(self):
        """
        Plot the flyscan path
        """
        fig = plt.figure()
        plt.plot(self.generated_path[:, 1], self.generated_path[:, 0])
        plt.show()

    def generate_raster_scan_path(
        self,
        pos_top_left: List[float],
        pos_bottom_right: List[float],
        vertical_spacing: float,
    ) -> List[List[float]]:
        """
        Generate a raster (regular) scan path starting at pos_top_left and
        scanning (down) towards pos_bottom_right. Creates long left-right scan
        segments and short vertical (down) segments. Vertical segments are not
        scanned (dead).

        :param pos_top_left: list[float, float]. Top left vertex of the scan
            grid in pixel. Coordinates are defined with regards to the support.
        :param pos_bottom_right: list[float, float]. Bottom right vertex of the
            scan grid in pixel.
        :param vertical_spacing: float. Spacing of adjacent scan lines in pixel.
        :return: list[list[float, float]]. A list of vertices that define the
            scan path.
        """
        # Indicates whether the current vertex is on the left or right of the grid.
        current_side = 0
        current_point = copy.deepcopy(np.array(pos_top_left))
        self.generated_path.append(copy.deepcopy(current_point))
        while True:
            if current_side == 0:  # left
                current_point[1] = pos_bottom_right[1]
            else:  # right
                current_point[1] = pos_top_left[1]
            current_side = 1 - current_side
            self.generated_path.append(copy.deepcopy(current_point))
            if current_point[0] + vertical_spacing > pos_bottom_right[0]:
                break
            current_point[0] += vertical_spacing
            self.generated_path.append(copy.deepcopy(current_point))
        self.generated_path = np.stack(self.generated_path)

        self.dead_segment_mask = np.ones(len(self.generated_path) - 1, dtype="bool")
        self.dead_segment_mask[1::2] = False

        if self.return_coordinates_type == "nm":
            return self.generated_path * self.psize_nm
        return self.generated_path


def convolve_probe_image(image: np.ndarray, probe: np.ndarray = None) -> np.ndarray:
    """
    Convolve the image with the probe point spread function

    :param image: np.ndarray[float, float]. Image to sample.
    :param probe: np.ndarray[float, float]. Probe point spread function.
    :return: Image convolved with the point spread function.
    """
    if probe is None:
        return image
    return ndi.convolve(image, probe, mode="nearest")


def get_interpolated_values_from_image(
    point_list: np.ndarray | List[List[float]],
    image: Optional[np.ndarray] = None,
    probe_image: Optional[np.ndarray] = None,
    probe: np.ndarray = np.ones((1, 1)),
    normalize_probe: bool = True,
):
    """
    Obtain interpolated values from the image at given locations.

    Apply the probe PSF as a convolution and then sample (with
    interpolation) the convolved image.

    :param point_list: list[list[float, float]]. List of point positions.
    :param image: np.ndarray. Image to measure from.
    :param probe_image: np.ndarray. Convolved image with point spread function.
    :param probe: np.ndarray. Point spread function.
    :param normalize_probe: bool. Make probe weights sum to 1.
    :return: list[float].
    """
    if image is None and probe_image is None:
        raise ValueError("No image provided.")
    if not isinstance(point_list, np.ndarray):
        point_list = np.array(point_list)  # [i_sample, (y,x)]
    if normalize_probe:
        probe /= np.sum(probe)

    # When the probe has an even shape, convolution if off center
    shift = np.zeros((2,))
    if probe.shape[0] % 2 == 0:
        shift[0] = -0.5
    if probe.shape[1] % 2 == 0:
        shift[1] = -0.5
    if probe_image is None:
        if np.max(probe.shape) > 1:
            probe_image = convolve_probe_image(image, probe)
        else:
            probe_image = image * probe
    sampled_vals = ndi.map_coordinates(
        probe_image,
        np.transpose(point_list) + shift[:, np.newaxis],
        order=1,
        mode="nearest",
    )
    return sampled_vals
