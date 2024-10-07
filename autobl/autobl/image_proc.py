"""Image processing including reconstruction, fitting"""

from typing import Literal, Optional, Tuple

# import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
import torch

from autobl.bounding_box import BoundingBox


def fit_circle(point_list, return_residue=False):
    """
    Solve a least squares system to fit points to a circle defined by center
    (yc, xc) and radius r

    A = [ |  |  |  ]
        [ y, x,ones]
        [ |  |  |  ]
    b = [      |     ]
        [ y**2 + x**2]
        [      |     ]
    x = [2*yc]
        [2*xc]
        [r**2 - yc**2 - xc**2]

    s.t. Ax = [2*y*yc + 2*x*xc + r**2 - yc**2 - xc**2] == b = [y**2 + x**2]

    Derived from:
       (y - yc)**2           + (x - xc)**2           = r**2
       y**2 - 2*y*yc + yc**2 + x**2 - 2*x*yc + xc**2 = r**2
    """
    point_list = np.array(point_list)
    y, x = point_list[:, 0], point_list[:, 1]
    a_mat = np.stack([y, x, np.ones_like(y)], axis=1)
    b_vec = y**2 + x**2
    x_vec = np.linalg.pinv(a_mat) @ b_vec
    residue = np.mean((a_mat @ x_vec - b_vec) ** 2)
    yc = x_vec[0] / 2
    xc = x_vec[1] / 2
    r = np.sqrt(x_vec[2] + yc**2 + xc**2)
    if return_residue:
        return (yc, xc, r), residue
    return yc, xc, r


def calculate_circle_fitting_residue(circle_params, ref_mask):
    """
    Calculate the MSE residue between a disk determined by (yc, xc, r) and a binary mask.
    """
    yc, xc, r = circle_params
    y, x = np.mgrid[: ref_mask.shape[0], : ref_mask.shape[1]]
    circ_mask = (y - yc) ** 2 + (x - xc) ** 2 <= r**2
    residue = np.mean((circ_mask - ref_mask) ** 2)

    return residue


def calculate_circle_fitting_iou(circle_params, ref_mask):
    """
    Calculate the MSE residue between a disk determined by (yc, xc, r) and a binary mask.
    """
    yc, xc, r = circle_params
    y, x = np.mgrid[: ref_mask.shape[0], : ref_mask.shape[1]]
    circ_mask = (y - yc) ** 2 + (x - xc) ** 2 <= r**2
    area_intersect = np.sum(circ_mask * ref_mask)
    iou = area_intersect / (np.sum(circ_mask) + np.sum(ref_mask) - area_intersect)
    return iou


def get_region_bbox(mask):
    """
    Get the bounding box of a region
    """
    ys, xs = np.nonzero(mask)
    return BoundingBox([ys.min(), xs.min(), ys.max() + 1, xs.max() + 1])


def find_window_location_with_most_peaks(window_size, peak_list):
    """
    Find the location of a window that contains the most peaks.

    :param window_size: int.
    :param peak_list: list[int]. Peak locations in pixel.
    :param range: tuple(int, int). Allowed range of the starting point of the window.
    :return: int.
    """
    peak_list = np.sort(peak_list)
    st = peak_list[0]
    end = peak_list[-1] - window_size
    if end <= st:
        return st
    pos_count = np.zeros([end - st + 1, 2])
    for i, x in enumerate(range(st, end + 1)):
        v1 = st
        v2 = st + window_size
        pos_count[i, 0] = x
        pos_count[i, 1] = np.count_nonzero(
            np.logical_and(peak_list >= v1, peak_list <= v2)
        )
    i_max = np.argmax(pos_count[:, 1])
    return int(pos_count[i_max, 0])


def point_to_line_distance(pts, line_pt_1, line_pt_2):
    """
    Find the perpendicular distance from an array of points to a vector.

    :param pts: np.ndarray. Array of point.
    :param line_pt_1: np.ndarray. The first point on the line.
    :param line_pt_2: np.ndarray. The second point on the line.
    :return: A [pts.shape[0],] array.
    """
    d = np.abs(
        np.cross(line_pt_2 - line_pt_1, pts - line_pt_1)
        / np.linalg.norm(line_pt_2 - line_pt_1)
    )
    return d


class Reconstructor:
    """
    Reconstruct an image from sparse points.

    Inverse distance weighted interpolation (IDW) is adapted from FaST:
    https://github.com/saugatkandel/fast_smart_scanning
    """

    def __init__(
        self,
        method: Literal["idw", "linear"] = "idw",
        options: Optional[dict] = None,
        backend: Literal["numpy", "torch"] = "numpy",
    ):
        """
        The constructor.

        :param method: str. The method of interpolation.
            'idw': inverse distance weighted interpolation.
            'linear': linear interpolation using scipy.interpolate.griddata.
        :param options: Optional[dict]. Options.
            When method is 'idw':
                n_neighbors: number of neighbors whose values are used to
                    calculate the interpolation for each point.
        :param backend: str. 'numpy' or 'torch'.
        :return: np.ndarray.
        """
        self.method = method
        self.options = options if options is not None else {}
        self.backend = backend

    def reconstruct(
        self,
        points: np.ndarray,
        values: np.ndarray,
        meshgrids: Tuple[np.ndarray, np.ndarray] = None,
        xi: np.ndarray = None,
        n_neighbors: int = None,
    ):
        """
        Reconstruct a dense image.

        :param points: np.ndarray. A (N, 2) array of measured points.
        :param values: np.ndarray. A 1-D array of measured values.
        :param meshgrids:
        :param n_neighbors: number of nearest neighbors for IDW reconstruction
        :return:
        """
        if self.method == "linear":
            recon = self.reconstruct_linear(points, values, meshgrids=meshgrids, xi=xi)
        elif self.method == "idw":
            recon = self.reconstruct_idw(
                points, values, meshgrids=meshgrids, xi=xi, n_neighbors=n_neighbors
            )
        else:
            raise ValueError(f"{self.method} is not a valid method.")
        return recon

    def reconstruct_linear(
        self,
        points: np.ndarray,
        values: np.ndarray,
        meshgrids: Tuple[np.ndarray, np.ndarray] = None,
        xi: np.ndarray = None,
    ):
        """
        Linear interpolation (using simplices) of points and values
        """
        if meshgrids is not None:
            grid_y, grid_x = meshgrids
            recon = scipy.interpolate.griddata(points, values, (grid_y, grid_x))
        elif xi is not None:
            recon = scipy.interpolate.griddata(points, values, xi)
        return recon

    def reconstruct_idw(
        self,
        points: np.ndarray,
        values: np.ndarray,
        meshgrids: Tuple[np.ndarray, np.ndarray] = None,
        xi: np.ndarray = None,
        n_neighbors: int = None,
        power: float = 2.0,
    ):
        """
        Inverse distance weighted interpolation with nearest neighbors

        :param points: np.ndarray. points to interpolate between, shape (num
            interp points, dimension)
        :param values: np.ndarray. values to interpolate between, shape (num
            interp points,)
        :param meshgrids: Tuple[np.ndarray, np.ndarray]. grid to get
            interpolated values for
        :param xi: np.ndarray. points to measure at
        :param n_neighbors: number of nearest neighbors to use in the
            reconstruction
        :param power: float. power of the inverse distance weighting
        :param backend: str. backend to use for inverse distance weighting.
            Can be "numpy" or "torch".
        """
        if self.backend == "torch":
            if isinstance(points, np.ndarray):
                points = torch.from_numpy(points)
            if isinstance(values, np.ndarray):
                values = torch.from_numpy(values)
            if meshgrids is not None and isinstance(meshgrids[0], np.ndarray):
                meshgrids = [torch.from_numpy(x) for x in meshgrids]
                xi = torch.stack(meshgrids, axis=-1).reshape(-1, len(meshgrids))
            if xi is not None and isinstance(xi, np.ndarray):
                xi = torch.from_numpy(xi)
            recon = self.reconstruct_idw_torch(
                points, values, meshgrids, xi, n_neighbors, power
            )
            return recon

        if meshgrids is not None:
            xi = np.stack(meshgrids, axis=-1).reshape(-1, len(meshgrids))
        if n_neighbors is None:
            n_neighbors = self.options.get("n_neighbors", 4)
        if n_neighbors == -1:
            # Complete reconstruction with all points
            recon = self._idw(
                points,
                values,
                xi,
                power=self.options.get("power", 2.0),
            )
            if meshgrids is not None:
                return recon.reshape(meshgrids[0].shape)
            return recon
        knn_engine = NearestNeighbors(n_neighbors=n_neighbors)
        knn_engine.fit(points)

        # Find nearest measured points for each queried point.
        nn_dists, nn_inds = knn_engine.kneighbors(xi, return_distance=True)
        nn_weights = self._compute_neighbor_weights(nn_dists, power=power)
        nn_values = np.take(values, nn_inds)

        recon = np.sum(nn_values * nn_weights, axis=1)
        if meshgrids is not None:
            recon = recon.reshape(meshgrids[0].shape)
        return recon

    def reconstruct_idw_torch(
        self,
        points: torch.Tensor,
        values: torch.Tensor,
        meshgrids: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        xi: Optional[torch.Tensor] = None,
        n_neighbors: Optional[int] = None,
        power: float = 2.0,
        nn_inds: torch.Tensor = None,
    ):
        """
        Inverse distance weighted interpolation with nearest neighbors (pytorch
        implementation)
        
        :param points: torch.Tensor. points to interpolate between, shape (num
            interp points, dimension)
        :param values: torch.Tensor. values to interpolate between, shape (num
            interp points,)
        :param meshgrids: Tuple[torch.Tensor, torch.Tensor]. grid to get
            interpolated values for
        :param xi: torch.Tensor. points to measure at
        :param n_neighbors: int. number of nearest neighbors to use in the
            reconstruction
        :param power: float. power for inverse distance weighting
        :param nn_inds: torch.Tensor. indices of nearest neighbors; if provided,
            neighbors are not computed here
        """
        # points = points.type(values.dtype)
        if meshgrids is not None and xi is None:
            # meshgrids = [x.type(values.dtype) for x in meshgrids]
            xi = torch.stack(meshgrids, dim=-1).reshape(-1, len(meshgrids))
        if n_neighbors is None:
            n_neighbors = self.options.get("n_neighbors", 4)
        if n_neighbors == -1:
            raise NotImplementedError("Not implemented for torch.")
        if nn_inds is None or xi.shape[0] != nn_inds.shape[0]:
            knn_engine = NearestNeighbors(n_neighbors=n_neighbors)
            knn_engine.fit(points.detach().cpu().numpy())

            # Find nearest measured points for each queried point.
            if isinstance(xi, np.ndarray):
                nn_inds = knn_engine.kneighbors(xi, return_distance=False)
            elif isinstance(xi, torch.Tensor):
                nn_inds = knn_engine.kneighbors(
                    xi.detach().numpy(), return_distance=False
                )
        nn_dists = torch.sqrt(
            # torch.sum((xi.unsqueeze(1) - points[nn_inds]) ** 2, dim=-1)
            torch.sum((xi[:, None, :] - points[torch.from_numpy(nn_inds)]) ** 2, dim=-1)
        )
        nn_weights = self._compute_neighbor_weights(
            nn_dists, power=power, backend="torch"
        )
        nn_values = values[torch.tensor(nn_inds)]

        recon = torch.sum(nn_values * nn_weights, dim=1)
        if meshgrids is not None:
            recon = recon.reshape(meshgrids[0].shape)
        return recon

    def reconstruct_idw_grad(
        self,
        points: np.ndarray,
        values: np.ndarray,
        meshgrids: Tuple[np.ndarray, np.ndarray] = None,
        xi: np.ndarray = None,
        n_neighbors: int = None,
        power: float = 2.0,
        epsilon: float = 1e-7,
    ):
        """
        Inverse distance weighted interpolation gradient with nearest neighbors

        :param points: np.ndarray. points to interpolate between, shape (num
            interp points, dimension)
        :param values: np.ndarray. values to interpolate between, shape (num
            interp points,)
        :param meshgrids: Tuple[np.ndarray, np.ndarray]. grid to get
            interpolated values for
        :param xi: np.ndarray. points to measure at
        :param n_neighbors: int. number of nearest neighbors to use in the
            reconstruction
        """
        if meshgrids is not None:
            xi = np.stack(meshgrids, axis=-1).reshape(-1, len(meshgrids))
        if n_neighbors is None:
            n_neighbors = self.options.get("n_neighbors", 4)
        if n_neighbors == -1:
            # Complete reconstruction with all points
            recon = self._idw_grad(
                points,
                values,
                xi,
                power=self.options.get("power", 2.0),
            )
            if meshgrids is not None:
                return np.moveaxis(
                    recon.reshape((meshgrids[0].shape[0], meshgrids[0].shape[1], 2)),
                    -1,
                    0,
                )
            return recon
        knn_engine = NearestNeighbors(n_neighbors=n_neighbors)
        knn_engine.fit(points)

        # Find nearest measured points for each queried point.
        nn_dists, nn_inds = knn_engine.kneighbors(xi, return_distance=True)
        values = np.take(values, nn_inds)

        grad = np.zeros((2, meshgrids[0].shape[0], meshgrids[0].shape[1]))

        # correct for zero distance
        # mask = nn_dists == 0.0
        # nn_dists[np.sum(mask, axis=1) > 0, :] = np.inf
        # nn_dists[mask] = 1.0
        # inverse distances from xi (axis = 0) to point (axis = 1)
        inv_distances = 1 / (nn_dists + epsilon)
        sum_inv_distances = np.sum(inv_distances**power, axis=1)

        inv_cubed = inv_distances ** (power + 2)
        sum_val_inv_distances = np.sum(values * (inv_distances**power), axis=1)
        val_inv_cubed = values * inv_distances ** (power + 2)
        diff_x = xi[:, np.newaxis, 0] - points[nn_inds, 0]
        diff_y = xi[:, np.newaxis, 1] - points[nn_inds, 1]

        grad[0, :, :] = (
            (
                -sum_inv_distances * np.sum(val_inv_cubed * diff_x, axis=1)
                + sum_val_inv_distances * np.sum(inv_cubed * diff_x, axis=1)
            )
            / (sum_inv_distances**2)
        ).reshape(meshgrids[0].shape)

        grad[1, :, :] = (
            (
                -sum_inv_distances * np.sum(val_inv_cubed * diff_y, axis=1)
                + sum_val_inv_distances * np.sum(inv_cubed * diff_y, axis=1)
            )
            / (sum_inv_distances**2)
        ).reshape(meshgrids[0].shape)

        return grad

    def reconstruct_idw_grad_torch(
        self,
        points: torch.Tensor,
        values: torch.Tensor,
        meshgrids: Tuple[torch.Tensor, torch.Tensor] = None,
        xi: torch.tensor = None,
        n_neighbors: int = None,
        nn_inds: torch.tensor = None,
        power: float = 2.0,
        epsilon: float = 1e-7,
    ):
        """
        Inverse distance weighted interpolation gradient with nearest neighbors
        (pytorch implementation)

        :param points: torch.Tensor. points to interpolate between, shape (num
            interp points, dimension)
        :param values: torch.Tensor. values to interpolate between, shape (num
            interp points,)
        :param meshgrids: Tuple[torch.Tensor, torch.Tensor]. grid to get
            interpolated values for
        :param xi: torch.Tensor. points to measure at
        :param n_neighbors: int. number of nearest neighbors to use in the
            reconstruction
        :param nn_inds: torch.Tensor. indices of nearest neighbors; if provided,
            neighbors are not computed here
        :param power: float. power for inverse distance weighting
        :param epsilon: float. small number added to distances to avoid division
            by zero
        """
        if meshgrids is not None:
            xi = torch.stack(meshgrids, axis=-1).reshape(-1, len(meshgrids))
        if nn_inds is None:
            if n_neighbors is None:
                n_neighbors = self.options.get("n_neighbors", 4)
            if n_neighbors == -1:
                n_neighbors = len(points)
            knn_engine = NearestNeighbors(n_neighbors=n_neighbors)
            knn_engine.fit(points.detach().numpy())

            # Find nearest measured points for each queried point.
            nn_inds = knn_engine.kneighbors(xi.detach().numpy(), return_distance=False)

        nn_dists = torch.sqrt(
            # torch.sum((xi.unsqueeze(1) - points[nn_inds]) ** 2, dim=-1)
            torch.sum((xi[:, None, :] - points[nn_inds]) ** 2, dim=-1)
        )
        values = values[torch.tensor(nn_inds)]

        inv_distances = 1 / (nn_dists + epsilon)
        sum_inv_distances = torch.sum(inv_distances**power, axis=1)

        inv_cubed = inv_distances ** (power + 2)
        sum_val_inv_distances = torch.sum(values * (inv_distances**power), axis=1)
        val_inv_cubed = values * inv_distances ** (power + 2)
        diff_x = xi[:, None, 0] - points[nn_inds, 0]
        diff_y = xi[:, None, 1] - points[nn_inds, 1]

        dx = (
            -sum_inv_distances * torch.sum(val_inv_cubed * diff_x, axis=1)
            + sum_val_inv_distances * torch.sum(inv_cubed * diff_x, axis=1)
        ) / (sum_inv_distances**2)

        dy = (
            -sum_inv_distances * torch.sum(val_inv_cubed * diff_y, axis=1)
            + sum_val_inv_distances * torch.sum(inv_cubed * diff_y, axis=1)
        ) / (sum_inv_distances**2)

        return torch.stack((dx, dy), dim=0)

    @staticmethod
    def _compute_neighbor_weights(
        neighbor_distances: np.ndarray | torch.Tensor,
        power: float = 2.0,
        epsilon: float = 1e-7,
        backend: Literal["numpy", "torch"] = "numpy",
    ) -> np.ndarray | torch.Tensor:
        """
        Calculating the weights for how each neighboring data point contributes
        to the reconstruction for the current location.

        First, the weights are calculated to be inversely proportional to the
        distance from teh current point. Next, the weights are normalized so
        that the total weight sums up to 1 for each reconstruction point.
        """
        be = np if backend == "numpy" else torch
        # dists = np.copy(neighbor_distances)
        # mask = dists == 0.0
        # dists[np.sum(mask, axis=1) > 0, :] = np.inf
        # dists[mask] = 1.0
        unnormalized_weights = 1.0 / ((neighbor_distances + epsilon) ** power)
        sum_over_row = be.sum(unnormalized_weights, axis=1, keepdims=True)
        weights = unnormalized_weights / sum_over_row
        return weights

    @staticmethod
    def _idw(
        points: np.ndarray, values: np.ndarray, xi: np.ndarray, power: float = 2.0
    ) -> np.ndarray:
        """Inverse distance weighted interpolation for all points

        Interpolate by weighting by inverse distance

        :param points: np.ndarray. points to interpolate between, shape (num interp
            points, dimension)
        :param values: np.ndarray. values to interpolate between, shape (num interp
            points,)
        :param xi: np.ndarray. points to get interpolated values at, shape (num
            sample points, dimension)
        :return: np.ndarray. interpolated values, shape (num sample points,)
        """
        dists = cdist(xi, points)
        # correct for zero distance
        mask = dists == 0.0
        dists[np.sum(mask, axis=1) > 0, :] = np.inf
        dists[mask] = 1.0
        # inverse distances from xi (axis = 0) to point (axis = 1)
        inv_distances = 1 / (dists**power)
        sum_inv_distances = np.sum(inv_distances, axis=1)
        # Multiply values by inverse distances and divide by sum
        vi = np.sum(values[np.newaxis, :] * inv_distances, axis=1) / sum_inv_distances
        return vi

    @staticmethod
    def _idw_grad(
        points: np.ndarray, values: np.ndarray, xi: np.ndarray, power: float = 2.0
    ) -> np.ndarray:
        """Gradient of inverse distance weighted interpolation for all points

        Interpolate by weighting by inverse distance

        :param points: np.ndarray. points to interpolate between, shape (num interp
            points, dimension)
        :param values: np.ndarray. values to interpolate between, shape (num interp
            points,)
        :param xi: np.ndarray. points to get gradient values at, shape (num
            sample points, dimension)
        :return: np.ndarray. gradient values, shape (num sample points,
            dimension)
        """
        grad = np.zeros(xi.shape)
        dists = cdist(xi, points)

        # correct for zero distance
        mask = dists == 0.0
        dists[np.sum(mask, axis=1) > 0, :] = np.inf
        dists[mask] = 1.0
        # inverse distances from xi (axis = 0) to point (axis = 1)
        inv_distances = 1 / dists
        sum_inv_distances = np.sum(inv_distances**power, axis=1)

        inv_cubed = inv_distances ** (power + 2)
        sum_val_inv_distances = np.sum(
            values[np.newaxis, :] * (inv_distances**power), axis=1
        )
        val_inv_cubed = values[np.newaxis, :] * inv_distances ** (power + 2)
        diff_x = xi[:, np.newaxis, 0] - points[np.newaxis, :, 0]
        diff_y = xi[:, np.newaxis, 1] - points[np.newaxis, :, 1]

        grad[:, 0] = (
            -sum_inv_distances * np.sum(val_inv_cubed * diff_x, axis=1)
            + sum_val_inv_distances * np.sum(inv_cubed * diff_x, axis=1)
        ) / (sum_inv_distances**2)

        grad[:, 1] = (
            -sum_inv_distances * np.sum(val_inv_cubed * diff_y, axis=1)
            + sum_val_inv_distances * np.sum(inv_cubed * diff_y, axis=1)
        ) / (sum_inv_distances**2)

        return grad


class DenseReconstructor(Reconstructor):
    """Reconstruct densely using a meshgrid"""

    def reconstruct(
        self,
        points: np.ndarray,
        values: np.ndarray,
        meshgrids: Tuple[np.ndarray, np.ndarray] = None,
        xi: np.ndarray = None,
        n_neighbors: int = None,
    ):
        """
        Reconstruct a dense image.

        :param points: np.ndarray. A (N, 2) array of measured points.
        :param values: np.ndarray. A 1-D array of measured values.
        :param meshgrids: dense meshgrid to do the reconstruction
        :param n_neighbors: number of nearest neighbors for IDW reconstruction
        :param backend: str. 'numpy' or 'torch'.
        :return:
        """
        if self.method == "linear":
            recon = self.reconstruct_linear(points, values, meshgrids=meshgrids)
        elif self.method == "idw":
            recon = self.reconstruct_idw(
                points, values, meshgrids=meshgrids, n_neighbors=n_neighbors
            )
        else:
            raise ValueError(f"{self.method} is not a valid method.")
        return recon
