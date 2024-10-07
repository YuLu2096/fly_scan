import logging
import time

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import scipy.signal
import skimage.transform
import skimage.filters
import sklearn.decomposition
import sklearn.neighbors
import tqdm

from autobl.bounding_box import BoundingBox
from autobl.message_logger import logger
from autobl.image_proc import *


class SegmentationAlgorithmSelector:

    option_defaults = {
        'residue_threshold': 0.2,
        'sinogram_std_threshold': 6
    }

    def __init__(self, downsample=4, method='shape_fit', options=None, *args, **kwargs):
        self.algorithm = None
        self.image = None
        self.downsample = downsample
        self.orig_shape = None
        self.method = method
        self.options = options if options is not None else {}
        self.debug = False

    def get_option_value(self, key):
        if key in self.options.keys():
            return self.options[key]
        else:
            return self.__class__.option_defaults[key]

    def set_camera_image(self, img):
        self.image = img
        self.orig_shape = img.shape

    def run_downsample(self):
        self.orig_shape = self.image.shape
        if self.downsample == 1:
            return
        self.image = skimage.transform.resize(self.image, [x // self.downsample for x in self.image.shape], order=1)

    def find_and_fit_circle(self):
        bubble_segmentor = BubbleSegmentor(downsample=1)
        bubble_segmentor.set_camera_image(self.image)
        bubble_segmentor.run()
        return bubble_segmentor.cell_window_mask_residue

    def xy2ij(self, xy_coords, y_range, x_range, shape, return_pixel_sizes=False):
        psize = np.array([(y_range[1] - y_range[0]) / shape[0],
                          (x_range[1] - x_range[0]) / shape[1]])
        ij_coords = np.array([(xy_coords[0] - y_range[0]) / psize[0], (xy_coords[1] - x_range[0]) / psize[1]])
        if return_pixel_sizes:
            return ij_coords, psize
        else:
            return ij_coords

    def run_selection_by_shape_fitting(self):
        residue_threshold = self.get_option_value('residue_threshold')
        fit_residue = self.find_and_fit_circle()
        if fit_residue > residue_threshold:
            return CapillarySegmentor
        else:
            return BubbleSegmentor

    def run_selection_by_gradient_distribution(self):
        sino_std_threshold = self.get_option_value('sinogram_std_threshold')

        grad_y = ndi.sobel(self.image, axis=0)
        grad_x = ndi.sobel(self.image, axis=1)
        grad_y_centered = grad_y - np.mean(grad_y)
        grad_x_centered = grad_x - np.mean(grad_x)
        grad_data = np.stack([grad_y_centered.reshape(-1), grad_x_centered.reshape(-1)], axis=-1)
        grad_data = grad_data[::4]

        y_coords = np.linspace(grad_data[:, 0].min(), grad_data[:, 0].max(), 30)
        x_coords = np.linspace(grad_data[:, 1].min(), grad_data[:, 1].max(), 30)
        xx, yy = np.meshgrid(x_coords, y_coords)

        bw = ((grad_data[:, 0].max() - grad_data[:, 0].min()) + (grad_data[:, 1].max() - grad_data[:, 1].min())) / 2
        bw = bw * 0.1
        kde = sklearn.neighbors.KernelDensity(kernel='linear', bandwidth=bw)
        kde.fit(grad_data)
        pdf = kde.score_samples(np.stack([yy.reshape(-1), xx.reshape(-1)], axis=-1))

        pdf = pdf.reshape(30, 30)
        finite_mask = np.isfinite(pdf)
        pdf_finite = pdf[finite_mask]
        thresh = np.percentile(pdf_finite, 0.5)
        pdf[pdf < thresh] = 0

        center_pos, psize = self.xy2ij(np.mean(grad_data, axis=0),
                                       [grad_data[:, 0].min(), grad_data[:, 0].max()],
                                       [grad_data[:, 1].min(), grad_data[:, 1].max()],
                                       pdf.shape, return_pixel_sizes=True)
        center_pos = np.round(center_pos).astype(int)
        rad = np.min([pdf.shape[0] - center_pos[0], center_pos[0], pdf.shape[1] - center_pos[1], center_pos[1]])
        pdf_cropped = pdf[center_pos[0] - rad:center_pos[0] + rad, center_pos[1] - rad:center_pos[1] + rad]
        if psize[0] / psize[1] != 1:
            pdf_cropped = ndi.zoom(pdf_cropped, zoom=[1, psize[1] / psize[0]])

        sinogram = skimage.transform.radon(pdf_cropped, theta=np.linspace(0., 180., 90), circle=True)
        sinogram_profile = sinogram[sinogram.shape[0] // 2, :]
        sino_std = np.std(sinogram_profile)
        if sino_std > sino_std_threshold:
            self.algorithm = CapillarySegmentor
        else:
            self.algorithm = BubbleSegmentor

        if self.debug:
            fig, ax = plt.subplots(1, 1)
            ax.set_aspect('equal')
            plt.scatter(grad_x.reshape(-1), grad_y.reshape(-1))
            plt.show()

            fig, ax = plt.subplots(1, 1)
            ax.set_aspect('equal')
            im = plt.imshow(pdf_cropped)
            plt.colorbar(im)
            plt.show()

        return self.algorithm

    def run_selection(self):
        self.run_downsample()
        if self.method == 'shape_fit':
            algorithm_class = self.run_selection_by_shape_fitting()
        elif self.method == 'gradient_distribution':
            algorithm_class = self.run_selection_by_gradient_distribution()
        else:
            raise ValueError('{} is not a valid method.'.format(self.method))
        self.algorithm = algorithm_class
        return algorithm_class


class Segmentor:

    def __init__(self, downsample=4, *args, **kwargs):
        self.image = None
        self.downsample = downsample
        self.orig_shape = None
        self.final_mask = None

    def set_camera_image(self, img):
        self.image = img

    def remove_kinks_on_arc(self, edge_mask, corner_threshold=0.8):
        """
        Remove the non-smooth or kink regions on an arc-shaped mask.

        :param edge_mask: np.ndarray.
        :param corner_threshold: float. Threshold to locate corners in the Harris corner map.
        :return: np.ndarray.
        """
        edge_mask = edge_mask.astype(float)
        corner_map = skimage.feature.corner_harris(edge_mask, sigma=12 / self.downsample)
        corner_map = corner_map / corner_map.max()
        corner_mask = corner_map > corner_threshold
        corner_mask = ndi.binary_dilation(corner_mask, np.ones([5, 5]))
        edge_mask[corner_mask] = 0
        edge_regions, n_regions = ndi.label(edge_mask)
        sorted_labels = self.sort_labels(edge_regions, n_regions, by='area')
        edge_mask = edge_regions == sorted_labels[0]

        return edge_mask

    def get_edge_pixel_mask(self, mask):
        mask = np.where(mask > 0, 1, 0)
        mask_edge = ndi.binary_dilation(mask, np.ones([3, 3])) - mask
        return mask_edge

    def is_round(self, mask, method='circle_fit', threshold=0.5):
        """
        Calculate the eigenvalues of the principle axes of the non-zero elements in a given mask, then calculate
        the ratio. If the ratio is close enough to 1, the distribution can be considered to be close enough to a disk.

        :param mask: np.ndarray.
        :return: bool.
        """
        if method == 'circle_fit':
            mask = mask.astype(int)
            edge_points = np.where(mask - ndi.binary_erosion(mask, structure=np.ones([3, 3])) > 0)
            edge_points = np.stack(edge_points, axis=1)
            circ_params = fit_circle(edge_points)
            iou = calculate_circle_fitting_iou(circ_params, mask)
            if iou > threshold:
                return True
            else:
                return False
        elif method == 'pca':
            eigen_value_ratio = self.calculate_region_principal_axis_eigenvalue_ratio(mask)
            if eigen_value_ratio < 10:
                return True
            else:
                return False

    def sort_labels(self, labeled_mask, n_labels=None, by='bbox_size'):
        if n_labels is None:
            n_labels = labeled_mask.max()
        res = []
        for label in range(1, n_labels + 1):
            if by == 'area':
                res.append([label, np.count_nonzero(labeled_mask == label)])
            elif by == 'bbox_size':
                bbox = get_region_bbox(labeled_mask == label)
                area = bbox.height * bbox.width
                res.append([label, area])
            else:
                raise ValueError('{} is invalid.'.format(by))
        res = np.array(res)
        res = res[np.argsort(res[:, 1])[::-1]]
        return res[:, 0].astype(int)

    def run_downsample(self):
        self.orig_shape = self.image.shape
        if self.downsample == 1:
            return
        self.image = skimage.transform.resize(self.image, [x // self.downsample for x in self.image.shape], order=1)

    def run_backsample(self):
        if self.downsample == 1:
            return
        self.final_mask = skimage.transform.resize(self.final_mask, self.orig_shape, order=1)
        self.final_mask = np.where(self.final_mask > 0.5, 1, 0)

    def run(self, *args, **kwargs):
        return


class BubbleSegmentor(Segmentor):

    def __init__(self, downsample=4, *args, **kwargs):
        super().__init__(downsample=downsample, *args, **kwargs)
        self.cell_window_bbox = None
        self.cell_window_mask = None
        self.cell_window_mask_residue = 0

    def run(self, return_original_scale=True):
        self.run_downsample()
        image_blur = ndi.gaussian_filter(self.image, 2 / self.downsample)
        thresh = skimage.filters.threshold_otsu(image_blur)
        thresh_mask = (self.image > thresh).astype(int)
        ndi.binary_closing(thresh_mask, structure=np.ones([5, 5]))
        self.final_mask = np.zeros_like(thresh_mask)
        labeled_thresh_mask, n_labels = ndi.label(thresh_mask)
        labels_sorted = self.sort_labels(labeled_thresh_mask, n_labels, by='bbox_size')
        # labels_sorted = self.process_sorted_labels(labels_sorted, labeled_thresh_mask)

        for i, label in enumerate(tqdm.tqdm(labels_sorted)):
            mask_this_label = labeled_thresh_mask == label
            if i == 0:
                self.final_mask += mask_this_label
                self.cell_window_bbox = self.estimate_cell_window_bbox(mask_this_label)
                self.cell_window_mask = self.estimate_cell_window_mask(mask_this_label)
            if not self.should_exclude(mask_this_label):
                self.final_mask += mask_this_label
        if return_original_scale:
            self.run_backsample()
        return self.final_mask

    def process_sorted_labels(self, labels_sorted, labeled_thresh_mask):
        # Make sure the first label is not rounder than the second one. Otherwise, swap them.
        pvr1 = self.calculate_region_principal_axis_eigenvalue_ratio(labeled_thresh_mask == labels_sorted[0])
        pvr2 = self.calculate_region_principal_axis_eigenvalue_ratio(labeled_thresh_mask == labels_sorted[1])
        logger.info('Principal eigenvalue ratios of the first 2 regions: {}, {}'.format(pvr1, pvr2))
        if pvr1 < pvr2:
            logger.info('Swapping the first 2 sorted labels...'.format(pvr1, pvr2))
            temp = labels_sorted[0]
            labels_sorted[0] = labels_sorted[1]
            labels_sorted[1] = temp
        return labels_sorted

    def estimate_cell_window_bbox(self, mask):
        bbox = get_region_bbox(mask)
        window_radius = max(bbox.height, bbox.width) / 2
        if bbox.height < bbox.width:
            bbox.set_sy(int(np.round(bbox.ey - 2 * window_radius)))
        return bbox

    def estimate_cell_window_mask(self, mask):
        """
        Calculate a circular mask given a partial mask.

        :param mask: np.ndarray.
        :return: np.ndarray.
        """
        bbox = self.estimate_cell_window_bbox(mask)
        mask_edge = self.get_edge_pixel_mask(mask)
        mask_edge[bbox.sy:int(bbox.sy + 0.7 * bbox.height), bbox.sx:bbox.ex] = 0
        mask_edge = self.remove_kinks_on_arc(mask_edge)
        y, x = np.where(mask_edge > 0)
        y = y[::40 // self.downsample]
        x = x[::40 // self.downsample]
        if len(y) < 3:
            raise ValueError('There are not enough pixels for fitting circle. Is the window area too small?')
        yc, xc, r = fit_circle(np.stack([y, x], axis=1))
        y, x = np.mgrid[:mask.shape[0], :mask.shape[1]]
        circ_mask = (y - yc) ** 2 + (x - xc) ** 2 <= r ** 2
        self.cell_window_mask_residue = np.mean((circ_mask.astype(float) - mask.astype(float)) ** 2)
        return circ_mask

    def should_exclude(self, mask):
        # bbox = get_region_bbox(mask)
        # if self.cell_window_bbox.is_isolated_from(bbox):
        if not self.intersects_with_cell_window_mask(mask):
            return True
        if np.count_nonzero(mask) < 500 / self.downsample ** 2:
            return True
        if self.is_round(mask, method='circle_fit'):
            return True
        return False

    def intersects_with_cell_window_mask(self, mask):
        if np.count_nonzero(mask * self.cell_window_mask) > 0:
            return True
        else:
            return False

    def calculate_region_principal_axis_eigenvalue_ratio(self, mask):
        pca = sklearn.decomposition.PCA(n_components=2)
        y_inds, x_inds = np.nonzero(mask)
        x = np.stack([y_inds, x_inds], axis=-1)
        res = pca.fit(x)
        eigen_value_ratio = res.explained_variance_ratio_[0] / res.explained_variance_ratio_[1]
        return eigen_value_ratio


class CapillarySegmentor(Segmentor):

    def __init__(self, downsample=4, *args, **kwargs):
        super().__init__(downsample=downsample, *args, **kwargs)
        self.safety_margin = (20 // self.downsample, 40 // self.downsample)
        self.roi_bbox = BoundingBox([0, 0, 1, 1])
        self.estimated_width = 800 // self.downsample

    def run(self, return_original_scale=True):
        self.run_downsample()
        self.determine_x_range()
        self.determine_y_range()

        self.final_mask = self.roi_bbox.generate_mask(self.image.shape)
        return self.final_mask

    def determine_x_range(self):
        std_y = np.std(self.image, axis=0)
        std_y = ndi.gaussian_filter(std_y, 2)
        grad_std_y = ndi.gaussian_filter(std_y, 1, order=1)
        grad_peaks, _ = scipy.signal.find_peaks(grad_std_y, height=grad_std_y.max() * 0.1)
        loc_window = find_window_location_with_most_peaks(self.estimated_width, grad_peaks)
        peaks_in_window = grad_peaks[np.logical_and(
            grad_peaks >= loc_window, grad_peaks <= loc_window + self.estimated_width)]
        x_st, x_end = peaks_in_window.min(), peaks_in_window.max()

        x_st += self.safety_margin[1]
        x_end -= self.safety_margin[1]
        self.roi_bbox.set_sx(x_st)
        self.roi_bbox.set_ex(x_end)

    def determine_y_range(self):
        img_cropped = self.image[:, self.roi_bbox.sx:self.roi_bbox.ex]
        std_x = np.std(img_cropped, axis=1)
        std_x = ndi.gaussian_filter(std_x, 2)
        grad_std_x = ndi.gaussian_filter(std_x, 1, order=1)
        grad_std_x = np.abs(grad_std_x)
        grad_peaks, _ = scipy.signal.find_peaks(grad_std_x, height=grad_std_x.max() * 0.2)
        sy = grad_peaks[-1]
        sy += self.safety_margin[0]
        self.roi_bbox.set_sy(sy)
        self.roi_bbox.set_ey(self.image.shape[0])

