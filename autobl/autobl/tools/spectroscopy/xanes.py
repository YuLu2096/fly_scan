from argparse import ArgumentParser
from typing import Optional, Tuple, Sequence

import numpy as np
import scipy


class XANESNormalizer:
    
    def __init__(self, fit_ranges=None, edge_loc=None, normalization_order=2) -> None:
        self.p_pre = None
        self.p_post = None
        self.edge_loc = edge_loc
        self.fit_ranges = fit_ranges
        self.normalization_order = normalization_order
        self.edge_height = 1.0
        
    def fit(self, data_x, data_y, fit_ranges=None):
        if fit_ranges is not None:
            self.fit_ranges = fit_ranges
        self.detilt_and_normalize(data_x, data_y, recalculate_fits=True)
        
    def apply(self, data_x, data_y):
        data = self.detilt_and_normalize(data_x, data_y, recalculate_fits=False)
        return data
    
    def detilt_and_normalize(self, data_x, data_y, recalculate_fits=False):
        if recalculate_fits:
            if self.fit_ranges is None or self.edge_loc is None:
                self.edge_loc, edge_width = estimate_edge_location_and_width(data_x, data_y)
                range_pre = (data_x[0], self.edge_loc - edge_width * 3)
                range_post = (self.edge_loc + edge_width * 3, data_x[-1])
            else:
                range_pre, range_post = self.fit_ranges
            
            # Fit pre-edge
            self.p_pre = self.fit_segment(data_x, data_y, range_pre, order=1)

            # Fit post-edge
            self.p_post = self.fit_segment(data_x, data_y, range_post, order=self.normalization_order)

        data_corrected = self.normalize_data(data_x, data_y)
        data_corrected = self.flatten_data(data_x, data_corrected, data_is_normalized=True)
        return data_corrected
    
    def normalize_data(self, data_x, data_y):
        data_corrected = data_y - np.poly1d(self.p_pre)(data_x)
        self.edge_height = self.find_edge_height()
        data_corrected = data_corrected / self.edge_height
        return data_corrected
    
    def flatten_data(self, data_x, data_y, data_is_normalized=True):
        diff = np.poly1d(self.p_post)(data_x) - np.poly1d(self.p_pre)(data_x) - self.edge_height
        if data_is_normalized:
            diff = diff / self.edge_height
        mask = data_x > self.edge_loc
        data_y[mask] = data_y[mask] - diff[mask]
        return data_y
    
    def fit_segment(self, data_x, data_y, range, order=1):
        mask = (data_x < range[1]) & (data_x >= range[0])
        x = data_x[mask]
        y = data_y[mask]
        p = np.polyfit(x, y, order)
        return p
    
    def find_edge_height(self):
        mu0 = np.poly1d(self.p_post)(self.edge_loc) - np.poly1d(self.p_pre)(self.edge_loc)
        return mu0
    
    def save_state(self, path):
        np.save(path, self.__dict__)
        
    def load_state(self, path):
        d = np.load(path, allow_pickle=True).item()
        for k in d.keys():
            self.__dict__[k] = d[k]
        

def estimate_edge_location_and_width(data_x: np.ndarray, data_y: np.ndarray, x_dense: Optional[np.ndarray] = None,
                                     return_in_pixel_unit=False) -> Tuple[float | int, float]:
    """
    Estimate the location and width of the absorption edge in XANES.

    :param data_x: ndarray. Energies in eV or normalized unit.
    :param data_y: ndarray.
    :param x_dense: Optional[ndarray]. A dense array of energies to interpolate data on.
    :param return_in_pixel_unit: bool. If True, the unit of results will be pixel.
    :return: peak location and width in the same unit as data_x.
    """
    # Interpolate a dense spectrum.
    if x_dense is None:
        x_dense = np.linspace(data_x[0], data_x[-1], len(data_x) * 10)
    y_dense = scipy.interpolate.CubicSpline(data_x, data_y)(x_dense)

    # Calculate gradient in the dense spectrum.
    grad_y = (y_dense[2:] - y_dense[:-2]) / (x_dense[2:] - x_dense[:-2])
    dense_psize = x_dense[1] - x_dense[0]

    # Find gradient peaks.
    peak_locs, peak_properties = scipy.signal.find_peaks(grad_y, height=grad_y.max() * 0.5, width=1)
    edge_ind = np.argmax(peak_properties['peak_heights'])
    peak_loc = peak_locs[edge_ind]
    peak_width = peak_properties['widths'][edge_ind]

    # Convert peak location and width from pixels to desired unit.
    if not return_in_pixel_unit:
        peak_loc = x_dense[peak_loc]
        peak_width = peak_width * dense_psize
    return peak_loc, peak_width
