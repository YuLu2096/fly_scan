import os
import pickle
from functools import wraps

import matplotlib.pyplot as plt
import numpy as np
import torch
import gpytorch

from autobl.util import *
from autobl.steering.guide import *
from autobl.steering.acquisition import *
from autobl.steering.configs import *


class Analyzer:

    def __init__(self, *args, **kwargs):
        pass


class ScanningExperimentAnalyzer(Analyzer):

    def __init__(self, configs: ExperimentAnalyzerConfig, guide, data_x, data_y,
                 n_target_measurements=70, n_init_measurements=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configs = configs
        self.name = configs.name
        self.guide = guide
        self.guide_configs = guide.config
        self.fig_conv, self.ax_conv = None, None
        self.n_measured_list = []
        self.metric_list = []
        self.fig_intermediate, self.ax_intermediate = None, None
        self.intermediate_data_dict = {}
        self.i_intermediate_plot = 0
        self.n_target_measurements = n_target_measurements
        self.n_init_measurements = n_init_measurements
        self.n_pts_measured = 0
        self.data_x = data_x
        self.data_y = data_y
        self.enabled = True

        self.build_dir()
        self.create_intermediate_figure()
        self.create_intermediate_data_dict()
        self.create_convergence_figure_and_data()

    def enable(self, b):
        self.enabled = b

    @staticmethod
    def set_enabled(func):
        @wraps(func)
        def wrapper(obj, *args, **kwargs):
            if obj.enabled:
                func(obj, *args, **kwargs)
            else:
                pass
        return wrapper

    def increment_n_points_measured(self, by):
        self.n_pts_measured += by

    def plot_data(self, additional_x=None, additional_y=None):
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        ax.plot(to_numpy(self.data_x), to_numpy(self.data_y))
        if additional_x is not None:
            ax.scatter(to_numpy(additional_x), to_numpy(additional_y))
        plt.show()

    @set_enabled
    def build_dir(self):
        if not os.path.exists(self.configs.output_dir):
            os.makedirs(self.configs.output_dir)

    @set_enabled
    def create_intermediate_figure(self):
        n_plots = int((self.n_target_measurements - self.n_init_measurements + 1) // self.configs.n_plot_interval + 1)
        n_cols = 3
        n_rows = int(np.ceil(n_plots / n_cols))
        self.fig_intermediate, self.ax_intermediate = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3),
                                                                   squeeze=False)

    @set_enabled
    def update_intermediate_figure(self):
        if not self.n_pts_measured % self.configs.n_plot_interval == 0:
            return
        n_rows = len(self.ax_intermediate)
        n_cols = len(self.ax_intermediate[0])
        self.guide.plot_posterior(
            self.data_x, ax=self.ax_intermediate[self.i_intermediate_plot // n_cols][self.i_intermediate_plot % n_cols])
        self.ax_intermediate[self.i_intermediate_plot // n_cols][self.i_intermediate_plot % n_cols].plot(
            to_numpy(self.data_x), self.data_y, label='Truth', color='gray', alpha=0.6, linewidth=1)
        self.ax_intermediate[self.i_intermediate_plot // n_cols][self.i_intermediate_plot % n_cols].set_title(
            '{} points'.format(self.n_pts_measured))
        self.ax_intermediate[self.i_intermediate_plot // n_cols][self.i_intermediate_plot % n_cols].legend()
        self.i_intermediate_plot += 1

    @set_enabled
    def save_intermediate_figure(self):
        self.fig_intermediate.savefig(os.path.join(self.configs.output_dir,
                                                   self.get_save_name_prefix() + '_intermediate.pdf'),
                                      bbox_inches='tight')

    @set_enabled
    def create_intermediate_data_dict(self):
        self.intermediate_data_dict = {
            'data_x': to_numpy(self.data_x),
            'data_y': to_numpy(self.data_y),
            'measured_x_list': [],
            'measured_y_list': [],
            'n_measured_list': [],
            'mu_list': [],
            'sigma_list': []
        }

    @set_enabled
    def update_intermediate_data_dict(self):
        mu, sigma = self.guide.get_posterior_mean_and_std(
            self.data_x[:, None],
            use_spline_interpolation_for_mean=self.guide_configs.use_spline_interpolation_for_posterior_mean
        )
        mu = mu.squeeze()
        sigma = sigma.squeeze()
        measured_x, measured_y = self.guide.untransform_data(x=self.guide.data_x, y=self.guide.data_y)
        self.intermediate_data_dict['n_measured_list'].append(self.n_pts_measured)
        self.intermediate_data_dict['mu_list'].append(to_numpy(mu))
        self.intermediate_data_dict['sigma_list'].append(to_numpy(sigma))
        self.intermediate_data_dict['measured_x_list'].append(to_numpy(measured_x.squeeze()))
        self.intermediate_data_dict['measured_y_list'].append(to_numpy(measured_y.squeeze()))

    @set_enabled
    def save_intermediate_data_dict(self):
        fname = self.get_save_name_prefix()
        fname = fname + '_intermediate_data.pkl'
        fname = os.path.join(self.configs.output_dir, fname)
        pickle.dump(self.intermediate_data_dict, open(fname, 'wb'))

    @set_enabled
    def create_convergence_figure_and_data(self):
        self.fig_conv, self.ax_conv = plt.subplots(1, 1)

    @set_enabled
    def update_convergence_data(self):
        mu, _ = self.guide.get_posterior_mean_and_std(
            self.data_x[:, None],
            use_spline_interpolation_for_mean=self.guide_configs.use_spline_interpolation_for_posterior_mean
        )
        mu = mu.squeeze()
        metric = rms(mu.detach().cpu().numpy(), self.data_y)
        self.n_measured_list.append(self.n_pts_measured)
        self.metric_list.append(metric)

    @set_enabled
    def plot_convergence(self):
        self.ax_conv.plot(self.n_measured_list, self.metric_list)
        self.ax_conv.set_xlabel('Points measured')
        self.ax_conv.set_ylabel('RMS')

    @set_enabled
    def save_convergence_figure_and_data(self):
        self.fig_conv.savefig(os.path.join(self.configs.output_dir, self.get_save_name_prefix() + '_conv.pdf'))
        np.savetxt(os.path.join(self.configs.output_dir, self.get_save_name_prefix() + '_conv.txt'),
                   np.stack([self.n_measured_list, self.metric_list]))

    def get_save_name_prefix(self):
        if isinstance(self.guide, GPExperimentGuide):
            acquisition_info = self.guide_configs.acquisition_function_class.__name__
            if self.guide_configs.acquisition_function_class in [GradientAwarePosteriorStandardDeviation,
                                                           FittingResiduePosteriorStandardDeviation]:
                acquisition_info += '_phi_{}'.format(self.guide.acquisition_function.phi)
            if self.guide_configs.acquisition_function_class == ComprehensiveAugmentedAcquisitionFunction:
                acquisition_info += '_gradOrder_{}_phiG_{}_phiR_{}'.format(self.guide.acquisition_function.gradient_order,
                                                                           self.guide.acquisition_function.phi_g,
                                                                           self.guide.acquisition_function.phi_r)
                if self.guide.acquisition_function.gradient_order == 2:
                    acquisition_info += '_phiG2_{}'.format(self.guide.acquisition_function.phi_g2)

            kernel_info = '{}_lengthscale_{:.3f}'.format(self.guide.model.covar_module.__class__.__name__,
                                                         self.guide.unscale_by_normalizer_bounds(
                                                             self.guide.model.covar_module.lengthscale.item()
                                                         ))
            if isinstance(self.guide.model.covar_module, gpytorch.kernels.MaternKernel):
                kernel_info += '_nu_{}'.format(self.guide.model.covar_module.nu)

            optimizer_info = self.guide_configs.optimizer_class.__name__

            save_name_prefix = '_'.join([self.name, acquisition_info, kernel_info, optimizer_info])
        else:
            save_name_prefix = '_'.join([self.name, 'UnknownExperimentGuide'])
        return save_name_prefix

    def update_analysis(self):
        self.update_intermediate_data_dict()
        self.update_intermediate_figure()
        self.update_convergence_data()

    def save_analysis(self):
        if self.configs.save:
            self.plot_convergence()
            self.save_intermediate_figure()
            self.save_convergence_figure_and_data()
            self.save_intermediate_data_dict()
        if self.configs.show:
            plt.show()
        plt.close('all')
