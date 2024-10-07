"""
OED Guide

Use an optimal experimental design objective function as a guide for the experiment
"""

import numpy as np
# import jax
# import scipy

from autobl.steering import guide

class OEDGuide(guide.ExperimentGuide):
    """
    Experiment guide using OED objective as acquisition function

    We would like to evaluate the OED objective for proposed measurements
    (step-scans and fly-scans)
    """

    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

    def build(self, *args, **kwargs):
        """Given some data, create the guide
        """

    def suggest(self):
        """Using the guide, generate suggestions
        
        This is the optimization process whereby the next move for the
        experiment is selected from possibilities
        """

    def update(self, *args, **kwargs):
        """Update the guide as more data is acquired
        """

def A_OED(covariance_matrix):
    """A (average) OED"""
    return np.trace(covariance_matrix)
