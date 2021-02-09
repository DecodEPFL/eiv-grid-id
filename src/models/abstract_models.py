from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np

"""
    Abstract models for identification classes
    
    Provides abstract classes to have a clear view of functions and parameters
    The following identification methods are implemented:
        - unweighted
        - Latency (y) weighted
        - fully weighted
    Models for power grids and cross validation are also provided
    
    Copyright @donelef on GitHub
"""


@dataclass
class IterationStatus:
    iteration: int
    fitted_parameters: np.array
    target_function: float


@dataclass
class CVTrialResult:
    hyperparameters: dict
    fitted_parameters: np.array


class UnweightedModel(ABC):
    @abstractmethod
    def fit(self, x: np.array, z: np.array):
        pass


class LatencyWeightedModel(ABC):
    @abstractmethod
    def fit(self, x: np.array, z: np.array, z_weight: np.array):
        pass


class MisfitWeightedModel(ABC):
    @abstractmethod
    def fit(self, x: np.array, z: np.array, x_weight: np.array, z_weight: np.array, y_init: np.array):
        pass


class GridIdentificationModel(ABC):

    def __init__(self):
        super().__init__()
        self._admittance_matrix = None

    @property
    def fitted_admittance_matrix(self) -> np.array:
        return self._admittance_matrix


class CVModel(ABC):

    def __init__(self, true_admittance, metric_func):
        super().__init__()
        self._cv_trials = None
        self._true_admittance = true_admittance
        self._metric_func = metric_func

    @property
    def cv_trials(self) -> List[CVTrialResult]:
        return self._cv_trials

    @property
    def best_trial(self) -> CVTrialResult:
        index, _ = min(enumerate(self.cv_trials),
                       key=lambda t: self._metric_func(self._true_admittance, t[1].fitted_parameters))
        return self.cv_trials[index]
