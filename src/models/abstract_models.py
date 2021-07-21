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

