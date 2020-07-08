from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class CVTrialResult:
    hyperparameters: dict
    fitted_parameters: np.array


class GridIdentificationModel(ABC):

    def __init__(self):
        self._admittance_matrix = None

    @property
    def fitted_admittance_matrix(self) -> np.array:
        return self._admittance_matrix

    @abstractmethod
    def fit(self, x: np.array, y: np.array):
        pass


class GridIdentificationModelCV(GridIdentificationModel, ABC):

    def __init__(self):
        super().__init__()
        self._cv_trials = None

    @property
    def cv_trials(self) -> List[CVTrialResult]:
        return self._cv_trials
