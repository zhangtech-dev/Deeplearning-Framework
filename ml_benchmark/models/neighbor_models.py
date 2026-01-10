"""
K 近邻模型

包括：KNeighborsClassifier, KNeighborsRegressor
"""

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from .base_model import MLModelBase
from typing import Dict, Any


class KNeighborsClassifierModel(MLModelBase):
    """K 近邻分类器"""

    def __init__(self):
        super().__init__("KNeighborsClassifier", "classification")

    def build_model(self, **params):
        default_params = {
            'n_neighbors': 5,
            'weights': 'uniform',
            'algorithm': 'auto',
        }
        default_params.update(params)
        self.model = KNeighborsClassifier(**default_params)
        return self.model

    def get_param_space(self):
        return {
            'n_neighbors': [3, 5, 7, 11],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree'],
        }


class KNeighborsRegressorModel(MLModelBase):
    """K 近邻回归器"""

    def __init__(self):
        super().__init__("KNeighborsRegressor", "regression")

    def build_model(self, **params):
        default_params = {
            'n_neighbors': 5,
            'weights': 'uniform',
            'algorithm': 'auto',
        }
        default_params.update(params)
        self.model = KNeighborsRegressor(**default_params)
        return self.model

    def get_param_space(self):
        return {
            'n_neighbors': [3, 5, 7, 11],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree'],
        }
