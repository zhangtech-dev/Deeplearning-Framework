"""
线性模型

包括：LogisticRegression, SGDClassifier, LinearRegression, Ridge, Lasso, ElasticNet
"""

from sklearn.linear_model import (
    LogisticRegression,
    SGDClassifier,
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)
from .base_model import MLModelBase
from typing import Dict, Any


class LogisticRegressionModel(MLModelBase):
    """Logistic 回归分类器"""

    def __init__(self):
        super().__init__("LogisticRegression", "classification")

    def build_model(self, **params):
        default_params = {
            'C': 1.0,
            'penalty': 'l2',
            'solver': 'lbfgs',
            'max_iter': 1000,
            'random_state': 42,
        }
        default_params.update(params)
        self.model = LogisticRegression(**default_params)
        return self.model

    def get_param_space(self):
        return {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'lbfgs'],
        }


class SGDClassifierModel(MLModelBase):
    """随机梯度下降分类器"""

    def __init__(self):
        super().__init__("SGDClassifier", "classification")

    def build_model(self, **params):
        default_params = {
            'loss': 'log_loss',
            'penalty': 'l2',
            'alpha': 0.0001,
            'max_iter': 1000,
            'random_state': 42,
        }
        default_params.update(params)
        self.model = SGDClassifier(**default_params)
        return self.model

    def get_param_space(self):
        return {
            'loss': ['log_loss', 'modified_huber'],
            'penalty': ['l1', 'l2', 'elasticnet'],
            'alpha': [0.0001, 0.001, 0.01, 0.1],
        }


class LinearRegressionModel(MLModelBase):
    """线性回归"""

    def __init__(self):
        super().__init__("LinearRegression", "regression")

    def build_model(self, **params):
        default_params = {}
        default_params.update(params)
        self.model = LinearRegression(**default_params)
        return self.model

    def get_param_space(self):
        return {
            'fit_intercept': [True, False],
        }


class RidgeModel(MLModelBase):
    """Ridge 回归"""

    def __init__(self):
        super().__init__("Ridge", "regression")

    def build_model(self, **params):
        default_params = {
            'alpha': 1.0,
            'random_state': 42,
        }
        default_params.update(params)
        self.model = Ridge(**default_params)
        return self.model

    def get_param_space(self):
        return {
            'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
        }


class LassoModel(MLModelBase):
    """Lasso 回归"""

    def __init__(self):
        super().__init__("Lasso", "regression")

    def build_model(self, **params):
        default_params = {
            'alpha': 1.0,
            'random_state': 42,
            'max_iter': 1000,
        }
        default_params.update(params)
        self.model = Lasso(**default_params)
        return self.model

    def get_param_space(self):
        return {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
        }


class ElasticNetModel(MLModelBase):
    """ElasticNet 回归"""

    def __init__(self):
        super().__init__("ElasticNet", "regression")

    def build_model(self, **params):
        default_params = {
            'alpha': 1.0,
            'l1_ratio': 0.5,
            'random_state': 42,
            'max_iter': 1000,
        }
        default_params.update(params)
        self.model = ElasticNet(**default_params)
        return self.model

    def get_param_space(self):
        return {
            'alpha': [0.01, 0.1, 1.0],
            'l1_ratio': [0.2, 0.5, 0.8],
        }
