"""
SVM 模型

包括：SVC, LinearSVC, SVR
"""

from sklearn.svm import SVC, LinearSVC, SVR
from .base_model import MLModelBase
from typing import Dict, Any


class SVCModel(MLModelBase):
    """支持向量分类器"""

    def __init__(self):
        super().__init__("SVC", "classification")

    def build_model(self, **params):
        default_params = {
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale',
            'probability': True,  # 需要 predict_proba
            'random_state': 42,
        }
        default_params.update(params)
        self.model = SVC(**default_params)
        return self.model

    def get_param_space(self):
        return {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto'],
        }


class LinearSVCModel(MLModelBase):
    """线性支持向量分类器"""

    def __init__(self):
        super().__init__("LinearSVC", "classification")

    def build_model(self, **params):
        default_params = {
            'C': 1.0,
            'random_state': 42,
        }
        default_params.update(params)
        self.model = LinearSVC(**default_params)
        return self.model

    def get_param_space(self):
        return {
            'C': [0.1, 1, 10],
        }

    def predict_proba(self, X):
        """LinearSVC 不支持 predict_proba"""
        return None


class SVRModel(MLModelBase):
    """支持向量回归器"""

    def __init__(self):
        super().__init__("SVR", "regression")

    def build_model(self, **params):
        default_params = {
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale',
        }
        default_params.update(params)
        self.model = SVR(**default_params)
        return self.model

    def get_param_space(self):
        return {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto'],
        }
