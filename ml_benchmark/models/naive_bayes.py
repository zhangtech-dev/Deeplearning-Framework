"""
朴素贝叶斯模型

包括：GaussianNB, MultinomialNB, BernoulliNB
"""

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from .base_model import MLModelBase
from typing import Dict, Any


class GaussianNBModel(MLModelBase):
    """高斯朴素贝叶斯"""

    def __init__(self):
        super().__init__("GaussianNB", "classification")

    def build_model(self, **params):
        default_params = {}
        default_params.update(params)
        self.model = GaussianNB(**default_params)
        return self.model

    def get_param_space(self):
        return {
            'var_smoothing': [1e-9, 1e-8, 1e-7],
        }


class MultinomialNBModel(MLModelBase):
    """多项式朴素贝叶斯"""

    def __init__(self):
        super().__init__("MultinomialNB", "classification")

    def build_model(self, **params):
        default_params = {
            'alpha': 1.0,
        }
        default_params.update(params)
        self.model = MultinomialNB(**default_params)
        return self.model

    def get_param_space(self):
        return {
            'alpha': [0.1, 0.5, 1.0, 2.0],
        }


class BernoulliNBModel(MLModelBase):
    """伯努利朴素贝叶斯"""

    def __init__(self):
        super().__init__("BernoulliNB", "classification")

    def build_model(self, **params):
        default_params = {
            'alpha': 1.0,
            'binarize': 0.0,
        }
        default_params.update(params)
        self.model = BernoulliNB(**default_params)
        return self.model

    def get_param_space(self):
        return {
            'alpha': [0.1, 0.5, 1.0, 2.0],
            'binarize': [0.0, 0.5, 1.0],
        }
