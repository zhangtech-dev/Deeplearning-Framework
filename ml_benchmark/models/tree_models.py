"""
树模型

包括：DecisionTree, RandomForest, ExtraTrees（分类和回归）
"""

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from .base_model import MLModelBase
from typing import Dict, Any


class DecisionTreeClassifierModel(MLModelBase):
    """决策树分类器"""

    def __init__(self):
        super().__init__("DecisionTreeClassifier", "classification")

    def build_model(self, **params):
        default_params = {
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
        }
        default_params.update(params)
        self.model = DecisionTreeClassifier(**default_params)
        return self.model

    def get_param_space(self):
        return {
            'max_depth': [3, 5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        }


class DecisionTreeRegressorModel(MLModelBase):
    """决策树回归器"""

    def __init__(self):
        super().__init__("DecisionTreeRegressor", "regression")

    def build_model(self, **params):
        default_params = {
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
        }
        default_params.update(params)
        self.model = DecisionTreeRegressor(**default_params)
        return self.model

    def get_param_space(self):
        return {
            'max_depth': [3, 5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        }


class RandomForestClassifierModel(MLModelBase):
    """随机森林分类器"""

    def __init__(self):
        super().__init__("RandomForestClassifier", "classification")

    def build_model(self, **params):
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
        }
        default_params.update(params)
        self.model = RandomForestClassifier(**default_params)
        return self.model

    def get_param_space(self):
        return {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        }


class RandomForestRegressorModel(MLModelBase):
    """随机森林回归器"""

    def __init__(self):
        super().__init__("RandomForestRegressor", "regression")

    def build_model(self, **params):
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
        }
        default_params.update(params)
        self.model = RandomForestRegressor(**default_params)
        return self.model

    def get_param_space(self):
        return {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        }


class ExtraTreesClassifierModel(MLModelBase):
    """极端随机树分类器"""

    def __init__(self):
        super().__init__("ExtraTreesClassifier", "classification")

    def build_model(self, **params):
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
        }
        default_params.update(params)
        self.model = ExtraTreesClassifier(**default_params)
        return self.model

    def get_param_space(self):
        return {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        }


class ExtraTreesRegressorModel(MLModelBase):
    """极端随机树回归器"""

    def __init__(self):
        super().__init__("ExtraTreesRegressor", "regression")

    def build_model(self, **params):
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
        }
        default_params.update(params)
        self.model = ExtraTreesRegressor(**default_params)
        return self.model

    def get_param_space(self):
        return {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        }
