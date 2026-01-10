"""
集成模型（Boosting）

包括：GradientBoosting, XGBoost, LightGBM, CatBoost, AdaBoost（分类和回归）
"""

from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    AdaBoostClassifier,
    AdaBoostRegressor
)
from .base_model import MLModelBase
from typing import Dict, Any
import warnings

# 尝试导入 XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost 未安装，XGBoost 模型将不可用")

# 尝试导入 LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    warnings.warn("LightGBM 未安装，LightGBM 模型将不可用")

# 尝试导入 CatBoost
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    warnings.warn("CatBoost 未安装，CatBoost 模型将不可用")


class GradientBoostingClassifierModel(MLModelBase):
    """梯度提升分类器"""

    def __init__(self):
        super().__init__("GradientBoostingClassifier", "classification")

    def build_model(self, **params):
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'random_state': 42,
        }
        default_params.update(params)
        self.model = GradientBoostingClassifier(**default_params)
        return self.model

    def get_param_space(self):
        return {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
        }


class GradientBoostingRegressorModel(MLModelBase):
    """梯度提升回归器"""

    def __init__(self):
        super().__init__("GradientBoostingRegressor", "regression")

    def build_model(self, **params):
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'random_state': 42,
        }
        default_params.update(params)
        self.model = GradientBoostingRegressor(**default_params)
        return self.model

    def get_param_space(self):
        return {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
        }


class XGBoostClassifierModel(MLModelBase):
    """XGBoost 分类器"""

    def __init__(self):
        super().__init__("XGBoostClassifier", "classification")
        if not XGBOOST_AVAILABLE:
            warnings.warn("XGBoost 未安装，此模型无法使用")

    def build_model(self, **params):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost 未安装，请先安装: pip install xgboost")

        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'random_state': 42,
            'eval_metric': 'logloss',
        }
        default_params.update(params)
        self.model = xgb.XGBClassifier(**default_params)
        return self.model

    def get_param_space(self):
        return {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
        }


class XGBoostRegressorModel(MLModelBase):
    """XGBoost 回归器"""

    def __init__(self):
        super().__init__("XGBoostRegressor", "regression")
        if not XGBOOST_AVAILABLE:
            warnings.warn("XGBoost 未安装，此模型无法使用")

    def build_model(self, **params):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost 未安装，请先安装: pip install xgboost")

        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'random_state': 42,
        }
        default_params.update(params)
        self.model = xgb.XGBRegressor(**default_params)
        return self.model

    def get_param_space(self):
        return {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
        }


class LightGBMClassifierModel(MLModelBase):
    """LightGBM 分类器"""

    def __init__(self):
        super().__init__("LightGBMClassifier", "classification")
        if not LIGHTGBM_AVAILABLE:
            warnings.warn("LightGBM 未安装，此模型无法使用")

    def build_model(self, **params):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM 未安装，请先安装: pip install lightgbm")

        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'random_state': 42,
            'verbose': -1,
        }
        default_params.update(params)
        self.model = lgb.LGBMClassifier(**default_params)
        return self.model

    def get_param_space(self):
        return {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
        }


class LightGBMRegressorModel(MLModelBase):
    """LightGBM 回归器"""

    def __init__(self):
        super().__init__("LightGBMRegressor", "regression")
        if not LIGHTGBM_AVAILABLE:
            warnings.warn("LightGBM 未安装，此模型无法使用")

    def build_model(self, **params):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM 未安装，请先安装: pip install lightgbm")

        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'random_state': 42,
            'verbose': -1,
        }
        default_params.update(params)
        self.model = lgb.LGBMRegressor(**default_params)
        return self.model

    def get_param_space(self):
        return {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
        }


class CatBoostClassifierModel(MLModelBase):
    """CatBoost 分类器"""

    def __init__(self):
        super().__init__("CatBoostClassifier", "classification")
        if not CATBOOST_AVAILABLE:
            warnings.warn("CatBoost 未安装，此模型无法使用")

    def build_model(self, **params):
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost 未安装，请先安装: pip install catboost")

        default_params = {
            'iterations': 100,
            'learning_rate': 0.1,
            'depth': 6,
            'random_state': 42,
            'verbose': False,
        }
        default_params.update(params)
        self.model = cb.CatBoostClassifier(**default_params)
        return self.model

    def get_param_space(self):
        return {
            'iterations': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'depth': [4, 6, 8],
        }


class CatBoostRegressorModel(MLModelBase):
    """CatBoost 回归器"""

    def __init__(self):
        super().__init__("CatBoostRegressor", "regression")
        if not CATBOOST_AVAILABLE:
            warnings.warn("CatBoost 未安装，此模型无法使用")

    def build_model(self, **params):
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost 未安装，请先安装: pip install catboost")

        default_params = {
            'iterations': 100,
            'learning_rate': 0.1,
            'depth': 6,
            'random_state': 42,
            'verbose': False,
        }
        default_params.update(params)
        self.model = cb.CatBoostRegressor(**default_params)
        return self.model

    def get_param_space(self):
        return {
            'iterations': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'depth': [4, 6, 8],
        }


class AdaBoostClassifierModel(MLModelBase):
    """AdaBoost 分类器"""

    def __init__(self):
        super().__init__("AdaBoostClassifier", "classification")

    def build_model(self, **params):
        default_params = {
            'n_estimators': 50,
            'learning_rate': 1.0,
            'random_state': 42,
        }
        default_params.update(params)
        self.model = AdaBoostClassifier(**default_params)
        return self.model

    def get_param_space(self):
        return {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 1.0],
        }


class AdaBoostRegressorModel(MLModelBase):
    """AdaBoost 回归器"""

    def __init__(self):
        super().__init__("AdaBoostRegressor", "regression")

    def build_model(self, **params):
        default_params = {
            'n_estimators': 50,
            'learning_rate': 1.0,
            'random_state': 42,
        }
        default_params.update(params)
        self.model = AdaBoostRegressor(**default_params)
        return self.model

    def get_param_space(self):
        return {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 1.0],
        }
