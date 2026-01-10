"""
抽象基类

定义所有机器学习模型的统一接口
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import numpy as np
from sklearn.base import BaseEstimator


class MLModelBase(ABC):
    """
    传统机器学习模型抽象基类

    所有模型类都应该继承这个基类并实现抽象方法
    """

    def __init__(self, model_name: str, task_type: str = "classification"):
        """
        初始化模型包装器

        Args:
            model_name: 模型名称
            task_type: 任务类型，'classification' 或 'regression'
        """
        self.model_name = model_name
        self.task_type = task_type
        self.model: Optional[BaseEstimator] = None
        self.param_space: Dict[str, Any] = {}

    @abstractmethod
    def build_model(self, **params) -> BaseEstimator:
        """
        构建模型实例

        Args:
            **params: 模型参数

        Returns:
            配置好的 sklearn 模型实例
        """
        pass

    @abstractmethod
    def get_param_space(self) -> Dict[str, Any]:
        """
        获取参数搜索空间

        Returns:
            参数空间字典，键为参数名，值为可选值列表
        """
        pass

    def fit(self, X_train, y_train, **kwargs):
        """
        训练模型

        Args:
            X_train: 训练特征
            y_train: 训练标签
            **kwargs: 额外的 fit 参数

        Returns:
            self
        """
        if self.model is None:
            self.model = self.build_model()

        self.model.fit(X_train, y_train, **kwargs)
        return self

    def predict(self, X):
        """
        预测

        Args:
            X: 特征

        Returns:
            预测结果
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用 fit() 方法")

        return self.model.predict(X)

    def predict_proba(self, X):
        """
        预测概率（仅分类任务）

        Args:
            X: 特征

        Returns:
            预测概率，如果模型不支持则返回 None
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用 fit() 方法")

        if self.task_type == "classification" and hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        return None

    def get_model_params(self) -> Dict[str, Any]:
        """
        获取当前模型的参数

        Returns:
            模型参数字典
        """
        if self.model is None:
            return {}
        return self.model.get_params()
