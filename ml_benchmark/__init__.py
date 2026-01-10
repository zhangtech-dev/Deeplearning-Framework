"""
传统机器学习 Benchmark 模块

提供统一接口运行多种传统机器学习模型，支持参数搜索和批量评估
"""

__version__ = "1.0.0"
__author__ = "Yang Zhou"

from .runner import MLBenchmarkRunner

__all__ = ["MLBenchmarkRunner"]
