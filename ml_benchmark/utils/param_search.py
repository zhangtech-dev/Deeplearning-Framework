"""
参数搜索工具

提供统一的参数搜索接口（Grid Search 和 Random Search）
"""

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from typing import Dict, Any, Tuple, Optional
import numpy as np


class ParamSearcher:
    """
    参数搜索器

    支持 Grid Search 和 Random Search 两种参数搜索方式
    """

    def __init__(
        self,
        model_wrapper,
        search_type: str = 'grid',
        cv: int = 5,
        n_iter: int = 10,
        n_jobs: int = -1,
        verbose: int = 1
    ):
        """
        初始化参数搜索器

        Args:
            model_wrapper: 模型包装器实例
            search_type: 搜索类型，'grid' 或 'random'
            cv: 交叉验证折数
            n_iter: 随机搜索迭代次数
            n_jobs: 并行任务数，-1 表示使用所有 CPU
            verbose: 详细程度
        """
        self.model_wrapper = model_wrapper
        self.search_type = search_type
        self.cv = cv
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.verbose = verbose

    def search(
        self,
        X_train,
        y_train,
        param_space: Optional[Dict[str, Any]] = None,
        scoring: Optional[str] = None
    ) -> Tuple[Dict[str, Any], float, Any]:
        """
        执行参数搜索

        Args:
            X_train: 训练特征
            y_train: 训练标签
            param_space: 参数搜索空间，如果为 None 则使用模型默认的参数空间
            scoring: 评估指标，如果为 None 则根据任务类型自动选择

        Returns:
            (最佳参数, 最佳分数, 最佳模型)
        """
        # 获取参数空间
        if param_space is None:
            param_space = self.model_wrapper.get_param_space()

        # 构建模型
        model = self.model_wrapper.build_model()

        # 确定 scoring
        if scoring is None:
            if self.model_wrapper.task_type == 'classification':
                scoring = 'accuracy'
            else:
                scoring = 'neg_mean_squared_error'

        # 选择搜索器
        if self.search_type == 'grid':
            searcher = GridSearchCV(
                model,
                param_space,
                cv=self.cv,
                scoring=scoring,
                n_jobs=self.n_jobs,
                verbose=self.verbose
            )
        elif self.search_type == 'random':
            searcher = RandomizedSearchCV(
                model,
                param_space,
                n_iter=self.n_iter,
                cv=self.cv,
                scoring=scoring,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                random_state=42
            )
        else:
            raise ValueError(f"不支持的搜索类型: {self.search_type}")

        # 执行搜索
        print(f"🔍 开始 {self.search_type.upper()} 搜索...")
        print(f"📊 参数空间大小: {len(param_space)} 个参数")
        if self.search_type == 'grid':
            # 计算总的参数组合数
            total_combinations = 1
            for param_values in param_space.values():
                total_combinations *= len(param_values)
            print(f"🔢 总参数组合数: {total_combinations}")
        else:
            print(f"🔢 随机采样迭代数: {self.n_iter}")

        searcher.fit(X_train, y_train)

        print(f"✅ 搜索完成！")
        print(f"🏆 最佳参数: {searcher.best_params_}")
        print(f"📈 最佳分数: {searcher.best_score_:.4f}")

        return searcher.best_params_, searcher.best_score_, searcher.best_estimator_
