"""
Benchmark 运行器

负责批量运行多个机器学习模型并记录结果
"""

import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import time
import numpy as np


class MLBenchmarkRunner:
    """
    ML Benchmark 运行器

    支持批量运行多个模型，计算多种评估指标，并保存结果
    """

    def __init__(
        self,
        task_type: str = 'classification',
        models_to_run: Optional[List[str]] = None,
        metrics_to_use: Optional[List[str]] = None,
        save_dir: str = 'ml_results',
        verbose: bool = False,
        save_format: Optional[Dict[str, bool]] = None
    ):
        """
        初始化运行器

        Args:
            task_type: 任务类型，'classification' 或 'regression'
            models_to_run: 要运行的模型名称列表
            metrics_to_use: 要计算的评估指标列表
            save_dir: 结果保存目录
            verbose: 是否输出详细信息（模型参数、训练时间等）
            save_format: 保存格式配置，如 {'csv': True, 'json': True}
        """
        self.task_type = task_type
        self.models_to_run = models_to_run or []
        self.metrics_to_use = metrics_to_use
        self.save_dir = Path(save_dir)
        self.verbose = verbose
        self.save_format = save_format or {'csv': True, 'json': True}

        # 创建保存目录
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 存储结果
        self.results = []

    def run_benchmark(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ):
        """
        运行 benchmark

        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_test: 测试特征
            y_test: 测试标签
        """
        from .models import MODEL_REGISTRY
        from .evaluation.metrics import MetricsCalculator

        print("=" * 60)
        print(f"🚀 开始 ML Benchmark")
        print(f"📊 任务类型: {self.task_type}")
        print(f"🔧 模型数量: {len(self.models_to_run)}")
        print(f"📈 评估指标: {', '.join(self.metrics_to_use)}")
        print("=" * 60)

        for model_name in self.models_to_run:
            if model_name not in MODEL_REGISTRY:
                print(f"⚠️  模型 {model_name} 未在注册表中找到，跳过")
                continue

            print(f"\n{'=' * 60}")
            print(f"🔨 运行模型: {model_name}")
            print(f"{'=' * 60}")

            try:
                # 创建模型实例
                model_class = MODEL_REGISTRY[model_name]
                model_wrapper = model_class()

                # 训练模型
                print("⏳ 训练中...")
                start_time = time.time()
                model_wrapper.fit(X_train, y_train)
                train_time = time.time() - start_time
                print(f"✅ 训练完成，耗时: {train_time:.2f} 秒")

                # 预测
                print("⏳ 预测中...")
                y_pred = model_wrapper.predict(X_test)
                y_proba = model_wrapper.predict_proba(X_test)
                print(f"✅ 预测完成")

                # 计算评估指标
                print("📊 计算评估指标...")
                metrics = MetricsCalculator.calculate(
                    y_test, y_pred, y_proba,
                    task_type=self.task_type,
                    metrics=self.metrics_to_use
                )

                # 构建结果字典
                result = {
                    'model_name': model_name,
                    'timestamp': datetime.now().isoformat(),
                    'task_type': self.task_type,
                    **metrics
                }

                # 如果是 verbose 模式，添加详细信息
                if self.verbose:
                    result['train_time_seconds'] = round(train_time, 3)
                    result['model_params'] = str(model_wrapper.get_model_params())

                # 打印结果
                print(f"📈 评估结果:")
                for metric, value in metrics.items():
                    if value is not None:
                        if isinstance(value, float):
                            print(f"   {metric}: {value:.4f}")
                        else:
                            print(f"   {metric}: {value}")

                self.results.append(result)
                print(f"✅ {model_name} 完成\n")

            except Exception as e:
                print(f"❌ {model_name} 运行失败: {str(e)}")
                # 记录失败信息
                result = {
                    'model_name': model_name,
                    'timestamp': datetime.now().isoformat(),
                    'task_type': self.task_type,
                    'error': str(e)
                }
                self.results.append(result)
                continue

        # 保存结果
        self._save_results()

        # 打印总结
        self._print_summary()

    def _save_results(self):
        """保存结果到文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存 CSV
        if self.save_format.get('csv', False):
            csv_path = self.save_dir / f'benchmark_{timestamp}.csv'
            df = pd.DataFrame(self.results)
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"✅ CSV 结果已保存到: {csv_path}")

        # 保存 JSON
        if self.save_format.get('json', False):
            json_path = self.save_dir / f'benchmark_{timestamp}.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            print(f"✅ JSON 结果已保存到: {json_path}")

    def _print_summary(self):
        """打印结果总结"""
        print("\n" + "=" * 60)
        print("📊 Benchmark 结果总结")
        print("=" * 60)

        # 按主要指标排序
        if self.results and all('error' not in r for r in self.results):
            # 确定主要指标
            if self.task_type == 'classification':
                main_metric = 'accuracy' if 'accuracy' in self.results[0] else 'f1'
            else:
                main_metric = 'r2' if 'r2' in self.results[0] else 'mse'

            # 排序（分类任务越高越好，回归任务看指标）
            reverse = main_metric in ['accuracy', 'f1', 'precision', 'recall', 'r2', 'roc_auc']
            sorted_results = sorted(
                [r for r in self.results if main_metric in r and r[main_metric] is not None],
                key=lambda x: x[main_metric],
                reverse=reverse
            )

            print(f"\n🏆 按 {main_metric} 排序:")
            for i, result in enumerate(sorted_results[:10], 1):  # 只显示前 10 名
                metric_value = result[main_metric]
                if isinstance(metric_value, float):
                    print(f"   {i:2d}. {result['model_name']:30s}: {metric_value:.4f}")
                else:
                    print(f"   {i:2d}. {result['model_name']:30s}: {metric_value}")

        # 统计失败数量
        failed_count = sum(1 for r in self.results if 'error' in r)
        if failed_count > 0:
            print(f"\n⚠️  失败的模型 ({failed_count} 个):")
            for result in self.results:
                if 'error' in result:
                    print(f"   - {result['model_name']}: {result['error']}")

        print("=" * 60)
