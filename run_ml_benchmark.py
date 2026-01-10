"""
传统机器学习 Benchmark 运行脚本
使用示例：
    python run_ml_benchmark.py
    python run_ml_benchmark.py output.verbose=true
    python run_ml_benchmark.py task_type=regression
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import sys
import os
# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml_benchmark.utils.data_loader import load_ml_data
from ml_benchmark.runner import MLBenchmarkRunner


@hydra.main(config_path="configs", config_name="ml_benchmark", version_base=None)
def main(cfg: DictConfig):
    # 打印配置
    print("=" * 60)
    print("🚀 ML Benchmark 配置")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    # 准备特征和标签
    print("\n⏳ 加载数据中...")
    from Feature_selected import FEATURE
    feature_cols = FEATURE.num_order_feat + FEATURE.cat_order_feat

    # 使用 ML Benchmark 独立的数据加载器
    X_train, y_train, X_test, y_test = load_ml_data(
        cfg,
        feature_cols=feature_cols,
        label_col=FEATURE.label,
        train_ratio=0.8,
        random_state=42
    )
    # 从配置中获取启用的模型
    from ml_benchmark.models import MODEL_REGISTRY
    models_to_run = [
        model_name for model_name, enabled in cfg.models.items()
        if enabled and model_name in MODEL_REGISTRY
    ]
    if not models_to_run:
        print("\n❌ 错误: 没有启用的模型！")
        print("   请在 configs/ml_benchmark/config.yaml 中设置至少一个模型为 true")
        return
    print(f"\n📋 将运行 {len(models_to_run)} 个模型:")
    for i, model in enumerate(models_to_run, 1):
        print(f"   {i:2d}. {model}")
    # 从配置中获取启用的评估指标
    from ml_benchmark.evaluation.metrics import MetricsCalculator
    metrics_to_use = [
        metric for metric, enabled in cfg.metrics.items()
        if enabled and metric in MetricsCalculator.METRIC_REGISTRY
    ]
    if not metrics_to_use:
        print("\n❌ 错误: 没有启用的评估指标！")
        print("   请在 configs/ml_benchmark/config.yaml 中设置至少一个指标为 true")
        return
    print(f"\n📊 将使用 {len(metrics_to_use)} 个评估指标: {', '.join(metrics_to_use)}")
    # 运行 benchmark
    runner = MLBenchmarkRunner(
        task_type=cfg.task_type,
        models_to_run=models_to_run,
        metrics_to_use=metrics_to_use,
        save_dir=cfg.output.save_dir,
        verbose=cfg.output.verbose,
        save_format=cfg.output.save_format
    )
    runner.run_benchmark(X_train, y_train, X_test, y_test)
    print("\n✅ Benchmark 完成！")
    print(f"📁 结果已保存到: {cfg.output.save_dir}")


if __name__ == '__main__':
    main()
