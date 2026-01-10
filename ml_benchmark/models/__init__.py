"""
模型注册表

所有传统机器学习模型的统一注册中心
"""

# 导入所有模型类
from .linear_models import (
    LogisticRegressionModel,
    SGDClassifierModel,
    LinearRegressionModel,
    RidgeModel,
    LassoModel,
    ElasticNetModel,
)
from .tree_models import (
    DecisionTreeClassifierModel,
    DecisionTreeRegressorModel,
    RandomForestClassifierModel,
    RandomForestRegressorModel,
    ExtraTreesClassifierModel,
    ExtraTreesRegressorModel,
)
from .ensemble_models import (
    GradientBoostingClassifierModel,
    GradientBoostingRegressorModel,
    XGBoostClassifierModel,
    XGBoostRegressorModel,
    LightGBMClassifierModel,
    LightGBMRegressorModel,
    CatBoostClassifierModel,
    CatBoostRegressorModel,
    AdaBoostClassifierModel,
    AdaBoostRegressorModel,
)
from .svm_models import (
    SVCModel,
    LinearSVCModel,
    SVRModel,
)
from .neighbor_models import (
    KNeighborsClassifierModel,
    KNeighborsRegressorModel,
)
from .naive_bayes import (
    GaussianNBModel,
    MultinomialNBModel,
    BernoulliNBModel,
)

# 模型注册表（映射模型名称到模型类）
MODEL_REGISTRY = {
    # 线性模型
    "LogisticRegression": LogisticRegressionModel,
    "SGDClassifier": SGDClassifierModel,
    "LinearRegression": LinearRegressionModel,
    "Ridge": RidgeModel,
    "Lasso": LassoModel,
    "ElasticNet": ElasticNetModel,

    # 树模型
    "DecisionTreeClassifier": DecisionTreeClassifierModel,
    "DecisionTreeRegressor": DecisionTreeRegressorModel,
    "RandomForestClassifier": RandomForestClassifierModel,
    "RandomForestRegressor": RandomForestRegressorModel,
    "ExtraTreesClassifier": ExtraTreesClassifierModel,
    "ExtraTreesRegressor": ExtraTreesRegressorModel,

    # Boosting 模型
    "GradientBoostingClassifier": GradientBoostingClassifierModel,
    "GradientBoostingRegressor": GradientBoostingRegressorModel,
    "XGBoostClassifier": XGBoostClassifierModel,
    "XGBoostRegressor": XGBoostRegressorModel,
    "LightGBMClassifier": LightGBMClassifierModel,
    "LightGBMRegressor": LightGBMRegressorModel,
    "CatBoostClassifier": CatBoostClassifierModel,
    "CatBoostRegressor": CatBoostRegressorModel,
    "AdaBoostClassifier": AdaBoostClassifierModel,
    "AdaBoostRegressor": AdaBoostRegressorModel,

    # SVM 模型
    "SVC": SVCModel,
    "LinearSVC": LinearSVCModel,
    "SVR": SVRModel,

    # KNN 模型
    "KNeighborsClassifier": KNeighborsClassifierModel,
    "KNeighborsRegressor": KNeighborsRegressorModel,

    # 朴素贝叶斯
    "GaussianNB": GaussianNBModel,
    "MultinomialNB": MultinomialNBModel,
    "BernoulliNB": BernoulliNBModel,
}

__all__ = ["MODEL_REGISTRY"]
