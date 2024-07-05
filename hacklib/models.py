from lightgbm import LGBMClassifier
from sklearn import (
    discriminant_analysis,
    ensemble,
    linear_model,
    naive_bayes,
    neighbors,
    svm,
    tree,
)
from xgboost import XGBClassifier


def select_model(name: str):
    if name == "LogisticRegression":
        return linear_model.LogisticRegression(random_state=30)
    elif name == "RandomForest":
        return ensemble.RandomForestClassifier(random_state=30)
    elif name == "GradientBoosting":
        return ensemble.GradientBoostingClassifier(random_state=30)
    elif name == "AdaBoost":
        return ensemble.AdaBoostClassifier(random_state=30)
    elif name == "ExtraTrees":
        return ensemble.ExtraTreesClassifier(random_state=30)
    elif name == "GaussianNB":
        return naive_bayes.GaussianNB()
    elif name == "KNeighbors":
        return neighbors.KNeighborsClassifier()
    elif name == "SVC":
        return svm.SVC(probability=True, random_state=30)
    elif name == "DecisionTree":
        return tree.DecisionTreeClassifier(random_state=30)
    elif name == "LDA":
        return discriminant_analysis.LinearDiscriminantAnalysis()
    elif name == "QDA":
        return discriminant_analysis.QuadraticDiscriminantAnalysis()
    elif name == "XGB":
        return XGBClassifier(random_state=30)
    elif name == "LGBM":
        return LGBMClassifier(random_state=30)
    else:
        raise ValueError("Invalid model name")
