import sklearn
from sklearn.discriminant_analysis import StandardScaler
import sklearn.linear_model
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xailib.explainers.lime_explainer import LimeXAITabularExplainer
from xailib.explainers.shap_explainer_tab import ShapXAITabularExplainer
from xailib.models.sklearn_classifier_wrapper import sklearn_classifier_wrapper
import pandas as pd
import numpy as np

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE


def select_explainer(name: str):
    if name == "shap":
        return shap_fn
    elif name == "lime":
        return lime_fn
    elif name == "lore":
        return lore_fn
    elif name == "anchors":
        return anchors_fn
    else:
        raise ValueError("Invalid explanation algorithm")


def lime_fn(model, X_train, y_train, X_test, y_test):
    bbox = sklearn_classifier_wrapper(model)

    limeExplainer = LimeXAITabularExplainer(bbox)
    config = {"feature_selection": "lasso_path"}
    lime_df = pd.concat([X_train, y_train], axis=1)
    limeExplainer.fit(lime_df, "Attrition_Flag", config)

    # Create new dataset with the features of lime from the test set
    explained_entries = []
    for entry, y_gt in zip(X_test.values, y_test.values):
        exp_list = limeExplainer.explain(entry).exp.as_list()
        dict_entry = {}
        for exp in exp_list:
            dict_entry[exp[0]] = exp[1]
        y_hat = model.predict([entry])
        # Predict the entry if y_gt different from the prediction
        dict_entry["Mismatch"] = (y_gt != y_hat).item()
        explained_entries.append(dict_entry)

    # Create dataframe
    return pd.DataFrame(explained_entries)


def shap_fn(model, X_train, _, X_test, y_test):
    bbox = sklearn_classifier_wrapper(model)

    shap_explainer = ShapXAITabularExplainer(bbox, X_train.columns)
    config = {"explainer": "tree", "X_train": X_train.values}
    shap_explainer.fit(config)
    explained_entries = []
    for entry, y_gt in zip(X_test.values, y_test.values):
        dict_entry = {}
        shap_exp_list = shap_explainer.explain(entry).exp
        for i, exp in enumerate(shap_exp_list):
            if isinstance(exp, np.ndarray):
                dict_entry["shap_" + X_test.columns[i]] = exp[0]
            else:
                dict_entry["shap_" + X_test.columns[i]] = exp

        y_hat = model.predict([entry])
        # Predict the entry if y_gt different from the prediction
        dict_entry["Mismatch"] = (y_gt != y_hat).item()
        explained_entries.append(dict_entry)

    return pd.DataFrame(explained_entries)


def lore_fn():
    pass


def anchors_fn():
    pass


def meta_explain(explained_df):
    explained_df = explained_df.fillna(0)

    # Split the dataset into X_explained a Y_explained (All features except Mismatch, and mismatch)
    X_explained = explained_df.copy().drop(columns=["Mismatch"])
    Y_explained = explained_df.copy()["Mismatch"]
    scaler = StandardScaler()
    X_explained = scaler.fit_transform(X_explained)

    # Split in training and test
    X_explained_train, X_explained_test, Y_explained_train, Y_explained_test = (
        train_test_split(
            X_explained,
            Y_explained,
            test_size=0.2,
            random_state=23,
            stratify=Y_explained,
        )
    )

    smote = SMOTE(sampling_strategy=0.6, random_state=23)
    X_sm, y_sm = smote.fit_resample(X_explained_train, Y_explained_train)

    under = RandomUnderSampler(random_state=23)

    X, y = under.fit_resample(X_sm, y_sm)

    ob = sum(Y_explained_train) / len(Y_explained_train)

    smoteb = y_sm.value_counts() / len(y_sm) * 100

    ub = y.value_counts() / len(y) * 100
    ub = pd.DataFrame(ub).round().reset_index()

    balance_dict = {
        "original": ob,
        "smote": smoteb.to_numpy().tolist(),
        "under": ub.to_numpy().tolist(),
    }

    # WE USE OVER/UNDERSAMPLES DATA
    X_explained_train = X
    Y_explained_train = y

    # Train the model
    meta_model = sklearn.linear_model.LogisticRegressionCV(solver="liblinear")
    meta_model.fit(X_explained_train, Y_explained_train)
    y_predict_explained = meta_model.predict_proba(X_explained_test)[:, 1]
    threshold = 0.5

    metrics = {
        "meta_f1": f1_score(Y_explained_test, y_predict_explained > threshold),
        "meta_recall": recall_score(Y_explained_test, y_predict_explained > threshold),
        "meta_precision": precision_score(
            Y_explained_test, y_predict_explained > threshold
        ),
        "meta_roc_auc": roc_auc_score(Y_explained_test, y_predict_explained),
    }
    metrics.update(balance_dict)
    return metrics
