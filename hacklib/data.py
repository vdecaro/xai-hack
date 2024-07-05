from typing import Tuple
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


DATA_PATH = "/home/decaro/xai-hack/data/credit_card_churn.csv"


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load the data from the csv file."""
    df = pd.read_csv(DATA_PATH)
    df.drop(
        [
            "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
            "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2",
        ],
        axis=1,
        inplace=True,
    )
    _reduce_mem_usage(df)

    # Label Encoding
    le_Education_Level = LabelEncoder()
    le_Gender = LabelEncoder()
    le_Income_Category = LabelEncoder()
    le_Card_Category = LabelEncoder()
    le_Marital_Status = LabelEncoder()

    df["Education_Level_n"] = le_Education_Level.fit_transform(df["Education_Level"])
    df["Income_Category_n"] = le_Income_Category.fit_transform(df["Income_Category"])
    df["Card_Category_n"] = le_Card_Category.fit_transform(df["Card_Category"])
    df["Gender_n"] = le_Gender.fit_transform(df["Gender"])
    df["Marital_Status_n"] = le_Marital_Status.fit_transform(df["Marital_Status"])

    df = df.drop(
        [
            "Education_Level",
            "Income_Category",
            "Card_Category",
            "CLIENTNUM",
            "Gender",
            "Marital_Status",
        ],
        axis=1,
    )
    df["Attrition_Flag"] = df["Attrition_Flag"].map(
        {"Existing Customer": 1, "Attrited Customer": 0}
    )
    cols1 = []
    cols2 = []
    value = []
    matrix = df.corr()
    for i in range(len(matrix.columns)):
        for j in range(i):
            cols1.append(matrix.columns[i])
            cols2.append(matrix.columns[j])
            value.append(matrix.iloc[i, j])
    new_df = pd.DataFrame(
        {"Feature Name 1": cols1, "Feature Name 2": cols2, "Correlation": value}
    )
    new_df.sort_values("Correlation", ascending=False, inplace=True)
    new_df.reset_index(drop=True, inplace=True)
    new_df = new_df[new_df["Correlation"] > 0.85]
    new_df["Feature Name 1"].unique()
    cols1 = []
    matrix = df.corr()
    for i in range(len(matrix.columns)):
        for j in range(i):
            if abs(matrix.iloc[i, j]) >= 0.85:
                cols1.append(matrix.columns[i])
    dropped = df.drop(cols1, axis=1)
    y_data = dropped["Attrition_Flag"]
    x_data = dropped[dropped.columns.difference(["Attrition_Flag"])]
    x_cols = x_data.columns
    scaler = StandardScaler()
    x_data = scaler.fit_transform(x_data)
    x_data = pd.DataFrame(x_data, columns=x_cols)
    X_train, X_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.2, random_state=23, stratify=y_data
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=23)
    clf = RandomForestClassifier(n_jobs=-1)
    rfecv = RFECV(
        estimator=clf, step=1, cv=cv, scoring="recall", n_jobs=-1
    )  # 5-fold cross-validation
    rfecv = rfecv.fit(X_train, y_train)

    print("Optimal number of features :", rfecv.n_features_)
    print("Best features :", X_train.columns[rfecv.support_])
    keep_cols = X_train.columns[rfecv.support_]
    X_train, X_test = X_train[keep_cols], X_test[keep_cols]
    return X_train, X_test, y_train, y_test


def _reduce_mem_usage(train_data: pd.DataFrame):
    """iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.
    """
    start_mem = train_data.memory_usage().sum() / 1024**2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    for col in train_data.columns:
        col_type = train_data[col].dtype

        if col_type != object:
            c_min = train_data[col].min()
            c_max = train_data[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    train_data[col] = train_data[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    train_data[col] = train_data[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    train_data[col] = train_data[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    train_data[col] = train_data[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    train_data[col] = train_data[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    train_data[col] = train_data[col].astype(np.float32)
                else:
                    train_data[col] = train_data[col].astype(np.float64)
        else:
            train_data[col] = train_data[col].astype("category")

    end_mem = train_data.memory_usage().sum() / 1024**2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    return train_data
