# General
import pandas as pd
import numpy as np
import sklearn
import shap

# Preprocessing
from itertools import cycle
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics
from scipy import stats
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.feature_selection import RFECV, RFE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

# Models
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
    LabelEncoder,
    label_binarize,
)
from sklearn import (
    svm,
    tree,
    linear_model,
    neighbors,
    naive_bayes,
    ensemble,
    discriminant_analysis,
    gaussian_process,
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, plot_importance
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.cluster import KMeans

# Model_Evaluation
from sklearn.metrics import (
    roc_curve,
    auc,
    make_scorer,
    accuracy_score,
    precision_score,
    recall_score,
)
from sklearn.metrics import f1_score, classification_report, average_precision_score

# from sklearn.metrics import plot_confusion_matrix,
from sklearn.metrics import roc_auc_score, precision_recall_curve

# Visualization
import matplotlib.pyplot as plt
import matplotlib.style as style
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# Setting_Parameters
import warnings

warnings.filterwarnings("ignore")
sns.set()
style.use("ggplot")
pd.set_option("display.max_rows", 100)


plt.rcParams["figure.figsize"] = (12, 8)
cmap = LinearSegmentedColormap.from_list(
    "", ["#FFFFFF", "#FFF5BD", "#FF4646", "#E41A1C", "#960018"]
)
DATA_PATH = "/home/decaro/xai-hack/data/credit_card_churn.csv"

from ray import tune
