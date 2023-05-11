import numpy as np
import pandas as pd

from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from keras.utils import np_utils

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.multiclass import OneVsOneClassifier

# 資料判斷
df_csv = pd.read_csv("Training Dataset/training datalist.csv")
print("資料資訊")
df_csv.info()


# ========sex轉成0 or 1============

def replace_sex(x):
    return x - 1


df_csv["Sex"] = df_csv["Sex"].apply(replace_sex)

# ========df(nan ==> 0)============

df_csv["PPD"].fillna(0, inplace=True)
df_csv["Voice handicap index - 10"].fillna(0, inplace=True)


# ==============normalization================

def replace_nor_age(x):
    return x / 50


def replace_nor_index(x):
    return x / 40


df_csv["Voice handicap index - 10"] = df_csv["Voice handicap index - 10"].apply(
    replace_nor_index)

df_csv["Age"] = df_csv["Age"].apply(replace_nor_age)

# =============x,y==================

y = df_csv["Disease category"].to_numpy().reshape(-1, 1)
print("資料label總數(矩陣): ", y.shape)
df_csv_1 = df_csv.drop(["Disease category", "ID"], axis=1)

x = df_csv_1.to_numpy()
np.save("med_array", x)
