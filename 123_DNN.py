import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from keras.utils import np_utils

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


import tensorflow as tf
from tensorflow.keras.layers import Activation, BatchNormalization, Dense, LayerNormalization
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score


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


y_one_hot = np_utils.to_categorical(y)

# 删除第0列，将 y_one_hot 变成维度为 (800, 5) 的数组
y_one_hot = np.delete(y_one_hot, 0, axis=1)
print(y_one_hot)

# ===============data切分=================
training_x, x_test, training_y, y_test = train_test_split(
    x, y_one_hot, test_size=0.2, random_state=42)
print("training_x shape :", training_x.shape, ", x_test shape :", x_test.shape)
print("training_y shape :", training_y.shape, ", y_test shape :", y_test.shape)


# ================DNN=========================

# 在 Keras 裡面我們可以很簡單的使用 Sequential 的方法建建立一個 Model
model = Sequential()
# 加入 hidden layer-1 of 78 neurons 指定 input_dim 為 26  (有 26 個特徵)
model.add(Dense(78, input_dim=26))
# 使用 'sigmoid' 當作 activation function
model.add(Activation('relu'))
# 加入 hidden layer-2 of 256 neurons
model.add(Dense(52))
# 使用 'sigmoid' 當作 activation function
model.add(Activation('relu'))
# 加入 hidden layer-3 of 128 neurons
model.add(Dense(26))
# 使用 'sigmoid' 當作 activation function
model.add(Activation('relu'))
# 加入 output layer of 10 neurons
model.add(Dense(5))
# 使用 'softmax' 當作 activation function
model.add(Activation('softmax'))


# 定義訓練方式
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# 開始訓練
train_results = model.fit(x=training_x,
                          y=training_y, validation_split=0.1,
                          epochs=10, batch_size=50, verbose=2,
                          callbacks=[EarlyStopping(monitor='val_loss', patience=5, mode='auto'),
                                     ModelCheckpoint("csv_DNN.h5", save_best_only=True)]
                          )


# ==========訓練資料預測結果=========

y_pred = model.predict(df_csv.iloc[:, :-1]).argmax(axis=1)
y_true = df_csv['Disease category'] - 1

results_recall = recall_score(y_true, y_pred, average=None)
print("Training UAR(Unweighted Average Recall) :", results_recall.mean())
ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred)).plot(cmap='Blues')
