import keras
import numpy as np
import pandas as pd
import librosa
from keras.utils import np_utils

from sklearn.model_selection import train_test_split
from scipy.stats import skew

from sklearn.multiclass import OneVsOneClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score
from keras.utils import pad_sequences

df_csv = pd.read_csv("Training Dataset/training datalist.csv")
print("資料資訊")
df_csv.info()

y = df_csv["Disease category"]

# 挑選出要訓練的Disease category 1、2、3、4、5
df_csv = df_csv.loc[df_csv['Disease category'].isin([1, 2, 3, 4, 5]), ['ID', 'Disease category']]

# 在dataframe中加入要訓練的音檔路徑
df_csv['wav_path'] = df_csv['ID'].map("./Training Dataset/training_voice_data/{}.wav".format)

print("Disease category in source_df :", df_csv['Disease category'].unique())
print("source_df :\n", df_csv["wav_path"])


def audio_to_mfccs(filename, sample_rate=44100, offset=0, duration=None):
    # 讀取音訊檔案，並設定取樣率、起始時間、及持續時間
    voice, sample_rate = librosa.load(
        filename, sr=sample_rate, offset=offset, duration=duration
    )

    # 將時間值轉換為 FFT 與 hop length 所需的框架數 (以取樣點表示)
    n_fft = int(16 / 1000 * sample_rate)  # 將 16 毫秒轉換為取樣點
    hop_length = int(8 / 1000 * sample_rate)  # 將 8 毫秒轉換為取樣點

    # 計算音訊數據的 MFCC 特徵
    mfcc_feature = librosa.feature.mfcc(
        y=voice, sr=sample_rate, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)

    # 計算 MFCC 的一階和二階差分特徵
    delta_mfcc_feature = librosa.feature.delta(mfcc_feature)

    # 將原始 MFCC 特徵和差分特徵串聯起來，得到所有幀的特徵向量
    mfccs = np.concatenate((mfcc_feature, delta_mfcc_feature))
    mfccs_features = np.transpose(mfccs)  # 將矩陣轉置，使每行代表一個幀

    # 返回特徵向量
    return mfccs_features


training_id = df_csv['ID'].tolist()
training_data = pd.DataFrame()

for i, voice_id in enumerate(training_id):
    mfccs_feature = audio_to_mfccs(df_csv[df_csv["ID"] == voice_id]["wav_path"].values[0])
    df = pd.DataFrame()
    for j in range(26):
        df_i = pd.DataFrame(np.array(mfccs_feature[0][j]).reshape(1, -1))
        df = pd.concat([df, df_i], axis=1)

    label = df_csv[df_csv["ID"] == voice_id]["Disease category"].values[0]

    if label == 1:
        df['c1'] = 1
        df['c2'] = 0
        df['c3'] = 0
        df['c4'] = 0
        df['c5'] = 0
    elif label == 2:
        df['c1'] = 0
        df['c2'] = 1
        df['c3'] = 0
        df['c4'] = 0
        df['c5'] = 0
    elif label == 3:
        df['c1'] = 0
        df['c2'] = 0
        df['c3'] = 1
        df['c4'] = 0
        df['c5'] = 0
    elif label == 4:
        df['c1'] = 0
        df['c2'] = 0
        df['c3'] = 0
        df['c4'] = 1
        df['c5'] = 0
    elif label == 5:
        df['c1'] = 0
        df['c2'] = 0
        df['c3'] = 0
        df['c4'] = 0
        df['c5'] = 1
    else:
        df['c1'] = np.nan
        df['c2'] = np.nan
        df['c3'] = np.nan
        df['c4'] = np.nan
        df['c5'] = np.nan

    training_data = pd.concat([training_data, df])

print("training_data.shape :", training_data.shape)

x = training_data.iloc[:, :-5]
x = x.values

print(x)

mean = np.mean(x, axis=1)
std = np.std(x, axis=1)
min = np.min(x, axis=1)
max = np.max(x, axis=1)

scaler = StandardScaler()
mfcc_standardized = scaler.fit_transform(x.T).T

print(mfcc_standardized)

scaler = MinMaxScaler()
mfcc_normalized = scaler.fit_transform(x.T).T

print("正歸化:", mfcc_normalized)
print("x.shape():", mfcc_normalized.shape)

time_step = 10
feature = 26

n_samples = mfcc_normalized.shape[0] - time_step + 1
X = np.zeros((n_samples, time_step, feature))

for i in range(n_samples):
    X[i] = mfcc_normalized[i:i+time_step]


'''
T = 5
x = 1000

time_arrat_data = np.zeros((996, T, 26))
for i in range(5):
    time_arrat_data[i] = mfcc_normalized[i:i + T]

print(time_arrat_data.shape)
'''

y = np.array(y)

y = y.reshape(-1, 1)

y = y[:-9, :]

y = y - 1

y = np_utils.to_categorical(y, num_classes=5)

print("x",X.shape)
print("y",y.shape)

import numpy as np
from keras.utils import np_utils
from keras.layers import Input, Dense, Dropout, Embedding, Concatenate, Flatten
from keras.models import Model
from keras.callbacks import EarlyStopping

