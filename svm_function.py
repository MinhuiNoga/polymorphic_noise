import tensorflow as tf
import numpy as np
import pandas as pd
from keras.layers import Dropout

from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import librosa
from sklearn.svm import SVC
from keras.utils import np_utils

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.multiclass import OneVsOneClassifier

import numpy as np
import pandas as pd

# 資料判斷
df_csv = pd.read_csv("Training Dataset/training datalist.csv")
print("資料資訊")
df_csv.info()

# 引入test

df_csv_test = pd.read_csv("Public Testing Dataset/test_datalist_public.csv")


# ========sex轉成0 or 1============

def replace_sex(x):
    return x - 1


df_csv["Sex"] = df_csv["Sex"].apply(replace_sex)
df_csv_test["Sex"] = df_csv_test["Sex"].apply(replace_sex)

# ========df(nan ==> 0)============

df_csv["PPD"].fillna(0, inplace=True)
df_csv["Voice handicap index - 10"].fillna(0, inplace=True)

df_csv_test["PPD"].fillna(0, inplace=True)
df_csv_test["Voice handicap index - 10"].fillna(0, inplace=True)


# ==============normalization================

def replace_nor_age(num):
    return num / 50


def replace_nor_index(num1):
    return num1 / 40


df_csv["Voice handicap index - 10"] = df_csv["Voice handicap index - 10"].apply(replace_nor_index)
df_csv["Age"] = df_csv["Age"].apply(replace_nor_age)

df_csv_test["Voice handicap index - 10"] = df_csv_test["Voice handicap index - 10"].apply(replace_nor_index)
df_csv_test["Age"] = df_csv_test["Age"].apply(replace_nor_age)

train_y = df_csv["Disease category"].to_numpy()

df_csv_1 = df_csv.drop(["Disease category", "ID"], axis=1)
train_x = df_csv_1.to_numpy()

df_csv_2 = df_csv_test.drop(["ID"], axis=1)
test_x = df_csv_2.to_numpy()

# =============svm預測===============


clf = OneVsOneClassifier(SVC(kernel="linear"))

clf.fit(train_x, train_y)

y_pred = clf.predict(test_x)

# =============處理音訊檔=====================

df_csv_voice = df_csv.loc[:, ['ID']]
df_csv_test_1 = df_csv_test.loc[:, ['ID']]

df_csv_voice['wav_path'] = df_csv_voice['ID'].map("./Training Dataset/training_voice_data/{}.wav".format)
df_csv_test_1['wav_path'] = df_csv_test_1['ID'].map("./Public Testing Dataset/test_data_public/{}.wav".format)


def audio_to_mfccs(filename, sample_rate=44100, offset=0, duration=None):
    voice, sample_rate = librosa.load(
        filename, sr=sample_rate, offset=offset, duration=duration
    )

    n_fft = int(16 / 1000 * sample_rate)
    hop_length = int(8 / 1000 * sample_rate)

    mfcc_feature = librosa.feature.mfcc(
        y=voice, sr=sample_rate, n_mfcc=13, n_fft=n_fft, hop_length=hop_length
    )

    delta_mfcc_feature = librosa.feature.delta(mfcc_feature)

    mfccs = np.concatenate((mfcc_feature, delta_mfcc_feature))
    mfccs_feature = np.transpose(mfccs)

    return mfccs_feature


voice_train_id = df_csv_voice["ID"].tolist()
train_data = pd.DataFrame()

voice_test_id = df_csv_test_1["ID"].tolist()
test_data = pd.DataFrame()

for i, voice_id in enumerate(voice_test_id):
    mfccs_feature = audio_to_mfccs(df_csv_test_1[df_csv_test_1["ID"] == voice_id]["wav_path"].values[0])
    df = pd.DataFrame()
    for j in range(26):
        df_i = pd.DataFrame(np.array(mfccs_feature[0][j]).reshape(1, -1))
        df = pd.concat([df, df_i], axis=1)
    test_data = pd.concat([test_data, df])

for i, voice_id in enumerate(voice_train_id):
    mfccs_feature = audio_to_mfccs(df_csv_voice[df_csv_voice["ID"] == voice_id]["wav_path"].values[0])
    df = pd.DataFrame()
    for j in range(26):
        df_i = pd.DataFrame(np.array(mfccs_feature[0][j]).reshape(1, -1))
        df = pd.concat([df, df_i], axis=1)
    train_data = pd.concat([train_data, df])

x_autoi = train_data.iloc[:, :]
voice_x = x_autoi.values

voice_x_test = test_data.iloc[:, :]
voice_x_test = voice_x_test.values

print(voice_x.shape)
print(voice_x_test.shape)

# ================標準化及歸一化====

scaler = StandardScaler()
mfcc_standardized = scaler.fit_transform(voice_x.T).T
mfcc_standardized_test = scaler.fit_transform(voice_x_test.T).T

scaler = MinMaxScaler()
mfcc_normalized = scaler.fit_transform(voice_x.T).T
mfcc_normalized_test = scaler.fit_transform(voice_x_test.T).T

mfcc_mix = np.concatenate((mfcc_standardized, mfcc_normalized), axis=1)
mfcc_mix_test = np.concatenate((mfcc_standardized_test, mfcc_normalized_test), axis=1)

mfcc_y = np.load('mfcc_y.npy')
mfcc_y = np.argmax(mfcc_y, axis=1)

mfcc_y = mfcc_y + 1

print(mfcc_y)

clf_1 = OneVsOneClassifier(SVC(kernel="linear"))
clf_1.fit(mfcc_mix, mfcc_y)

voice_x_1 = clf_1.predict(mfcc_mix_test)

print(voice_x_1)

voice_x_1 = voice_x_1 / 5
y_pred = y_pred / 5

voice_x_1 = voice_x_1.reshape(-1, 1)
y_pred = y_pred.reshape(-1, 1)

merge_svm_train = np.concatenate((train_y.reshape(-1, 1), train_y.reshape(-1, 1), train_x, mfcc_mix), axis=1)
merge_svm_test = np.concatenate((voice_x_1, y_pred, test_x, mfcc_mix_test), axis=1)

'''
y轉成one-hot編碼
'''

train_y = train_y - 1

train_y = np_utils.to_categorical(train_y, num_classes=5)

'''

設定recall的function()

'''

from keras.callbacks import Callback


class RecallCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        recall = logs.get("val_call")
        if recall is not None:
            print(f'val_recall: {recall:.4f}')


'''
建立模型並引入
'''

import tensorflow as tf
from sklearn.metrics import classification_report

model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(512, activation="relu", input_dim=70))

model.add(tf.keras.layers.Dense(256, activation="relu"))

model.add(Dropout(0.3))

model.add(tf.keras.layers.Dense(128, activation="relu"))

model.add(Dropout(0.2))

model.add(tf.keras.layers.Dense(64, activation="relu"))

model.add(Dropout(0.1))

model.add(tf.keras.layers.Dense(5, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

history = model.fit(merge_svm_train, train_y, batch_size=20, epochs=20, verbose=1, callbacks=[RecallCallback()])

formal_pred_y = model.predict(merge_svm_test)
predicted_class = np.argmax(formal_pred_y, axis=1)
print("formal_pred_y:", formal_pred_y)
print("formal_pred_y.shape:", formal_pred_y.shape)
print(predicted_class)
