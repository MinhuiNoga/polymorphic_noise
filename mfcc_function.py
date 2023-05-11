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

df_csv = pd.read_csv("Training Dataset/training datalist.csv")

df_csv_voice = df_csv.loc[df_csv['Disease category'].isin([1, 2, 3, 4, 5]), ['ID', 'Disease category']]

df_csv_voice['wav_path'] = df_csv_voice['ID'].map("./Training Dataset/training_voice_data/{}.wav".format)

print("Disease category in source_df :", df_csv_voice['Disease category'].unique())
print("source_df :\n", df_csv_voice)


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

for i, voice_id in enumerate(voice_train_id):
    mfccs_feature = audio_to_mfccs(df_csv_voice[df_csv_voice["ID"] == voice_id]["wav_path"].values[0])
    df = pd.DataFrame()
    for j in range(26):
        df_i = pd.DataFrame(np.array(mfccs_feature[0][j]).reshape(1, -1))
        df = pd.concat([df, df_i], axis=1)

    label = df_csv_voice[df_csv_voice["ID"] == voice_id]["Disease category"].values[0]

    # one-hot編碼
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

    train_data = pd.concat([train_data, df])

x_autoi = train_data.iloc[:, :-5]
voice_x = x_autoi.values

# ================標準化及歸一化

mean = np.mean(voice_x, axis=1)
std = np.std(voice_x, axis=1)
min = np.min(voice_x, axis=1)
max = np.max(voice_x, axis=1)

scaler = StandardScaler()
mfcc_standardized = scaler.fit_transform(voice_x.T).T

scaler = MinMaxScaler()
mfcc_normalized = scaler.fit_transform(voice_x.T).T

mfcc_mix = np.concatenate((mfcc_standardized, mfcc_normalized), axis=1)

np.save("mfcc_array", mfcc_mix)


