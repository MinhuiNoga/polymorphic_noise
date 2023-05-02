import numpy as np
import pandas as pd

from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
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


# ========sex轉成0 or 1============

def replace_sex(x):
    return x - 1


df_csv["Sex"] = df_csv["Sex"].apply(replace_sex)

# ========df(nan ==> 0)============

df_csv["PPD"].fillna(0, inplace=True)
df_csv["Voice handicap index - 10"].fillna(0, inplace=True)


# ==============normalization================

def replace_nor_age(num):
    return num / 50


def replace_nor_index(num1):
    return num1 / 40


df_csv["Voice handicap index - 10"] = df_csv["Voice handicap index - 10"].apply(replace_nor_index)

df_csv["Age"] = df_csv["Age"].apply(replace_nor_age)

y = df_csv["Disease category"].to_numpy().reshape(-1, 1)
print("資料label總數(矩陣): ", y.shape)
df_csv_1 = df_csv.drop(["Disease category", "ID"], axis=1)

x = df_csv_1.to_numpy()

y_one_hot = np_utils.to_categorical(y)

y_one_hot = np.argmax(y_one_hot, axis=1)

print(y_one_hot)

# =============svm預測===============


x_train, x_test, y_train, y_test = train_test_split(x, y_one_hot, test_size=0.8, random_state=42)

clf = OneVsOneClassifier(SVC(kernel="linear"))

clf.fit(x_train, y_train)

y_pred = clf.predict(x)

# =============把資料丟入第2階段==============

medical_x = y_pred.reshape(-1, 1)

# =============處理音訊檔=====================


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

print(x_autoi)
