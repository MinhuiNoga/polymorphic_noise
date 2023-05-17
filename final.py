import numpy as np
import pandas as pd
import librosa

import tensorflow as tf
from keras.layers import Activation, BatchNormalization, Dense, LayerNormalization
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score

source_df = pd.read_csv('Training Dataset/training datalist.csv')
test_df = pd.read_csv('Private Testing Dataset/test_datalist_private.csv')


def medical_data_proccessing(df):
    # 這邊要加入ID  用於轉換資料時對應
    medical_col = ['ID', 'Sex', 'Age', 'Narrow pitch range',
                   'Decreased volume', 'Fatigue', 'Dryness', 'Lumping', 'heartburn',
                   'Choking', 'Eye dryness', 'PND', 'Smoking', 'PPD', 'Drinking',
                   'frequency', 'Diurnal pattern', 'Onset of dysphonia ', 'Noise at work',
                   'Occupational vocal demand', 'Diabetes', 'Hypertension', 'CAD',
                   'Head and Neck Cancer', 'Head injury', 'CVA',
                   'Voice handicap index - 10', 'Disease category']

    df = df.loc[df['Disease category'].isin([1, 2, 3, 4, 5]), medical_col]

    # 將性別編碼0,1
    df['Sex'] = df['Sex'] - 1

    # 將空值填0
    df['PPD'] = df['PPD'].fillna(0)
    df['Voice handicap index - 10'] = df['Voice handicap index - 10'].fillna(0)

    # 正規化過大的數值
    df['Age'] = df['Age'] / 50
    df['Voice handicap index - 10'] = df['Voice handicap index - 10'] / 40

    return df


def medical_data_proccessing_test(df):
    # 這邊要加入ID  用於轉換資料時對應
    medical_col = ['ID', 'Sex', 'Age', 'Narrow pitch range',
                   'Decreased volume', 'Fatigue', 'Dryness', 'Lumping', 'heartburn',
                   'Choking', 'Eye dryness', 'PND', 'Smoking', 'PPD', 'Drinking',
                   'frequency', 'Diurnal pattern', 'Onset of dysphonia ', 'Noise at work',
                   'Occupational vocal demand', 'Diabetes', 'Hypertension', 'CAD',
                   'Head and Neck Cancer', 'Head injury', 'CVA',
                   'Voice handicap index - 10', 'Disease category']

    # 將性別編碼0,1
    df['Sex'] = df['Sex'] - 1

    # 將空值填0
    df['PPD'] = df['PPD'].fillna(0)
    df['Voice handicap index - 10'] = df['Voice handicap index - 10'].fillna(0)

    # 正規化過大的數值
    df['Age'] = df['Age'] / 50
    df['Voice handicap index - 10'] = df['Voice handicap index - 10'] / 40

    return df


training_df = medical_data_proccessing(source_df)
training_df['wav_path'] = training_df['ID'].map("./Training Dataset/training_voice_data/{}.wav".format)

print("Disease category in training_df :", training_df['Disease category'].unique())
print("training_df col :\n", training_df.columns)
print("training_df shape :", training_df.shape)

test_df = medical_data_proccessing_test(test_df)
test_df['wav_path'] = test_df['ID'].map("./Private Testing Dataset/test_data_private/{}.wav".format)

print("test_df columns :", test_df.columns)
print("test_df.shape :", test_df.shape)

training_data_medical = training_df.iloc[:, :27]

print("training_data col :\n", training_data_medical.columns)
print("training_data shape :", training_data_medical.shape)

test_data_medical = test_df.iloc[:, :27]

print("Test medical data shape :", test_data_medical.shape)
print("Test medical data columns :", test_data_medical.columns)


def audio_to_mfccs(filename, sample_rate=44100, offset=0, duration=None):
    voice, sample_rate = librosa.load(filename, sr=sample_rate, offset=offset, duration=duration)

    n_fft = int(16 / 1000 * sample_rate)  # Convert 16 ms to samples
    hop_length = int(8 / 1000 * sample_rate)  # Convert 8 ms to samples
    mfcc_feature = librosa.feature.mfcc(y=voice, sr=sample_rate, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)

    delta_mfcc_feature = librosa.feature.delta(mfcc_feature)
    mfccs = np.concatenate((mfcc_feature, delta_mfcc_feature))
    mfccs_features = np.transpose(mfccs)  # all frames

    return mfccs_features


def second_stage_dataproccessing(training_df, acoustic_model, medical_model):
    training_id = training_df['ID'].tolist()
    training_data = pd.DataFrame()
    for id in training_id:
        mfccs_feature = audio_to_mfccs(training_df[training_df['ID'] == id]['wav_path'].values[0])
        mfccs_df = pd.DataFrame(mfccs_feature)
        df = mfccs_df.copy()

        # 取病理資料接續在mfcc特徵後面
        medical_data = training_df[training_df['ID'] == id].iloc[:, 1:27]
        for col_name in medical_data.columns:
            df[col_name] = medical_data[col_name].values[0]

        # 透過聲學模型預測此病人 (By frame)
        frame_pred = acoustic_model.predict(mfccs_df)
        frame_pred_df = pd.DataFrame(frame_pred)
        df = pd.concat([df, frame_pred_df], axis=1)  # 將聲學模型預測結果接在每個frame的特徵後面

        # 透過病理模型預測此病人
        medical_pred = medical_model.predict(medical_data)
        df['medical_pred_c1'] = medical_pred[0][0]
        df['medical_pred_c2'] = medical_pred[0][1]
        df['medical_pred_c3'] = medical_pred[0][2]
        df['medical_pred_c4'] = medical_pred[0][3]
        df['medical_pred_c5'] = medical_pred[0][4]

        label = training_df[training_df['ID'] == id]['Disease category'].values[0]
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

    return training_data


def second_stage_dataproccessing_test(training_df, acoustic_model, medical_model):
    training_id = training_df['ID'].tolist()
    training_data = pd.DataFrame()
    for id in training_id:
        mfccs_feature = audio_to_mfccs(training_df[training_df['ID'] == id]['wav_path'].values[0])
        mfccs_df = pd.DataFrame(mfccs_feature)
        df = mfccs_df.copy()

        # 取病理資料接續在mfcc特徵後面
        medical_data = training_df[training_df['ID'] == id].iloc[:, 1:27]
        for col_name in medical_data.columns:
            df[col_name] = medical_data[col_name].values[0]

        # 透過聲學模型預測此病人 (By frame)
        frame_pred = acoustic_model.predict(mfccs_df)
        frame_pred_df = pd.DataFrame(frame_pred)
        df = pd.concat([df, frame_pred_df], axis=1)  # 將聲學模型預測結果接在每個frame的特徵後面

        # 透過病理模型預測此病人
        medical_pred = medical_model.predict(medical_data)
        df['medical_pred_c1'] = medical_pred[0][0]
        df['medical_pred_c2'] = medical_pred[0][1]
        df['medical_pred_c3'] = medical_pred[0][2]
        df['medical_pred_c4'] = medical_pred[0][3]
        df['medical_pred_c5'] = medical_pred[0][4]

        training_data = pd.concat([training_data, df])

    return training_data


acoustic_model = load_model("swim_LSTM.h5")
medical_model = load_model("swim_med.h5")

training_data = second_stage_dataproccessing(training_df, acoustic_model, medical_model)

print("training_data.shape :", training_data.shape)

x_train = training_data.iloc[:, :-5]
y_train = training_data.iloc[:, -5:]
print("x_train.shape, y_train.shape :", x_train.shape, y_train.shape)
print("y_train.columns :", y_train.columns.tolist())

test_data = second_stage_dataproccessing_test(test_df, acoustic_model, medical_model)

print("test_data.shape :", test_data.shape)

x_val = test_data.iloc[:, :-5]
y_val = test_data.iloc[:, -5:]
print("x_val.shape, y_val.shape :", x_val.shape, y_val.shape)
print("y_val.columns :", y_val.columns.tolist())

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(256, activation="relu", input_shape=(x_train.shape[1],)))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(5, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=1024, epochs=15)

y_pred = []
training_id = test_df['ID'].tolist()
for id in training_id:
    mfccs_feature = audio_to_mfccs(test_df[test_df['ID'] == id]['wav_path'].values[0])
    mfccs_df = pd.DataFrame(mfccs_feature)
    df = mfccs_df.copy()

    # 取病理資料接續在mfcc特徵後面
    medical_data = test_data_medical[test_data_medical['ID'] == id].iloc[:, 1:]
    for col_name in medical_data.columns:
        df[col_name] = medical_data[col_name].values[0]

    # 透過聲學模型預測此病人 (By frame)
    frame_pred = acoustic_model.predict(mfccs_df)
    frame_pred_df = pd.DataFrame(frame_pred)
    df = pd.concat([df, frame_pred_df], axis=1)  # 將聲學模型預測結果接在每個frame的特徵後面

    # 透過病理模型預測此病人
    medical_pred = medical_model.predict(medical_data)
    df['medical_pred_c1'] = medical_pred[0][0]
    df['medical_pred_c2'] = medical_pred[0][1]
    df['medical_pred_c3'] = medical_pred[0][2]
    df['medical_pred_c4'] = medical_pred[0][3]
    df['medical_pred_c5'] = medical_pred[0][4]

    frame_pred = model.predict(df)
    frame_pred_results = frame_pred.argmax(axis=1)
    person_pred = np.array([np.sum(frame_pred_results == 0), np.sum(frame_pred_results == 1),
                            np.sum(frame_pred_results == 2), np.sum(frame_pred_results == 3),
                            np.sum(frame_pred_results == 4)]).argmax()  # 注意!如三類別相同票數，預測會為0

    y_pred.append(person_pred)

y_pred = np.array(y_pred)
print(y_pred.shape)
print(y_pred)
