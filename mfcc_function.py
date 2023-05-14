import numpy as np
import pandas as pd
import librosa

import tensorflow as tf
from keras.layers import Activation, BatchNormalization, Dense, LayerNormalization, Dropout, LSTM
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score
from sklearn.utils import shuffle

source_df = pd.read_csv('Training Dataset/training datalist.csv')
print("source_df.shape :", source_df.shape)
print("source_df.columns :", source_df.columns)

source_df = source_df.loc[source_df['Disease category'].isin([1, 2, 3, 4, 5]), ['ID', 'Disease category']]

# 在dataframe中加入要訓練的音檔路徑
source_df['wav_path'] = source_df['ID'].map("./Training Dataset/training_voice_data/{}.wav".format)

print("Disease category in source_df :", source_df['Disease category'].unique())
print("source_df :\n", source_df)

training_df, test_df = train_test_split(source_df, test_size=0.2, random_state=333)

print("training_df shape :", training_df.shape, ", test_df shape :", test_df.shape)


# define function
def audio_to_mfccs(filename, sample_rate=44100, offset=0, duration=None):
    voice, sample_rate = librosa.load(filename, sr=sample_rate, offset=offset, duration=duration)

    n_fft = int(16 / 1000 * sample_rate)  # Convert 16 ms to samples
    hop_length = int(8 / 1000 * sample_rate)  # Convert 8 ms to samples
    mfcc_feature = librosa.feature.mfcc(y=voice, sr=sample_rate, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)

    delta_mfcc_feature = librosa.feature.delta(mfcc_feature)

    mfccs = np.concatenate((mfcc_feature, delta_mfcc_feature))
    mfccs_features = np.transpose(mfccs)  # all frames

    return mfccs_features


training_id = training_df['ID'].tolist()
training_data = pd.DataFrame()
for id in training_id:
    mfccs_feature = audio_to_mfccs(training_df[training_df['ID'] == id]['wav_path'].values[0])
    df = pd.DataFrame(mfccs_feature)
    # print("id :",id, ", number of frames :", df.shape[0])

    # 訓練資料標記
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

print("training_data.shape :", training_data.shape)

x_train = training_data.iloc[:, :-5]
y_train = training_data.iloc[:, -5:]
print("x_train.shape, y_train.shape :", x_train.shape, y_train.shape)
print("y_train.columns :", y_train.columns.tolist())

X_train = tf.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
Y = y_train.values

test_id = test_df['ID'].tolist()
test_data = pd.DataFrame()
for id in test_id:
    mfccs_feature = audio_to_mfccs(test_df[test_df['ID'] == id]['wav_path'].values[0])
    df = pd.DataFrame(mfccs_feature)
    label = test_df[test_df['ID'] == id]['Disease category'].values[0]
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

    test_data = pd.concat([test_data, df])

print("training_data.shape :", test_data.shape)

x_val = test_data.iloc[:, :-5]
y_val = test_data.iloc[:, -5:]
print("x_val.shape, y_val.shape :", x_val.shape, y_val.shape)
print("y_val.columns :", y_val.columns.tolist())

model = tf.keras.Sequential()
model = Sequential()
model.add(LSTM(64, input_shape=(26, 1), return_sequences=True))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(16))
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
history = model.fit(X_train, Y, batch_size=30000, epochs=20, validation_data=(x_val, y_val))

y_true = test_df['Disease category'] - 1
y_pred = []
for id in test_df['ID'].tolist():
    mfccs_feature = audio_to_mfccs(test_df[test_df['ID'] == id]['wav_path'].values[0])
    df = pd.DataFrame(mfccs_feature)
    frame_pred = model.predict(df)
    frame_pred_results = frame_pred.argmax(axis=1)

    person_pred = np.array([np.sum(frame_pred_results == 0), np.sum(frame_pred_results == 1),
                            np.sum(frame_pred_results == 2), np.sum(frame_pred_results == 3),
                            np.sum(frame_pred_results == 4)]).argmax()  # 注意!如三類別相同票數，預測會為0
    y_pred.append(person_pred)

y_pred = np.array(y_pred)
y_pred = y_pred.reshape(-1, 1)
print(y_pred)
