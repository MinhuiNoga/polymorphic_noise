import librosa
import numpy as np
import tensorflow as tf
from keras.layers import Dense, LSTM
import pandas as pd
import os


# 加载音频文件并提取 MFCC 特征
def audio_to_mfccs(filename):
    voice, sample_rate = librosa.load(filename)

    mfcc_feature = librosa.feature.mfcc(y=voice, n_mfcc=13)
    delta_mfcc_feature = librosa.feature.delta(mfcc_feature)

    mfccs = np.concatenate((mfcc_feature, delta_mfcc_feature))
    mfccs_features = np.transpose(mfcc_feature)  # all frames

    return mfccs_features


# 定义训练集和标签
train_x = []
train_y = np.array([])

# 加载音频数据集和标签
# ...
df_csv = pd.read_csv("Training Dataset/training datalist.csv")
train_y = df_csv["Disease category"].to_numpy()
df_id = np.array(df_csv["ID"])
df_id = "Training Dataset/training_voice_data/" + df_id + ".wav"

for file_path, label in zip(df_id, train_y):
    features = audio_to_mfccs(file_path)
    train_x.append(features)
    train_y = np.append(train_y, label)

train_x = np.vstack(train_x)
print(train_x.shape)
print(train_y.shape)



'''
# 提取 MFCC 特征并将其用于训练 LSTM 模型
for file_path, label in zip(df_id, train_y):
    features = extract_features(file_path)
    train_x.append(features)
    train_y.append(label)

train_x = np.array(train_x)
train_y = np.array(train_y)

# 构建 LSTM 模型
model = Sequential()
model.add(LSTM(128, input_shape=(train_x.shape[1], train_x.shape[2])))
model.add(Dense(64, activation='relu'))
model.add(Dense(5, activation='softmax'))

# 编译模型并进行训练
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_x, train_y, epochs=50, batch_size=32, validation_split=0.2)


'''
