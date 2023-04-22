import numpy as np
import pandas as pd
import librosa


# 資料判斷
df_csv = pd.read_csv("Training Dataset/training datalist.csv")
print("資料資訊")
df_csv.info()


# 挑選出要訓練的Disease category 1、2、3
source_df = source_df.loc[source_df['Disease category'].isin([1, 2, 3]), [
    'ID', 'Disease category']]

# 在dataframe中加入要訓練的音檔路徑
source_df['wav_path'] = source_df['ID'].map(
    "./Training Dataset/training_voice_data{}.wav".format)

print("Disease category in source_df :",
      source_df['Disease category'].unique())
print("source_df :\n", source_df)


def replace_sex(x):
    return x - 1

#sex轉成0 or 1


df_csv["Sex"] = df_csv["Sex"].apply(replace_sex)


# 資料處理

y = df_csv["Disease category"].to_numpy().reshape(-1, 1)
print("資料label總數(矩陣): ", y.shape)

x = pd.concat([df_csv.iloc[0:2], df_csv[3:26]])


# 定義函數
def audio_to_mfccs(filename, sample_rate=44100, offset=0, duration=None):
    # 讀取音訊檔案，並設定取樣率、起始時間、及持續時間
    voice, sample_rate = librosa.load(
        filename, sr=sample_rate, offset=offset, duration=duration
    )

    # 將時間值轉換為 FFT 與 hop length 所需的框架數 (以取樣點表示)
    n_fft = int(16/1000 * sample_rate)  # 將 16 毫秒轉換為取樣點
    hop_length = int(8/1000 * sample_rate)  # 將 8 毫秒轉換為取樣點

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
