import tensorflow as tf
import pandas as pd

df_csv = pd.read_csv("Training Dataset/training datalist.csv")


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

y = df_csv["Disease category"].to_numpy().reshape(-1, 1)

df_csv_1 = df_csv.drop(["Disease category", "ID"], axis=1)

x = df_csv_1.to_numpy()




voice_model = tf.keras.models.load_model("AI_CUP_acoustic_sample_model.h5")
med_model = tf.keras.models.load_model("AI_CUP_medical_sample_model.h5")

voice_model_x = voice_model.predict(a)
med_model_x = voice_model.predict(b)

train_df = pd.concat([x])
