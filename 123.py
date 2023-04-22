import numpy as np
import pandas as pd





#資料判斷
df_csv = pd.read_csv("Training Dataset/training datalist.csv")
print("資料資訊")
df_csv.info()

#========sex轉成0 or 1============

def replace_sex(x):
    return x - 1

df_csv["Sex"] = df_csv["Sex"].apply(replace_sex)

#========df ppd replace (nan ==> 0)============

df_csv["PPD"].fillna(0, inplace=True)




#資料處理

y = df_csv["Disease category"].to_numpy().reshape(-1,1)
print("資料label總數(矩陣): ",y.shape)

x = pd.concat([df_csv.iloc[0:2],df_csv[3:26]])

print(df_csv["PPD"])

