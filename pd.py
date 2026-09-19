import pandas as pd
import numpy as np

df = pd.read_csv("students.csv")

# print(df)

# print(df.head())

# خواندن 10 ردیف 
print(df.head(10))

# ابعاد Dataset
print(df.shape)

# نام ستون‌ها
print(df.columns)

# برای خوانایی بیشتر:
for column in df.columns:
    print(column)