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

# بررسی اطلاعات Dataset
# یکی از مهم‌ترین دستورات Pandas:
df.info()

# بررسی Missing Values
# یکی از اولین بررسی‌های مهم:
df.isnull().sum()


# درصد Missing Values
missing_percentage = (
    df.isnull().sum() / len(df) * 100
)

print(missing_percentage)



# بررسی Duplicate
df.duplicated().sum()



# بررسی Unique بودن Student + Semester
# چون تعریف کردیم:
# هر ردیف = یک دانشجو در یک ترم
# پس ترکیب زیر باید Unique باشد:
# student_id + semester
duplicate_student_semester = df.duplicated(
    subset=["student_id", "semester"]
).sum()

print(duplicate_student_semester)



# تعداد دانشجویان
print(df["student_id"].nunique())


# تعداد رکورد هر دانشجو
records_per_student = df.groupby("student_id").size()

print(records_per_student.value_counts())


# بررسی Target
print(df["at_risk"].value_counts())


# درصد Target

print(
    df["at_risk"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# بررسی آماری Dataset
print(df.describe())

