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


# 
gpa_columns = [
    "previous_gpa",
    "cumulative_gpa",
    "midterm_average",
    "quiz_average",
    "assignment_average",
    "project_average",
    "final_gpa"
]

print(df[gpa_columns].describe().round(2))



# بررسی مقادیر غیرمجاز GPA
for column in gpa_columns:
    invalid = ((df[column] < 0) | (df[column] > 20)).sum()
    print(column, invalid)

# سنجش سلامت و رفتار متغیر درصد حضور و غیاب
# بررسی Attendance
print(df["attendance_rate"].describe())

invalid = (
    (df["attendance_rate"] < 0) |
    (df["attendance_rate"] > 100)
).sum()

print(invalid)


# نمایش سطرهایی که داده غلط دارند
invalid_rows = df[(df["attendance_rate"] < 0) | (df["attendance_rate"] > 100)]
print(invalid_rows)

# بررسی Age
print(df["age"].describe())


invalid_age = (
    (df["age"] < 18) |
    (df["age"] > 30)
).sum()

print(invalid_age)


# بخش دوم: اعتبارسنجی سن مجاز
invalid_age = (
    (df["age"] < 18) |
    (df["age"] > 30)
).sum()

print(invalid_age)