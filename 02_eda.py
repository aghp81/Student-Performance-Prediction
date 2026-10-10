"""EDA script for Student Performance Prediction. Run in the same folder as students.csv."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = Path("students.csv")
OUTPUT_DIR = Path("output_eda")
OUTPUT_DIR.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")
if not DATA_PATH.exists():
    raise FileNotFoundError(f"{DATA_PATH.resolve()} not found. Put this script beside students.csv.")
df = pd.read_csv(DATA_PATH)
print("DATASET:", df.shape, "| unique students:", df.student_id.nunique())
print("\nTARGET COUNTS:\n", df.at_risk.value_counts().sort_index())
print("\nTARGET PERCENTAGES:\n", (df.at_risk.value_counts(normalize=True).sort_index()*100).round(2))
df["risk_group"] = df.at_risk.map({0:"Normal", 1:"At Risk"})

# 1) Histograms
hist_cols = ["previous_gpa", "cumulative_gpa", "attendance_rate", "midterm_average", "quiz_average", "assignment_average", "lms_login_count", "learning_time_hours", "final_gpa"]
hist_cols = [c for c in hist_cols if c in df]
df[hist_cols].hist(figsize=(15, 11), bins=25, edgecolor="black")
plt.suptitle("Feature Distributions (Histograms)")
plt.tight_layout(); plt.savefig(OUTPUT_DIR/"01_histograms.png", dpi=150); plt.close()

# 2) Boxplots by risk group
box_cols = ["previous_gpa", "attendance_rate", "midterm_average", "failed_courses", "lms_active_days", "learning_time_hours", "gpa_trend"]
for c in [c for c in box_cols if c in df]:
    plt.figure(figsize=(8,5)); sns.boxplot(data=df, x="risk_group", y=c, order=["Normal", "At Risk"])
    plt.title(f"{c}: Normal vs At Risk"); plt.tight_layout(); plt.savefig(OUTPUT_DIR/f"02_boxplot_{c}.png", dpi=150); plt.close()

# 3) Countplots for categorical features
for c in [c for c in ["gender", "major", "academic_level", "semester"] if c in df]:
    plt.figure(figsize=(10,5)); sns.countplot(data=df, x=c, hue="risk_group")
    plt.title(f"Record counts by {c} and risk group"); plt.xticks(rotation=25, ha="right")
    plt.tight_layout(); plt.savefig(OUTPUT_DIR/f"03_countplot_{c}.png", dpi=150); plt.close()

# 4) Correlation matrix. Exclude ID and final_gpa (directly defines target = leakage).
corr_cols = ["age", "semester", "previous_gpa", "cumulative_gpa", "failed_courses", "completed_credits", "withdrawn_courses", "attendance_rate", "absence_count", "late_count", "midterm_average", "quiz_average", "assignment_average", "project_average", "completed_assignments", "late_assignments", "lms_login_count", "lms_active_days", "learning_time_hours", "gpa_trend", "performance_trend", "at_risk"]
corr_cols = [c for c in corr_cols if c in df]
corr = df[corr_cols].corr(numeric_only=True)
plt.figure(figsize=(16,12)); sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
plt.title("Correlation Matrix (ID and final_gpa excluded)"); plt.tight_layout()
plt.savefig(OUTPUT_DIR/"04_correlation_matrix.png", dpi=150); plt.close()
print("\nCORRELATION WITH at_risk (Pearson):\n", corr.at_risk.drop("at_risk").sort_values(key=lambda x:x.abs(), ascending=False).round(3).to_string())

# 5) Group statistics
features = ["previous_gpa", "cumulative_gpa", "attendance_rate", "absence_count", "midterm_average", "quiz_average", "assignment_average", "failed_courses", "withdrawn_courses", "completed_assignments", "late_assignments", "lms_login_count", "lms_active_days", "learning_time_hours", "gpa_trend", "performance_trend", "final_gpa"]
features = [c for c in features if c in df]
summary = df.groupby("risk_group")[features].agg(["mean", "median"])
print("\nGROUP COMPARISON (mean and median):\n", summary.round(2).to_string())
summary.to_csv(OUTPUT_DIR/"group_comparison.csv")
selected = [c for c in ["previous_gpa", "attendance_rate", "midterm_average", "failed_courses", "lms_active_days", "learning_time_hours", "gpa_trend"] if c in df]
df.groupby("risk_group")[selected].mean().T.plot(kind="bar", figsize=(12,6))
plt.title("Mean feature values by risk group (different scales)"); plt.ylabel("Mean value"); plt.xticks(rotation=35, ha="right")
plt.tight_layout(); plt.savefig(OUTPUT_DIR/"05_group_means.png", dpi=150); plt.close()

# 6) Risk percentage by semester
risk_by_sem = df.groupby("semester").at_risk.mean().mul(100)
print("\nAT-RISK RATE BY SEMESTER (%):\n", risk_by_sem.round(2).to_string())
plt.figure(figsize=(8,5)); sns.lineplot(x=risk_by_sem.index, y=risk_by_sem.values, marker="o")
plt.title("At-risk rate by semester"); plt.xlabel("Semester"); plt.ylabel("At-risk records (%)")
plt.tight_layout(); plt.savefig(OUTPUT_DIR/"06_risk_rate_by_semester.png", dpi=150); plt.close()
print(f"\nDone. Charts and summaries saved in: {OUTPUT_DIR.resolve()}")
print("Reminder: never include final_gpa in X when predicting at_risk, because it defines the target.")
