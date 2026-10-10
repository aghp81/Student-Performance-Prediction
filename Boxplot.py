import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd





df = pd.read_csv("students.csv")

# تحلیل اول: توزیع متغیر هدف
print(df["at_risk"].value_counts())

print(
    df["at_risk"]
    .value_counts(normalize=True)
    .mul(100)
)


# تحلیل دوم: Histogram
df["previous_gpa"].hist(bins=25)

plt.title("Previous GPA Distribution")
plt.xlabel("Previous GPA")
plt.ylabel("Frequency")

plt.show()


# تحلیل سوم: Boxplot
sns.boxplot(
    data=df,
    x="at_risk",
    y="previous_gpa"
)

plt.xlabel("At Risk (0 = Normal, 1 = At Risk)")
plt.ylabel("Previous GPA")
plt.show()