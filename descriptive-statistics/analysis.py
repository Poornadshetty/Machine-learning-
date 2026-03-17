import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['Species'] = iris.target


# -----------------------------
# Numerical column analysis
# -----------------------------
numerical_column = "sepal length (cm)"
categorical_column = "Species"

# Descriptive statistics
mean = df[numerical_column].mean()
median = df[numerical_column].median()
mode = df[numerical_column].mode()[0]
std_dev = df[numerical_column].std()
variance = df[numerical_column].var()
data_range = df[numerical_column].max() - df[numerical_column].min()

print("Descriptive Statistics:")
print(f"Mean: {mean}")
print(f"Median: {median}")
print(f"Mode: {mode}")
print(f"Standard Deviation: {std_dev}")
print(f"Variance: {variance}")
print(f"Range: {data_range}")

# -----------------------------
# Histogram
# -----------------------------
plt.figure(figsize=(10, 6))
plt.hist(df[numerical_column], bins=30, edgecolor='black')
plt.title(f"Histogram of {numerical_column}")
plt.xlabel(numerical_column)
plt.ylabel("Frequency")
plt.show()

# -----------------------------
# Boxplot
# -----------------------------
plt.figure(figsize=(8, 6))
plt.boxplot(df[numerical_column], vert=False)
plt.title(f"Boxplot of {numerical_column}")
plt.xlabel(numerical_column)
plt.show()

# -----------------------------
# Outlier detection using IQR
# -----------------------------
Q1 = df[numerical_column].quantile(0.25)
Q3 = df[numerical_column].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df[numerical_column] < lower_bound) |
              (df[numerical_column] > upper_bound)]

print("Outliers:")
print(outliers)

# -----------------------------
# Categorical column analysis
# -----------------------------
categorical_column = "Species"
category_counts = df[categorical_column].value_counts()

print(f"Frequency of categories in {categorical_column}:")
print(category_counts)

# -----------------------------
# Bar chart
# -----------------------------
plt.figure(figsize=(10, 6))
plt.bar(category_counts.index, category_counts.values)
plt.title(f"Bar Chart of {categorical_column}")
plt.xlabel(categorical_column)
plt.ylabel("Frequency")
plt.show()

# -----------------------------
# Pie chart
# -----------------------------
plt.figure(figsize=(8, 8))
plt.pie(
    category_counts.values,
    labels=category_counts.index,
    autopct="%1.1f%%",
    startangle=90
)
plt.title(f"Pie Chart of {categorical_column}")
plt.show()
