import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(iris.data,columns=iris.feature_names)
sns.scatterplot(x=df['sepal length (cm)'],y=df['sepal width (cm)'])
plt.title('sepal length vs sepal width')
plt.show()
print(f"Pearson correlation: {df['sepal length (cm)'].corr(df['sepal width (cm)']):.2f}")
print(f"\nCovariance matrix:\n{df.cov()}\n")
print(f"correlation matrix:\n{df.corr()}\n")
sns.heatmap(df.corr(), annot=True,cmap='coolwarm',fmt='.2f')
plt.title('Correlation Matrix Heatmap')
plt.show()
