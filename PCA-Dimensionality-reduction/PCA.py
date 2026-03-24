from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
x = load_iris().data
x_pca = PCA(n_components=2).fit_transform(x)
plt.scatter(x_pca[:,0], x_pca[:,1], c=load_iris().target)
plt.xlabel('PC1 (Most Imprtant Differences)')
plt.ylabel('PC2 (Next Most Important)')
plt.title('Iris Flowers in 2D after PCA')
plt.show()
