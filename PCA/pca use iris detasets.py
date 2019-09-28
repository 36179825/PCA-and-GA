"""

===============
Incremental PCA
===============

Incremental principal component analysis (IPCA) is typically used as a
replacement for principal component analysis (PCA) when the dataset to be
decomposed is too large to fit in memory. IPCA builds a low-rank approximation
for the input data using an amount of memory which is independent of the
number of input data samples. It is still dependent on the input data features,
but changing the batch size allows for control of memory usage.

This example serves as a visual check that IPCA is able to find a similar
projection of the data to PCA (to a sign flip), while only processing a
few samples at a time. This can be considered a "toy example", as IPCA is
intended for large datasets which do not fit in main memory, requiring
incremental approaches.

"""

# Authors: Kyle Kastner
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

iris = load_iris()
X = iris.data
y = iris.target

print (X.shape) #150個樣本,4個特徵
print (y.shape) 

n_components = 2

pca = PCA(n_components=n_components) #取前兩個
X_pca = pca.fit_transform(X)
print (X_pca.shape)

colors = ['navy', 'turquoise', 'darkorange']

for X_transformed,title in [(X_pca, "PCA")]:
    plt.figure(figsize=(8, 8))
    for color, i, target_name in zip(colors, [0, 1, 2], iris.target_names):
        plt.scatter(X_transformed[y == i, 0], X_transformed[y == i, 1],
                    color=color, lw=2, label=target_name)
        plt.title(title + " of iris dataset")
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.axis([-4, 4, -1.5, 1.5])

plt.show()

plt.figure(figsize=(8,8))
plt.scatter(X_pca[:,0],X_pca[:,1],c=y)
plt.xlabel('First Principle Component')
plt.ylabel('Second Principle Component')
plt.show()