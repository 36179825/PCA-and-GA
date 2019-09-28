from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import warnings,math
import matplotlib.pyplot as plt
 
##讀取wine 資料+
df_wine = pd.read_csv('wine_data.csv',
       header=None)      
        
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
       'Alcalinity of ash', 'Magnesium', 'Total phenols',
       'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
       'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
       'Proline']
from sklearn.model_selection import train_test_split
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values #0:d為 class  1:之後為13個特徵行
X_train, X_test, y_train, y_test = \
 train_test_split(X, y, test_size=0.3, random_state=0)
 
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.fit_transform(X_test)
##讀取wine 資料-
 
#求共變異係數矩陣
cov_mat = np.cov(X_train_std.T)
print("共變異係數矩陣.shape=",cov_mat.shape)
print("共變異係數矩陣=",cov_mat)
 
#求共變異係數矩陣 的特徵向量及特徵值
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print("特徵向量.shape=",eigen_vecs.shape)
print("特徵向量=",eigen_vecs)
print("特徵值=",eigen_vals)
 
 
#計算解釋變異數比率 各特徵值/特徵值總和
tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
print("各特徵值變異數比率：",var_exp)
print("特徵值變異數比率累加：",cum_var_exp)
 
#畫圖 ：解釋變異數比率 ，各特徵值/特徵值總和
plt.bar(range(1, 14), var_exp, alpha=0.5, align='center',
        label='individual explained variance')
plt.step(range(1, 14), cum_var_exp, where='mid',
         label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
 
# Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
               for i in range(len(eigen_vals))]
print("特徵值，特徵向量length：",len(eigen_pairs))
print("特徵值，特徵向量：",eigen_pairs)
# Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs.sort(key=lambda k: k[0], reverse=True)
print("特徵值，特徵向量排序：",eigen_pairs.sort(key=lambda k: k[0], reverse=True))
#保留兩個最具影響力的特徵向量組成13x2 的投影矩陣W
w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]))
print('Matrix W:\n', w)
 
#畫出轉換後的數據集 散點圖
#print("X_train_std[0].dot(w)=",X_train_std[0].dot(w))
X_train_pca = X_train_std.dot(w)
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train == l, 0], 
                X_train_pca[y_train == l, 1], 
                c=c, label=l, marker=m)
 
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
 
##劃出決策分布圖
from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)
      
#使用LogisticRegression 並用兩個PCA 主成分做訓練分類 
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr = lr.fit(X_train_pca, y_train)
plot_decision_regions(X_train_pca, y_train, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()