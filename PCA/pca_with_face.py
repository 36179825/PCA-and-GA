import numpy as np
import matplotlib.pyplot as plt
from DL_1113 import loadim

resize_ratio = 0.25
h = int(120 * 0.25)
w = int(100 * 0.25)


def compute_eigen(x, energy=0.85):
    n_data = len(x)
    tmp = np.copy(x)

    # compute co-varaince matrix
    img_mean = np.mean(x, axis=0)
    tmp -= img_mean
    cov = np.matrix(tmp.T) * np.matrix(tmp)

    # eigens
    eigen_values, eigen_vectors = np.linalg.eig(cov)
    sort_indice = eigen_values.argsort()[::-1]
    eigen_values = eigen_values[sort_indice]
    eigen_vectors = eigen_vectors[sort_indice]

    evalues_sum = np.sum(eigen_values)
    evalues_count = 0  # that they include approx. 85% of the energy
    evalues_energy = 0.0
    for evalue in eigen_values:
        evalues_count += 1
        evalues_energy += evalue / evalues_sum

        if evalues_energy >= energy:
            break

    eigen_values = eigen_values[0:evalues_count]  # reduce the number of eigenvectors/values to consider
    eigen_vectors = eigen_vectors[0:evalues_count]
    evnorm = np.linalg.norm(eigen_vectors)
    eigen_vectors /= evnorm
    return (eigen_values, eigen_vectors)


# Main Training
path = "../datasets/DB/"
(x_train, y_train), (x_test, y_test) = loadim.load_data(path=path, resize_ratio=resize_ratio)

eigen_vals, eigen_vecs = compute_eigen(x_train)
print(eigen_vals.shape)
print(eigen_vecs.shape)

X = np.matrix(x_train) * np.matrix(eigen_vecs.T)
x = np.array(list(X[:, 0].flat))
y = np.array(list(X[:, 1].flat))

plt.scatter(x[:39], y[:39],c='b',marker='x')
plt.scatter(x[39:], y[39:],c='r',marker='o')
plt.show()
