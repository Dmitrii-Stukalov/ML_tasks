from random import sample

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class PCA:
    def __init__(self, no_of_components):
        self.no_of_components = no_of_components

    def fit_transform(self, x):
        mean = np.mean(x, axis=0)

        cov_matrix = np.cov(x - mean, rowvar=False)

        eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)
        eigen_vectors = eigen_vectors.T

        sorted_components = np.argsort(eigen_values)[::-1]

        projection_matrix = eigen_vectors[sorted_components[:self.no_of_components]]
        return np.dot(x - mean, projection_matrix.T)


def normalize(dataset):
    minmax = []
    for i in range(1, len(dataset[0])):
        minmax.append([dataset[:][i].min(), dataset[:][i].max()])
    for row in dataset:
        for i in range(0, len(row) - 1):
            row[i + 1] = (row[i + 1] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
        row[0] -= 1

    return dataset[:, 1:], dataset[:, 0]


def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)):
        distance += (row1[i] - row2[i]) ** 2
    return np.sqrt(distance)


def manhattan_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)):
        distance += abs(row1[i] - row2[i])
    return distance


def chebyshev_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)):
        tmp = abs(row1[i] - row2[i])
        if tmp > distance:
            distance = tmp
    return distance


def recalculate_clusters(X, centroids, k, dist):
    clusters = {}
    for i in range(k):
        clusters[i] = []
    for data in X:
        euc_dist = []
        for j in range(k):
            # euc_dist.append(np.linalg.norm(data - centroids[j]))
            euc_dist.append(dist(data, centroids[j]))
        clusters[euc_dist.index(min(euc_dist))].append(data)
    return clusters


def recalculate_centroids(centroids, clusters, k):
    for i in range(k):
        centroids[i] = np.average(clusters[i], axis=0)
    return centroids


def train(X, epochs, k, dist):
    centroids_idx = sample(range(0, len(X) - 1), k)
    centroids = []
    for i in centroids_idx:
        centroids.append(X[i])
    clusters = {}
    for i in range(epochs):
        clusters = recalculate_clusters(X, centroids, k, dist)
        old_centroids = centroids
        centroids = recalculate_centroids(centroids, clusters, k)
        if old_centroids == centroids:
            break
    return clusters


def compactness(clusters, row, i):
    a = 0
    for point in clusters[i]:
        a += np.linalg.norm(point - row)
    return a / (len(clusters[i]) + 1e-10)


def separability(clusters, row, i, k):
    min = 1e100
    for j in range(k):
        if i == j:
            continue
        b = compactness(clusters, row, j)
        if b < min:
            min = b
    return min


def calculate_f(X, Y, clusters, k):
    matrix = np.zeros((len(set(Y)), k))
    m = len(set(Y))
    for i in range(len(Y)):
        for j in range(k):
            stop = False
            for row in clusters[j]:
                if (X[i] == row).all():
                    matrix[int(Y[i])][j] += 1
                    stop = True
                    break
            if stop:
                break

    n = len(X)
    for i in range(m):
        for j in range(k):
            matrix[i][j] += 1e-10
            matrix[i][j] /= n

    F = 0
    for j in range(k):
        max_row = -1
        for i in range(m):
            current_max = 2 * (matrix[i][j] / sum(matrix[i])) * (matrix[i][j] / sum(matrix[:, j])) / (
                    matrix[i][j] / sum(matrix[i]) + matrix[i][j] / sum(matrix[:, j]))
            if current_max > max_row:
                max_row = current_max
        F += sum(matrix[:, j]) * max_row
    return F


def calculate_silhouette(clusters, k):
    sil = 0
    n = 0
    for i in range(k):
        n += len(clusters[i])
        for row in clusters[i]:
            a = compactness(clusters, row, i)
            b = separability(clusters, row, i, k)
            sil += (b - a) / max(a, b)
    sil /= n
    return sil


dataset = pd.read_csv('datasets/Clust/wine.csv')
X, Y = normalize(dataset.values)
max_epochs = 5000
k = 3
best_dist = None
best_F = -1
dists = [euclidean_distance, manhattan_distance, chebyshev_distance]
for dist in dists:
    print(dist.__name__)
    clusters = train(X, max_epochs, k, dist)

    F = calculate_f(X, Y, clusters, k)
    print('F-measure:', F)

    sil = calculate_silhouette(clusters, k)
    print('Silhouette:', sil)
    if F > best_F:
        best_F = F
        best_dist = dist
print()
print('Best dist', best_dist.__name__)
clusters = train(X, max_epochs, k, best_dist)
F = calculate_f(X, Y, clusters, k)
print('Best F-measure:', F)

sil = calculate_silhouette(clusters, k)
print('Best Silhouette:', sil)

x_clusters = []
y_clusters = []
for i in range(k):
    for row in clusters[i]:
        x_clusters.append(row)
        y_clusters.append(i)

pca = PCA(2)
components = pca.fit_transform(X)
components_clusters = pca.fit_transform(np.array(x_clusters))

colors = ['red', 'green', 'blue']
for i in range(len(Y)):
    plt.scatter(components[i][0], components[i][1], color=colors[int(Y[i])])
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()

for i in range(len(y_clusters)):
    plt.scatter(components_clusters[i][0], components_clusters[i][1], color=colors[y_clusters[i]])
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()

max_epochs = 1000
xs = []
fs = []
sils = []
for k in range(3, 30):
    clusters = train(X, max_epochs, k, best_dist)
    xs.append(k)
    fs.append(calculate_f(X, Y, clusters, k))
    sils.append(calculate_silhouette(clusters, k))

plt.plot(xs, fs, label='F1-measure')
plt.plot(xs, sils, label='Silhouette')
plt.xlabel('k')
plt.legend()
plt.grid()
plt.show()
