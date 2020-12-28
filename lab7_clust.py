from random import randint, sample

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def normalize(dataset):
    minmax = []
    # for row in dataset:
    #     if row[-1] == 'opel':
    #         row[-1] = 0
    #     elif row[-1] == 'saab':
    #         row[-1] = 1
    #     elif row[-1] == 'bus':
    #         row[-1] = 2
    #     elif row[-1] == 'van':
    #         row[-1] = 3
    for i in range(1, len(dataset[0])):
        minmax.append([dataset[:][i].min(), dataset[:][i].max()])
    for row in dataset:
        for i in range(0, len(row) - 1):
            row[i + 1] = (row[i + 1] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
        row[0] -= 1

    return dataset[:, 1:], dataset[:, 0]


def recalculate_clusters(X, centroids, k):
    clusters = {}
    for i in range(k):
        clusters[i] = []
    for data in X:
        euc_dist = []
        for j in range(k):
            euc_dist.append(np.linalg.norm(data - centroids[j]))
        clusters[euc_dist.index(min(euc_dist))].append(data)
    return clusters


def recalculate_centroids(centroids, clusters, k):
    for i in range(k):
        centroids[i] = np.average(clusters[i], axis=0)
    return centroids


def train(X, epochs, k):
    centroids_idx = sample(range(0, len(X) - 1), k)
    centroids = []
    for i in centroids_idx:
        centroids.append(X[i])
    clusters = {}

    for i in range(epochs):
        clusters = recalculate_clusters(X, centroids, k)
        centroids = recalculate_centroids(centroids, clusters, k)

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
epochs = 5000
k = 3
clusters = train(X, epochs, k)

F = calculate_f(X, Y, clusters, k)
print('F-measure:', F)

sil = calculate_silhouette(clusters, k)
print('Silhouette:', sil)

x_clusters = []
y_clusters = []
for i in range(k):
    for row in clusters[i]:
        x_clusters.append(row)
        y_clusters.append(i)

pca = PCA(n_components=2)
components = pca.fit_transform(X)
components_clusters = pca.fit_transform(x_clusters)

colors = ['red', 'green', 'blue', 'yellow']
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

epochs = 10
xs = []
fs = []
sils = []
for k in range(1, 30):
    clusters = train(X, epochs, k)
    xs.append(k)
    fs.append(calculate_f(X, Y, clusters, k))
    sils.append(calculate_silhouette(clusters, k))

plt.plot(xs, fs, label='F1-measure')
plt.plot(xs, sils, label='Silhouette')
plt.xlabel('k')
plt.legend()
plt.grid()
plt.show()
