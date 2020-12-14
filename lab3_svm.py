from random import randint

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold


def read_dataset(path):
    file = open(path)
    file.readline()
    X = []
    Y = []
    while line := file.readline():
        chars = line.split(',')
        X.append([float(chars[0]), float(chars[1])])
        Y.append(-1) if chars[-1] == 'N\n' else Y.append(1)
    X = np.array(X)
    Y = np.array(Y)
    # normalize_features(X)
    return X, Y


def normalize_features(X):
    minmax = []
    for i in range(len(X[0])):
        minmax.append([X[:][i].min(), X[:][i].max()])
    for row in X:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


def f(row_index, X, Y, alpha, b, kernel, kernel_param):
    ans = b
    if kernel == 'linear':
        for i in range(len(X)):
            if i != row_index:
                ans += alpha[i] * Y[i] * np.dot(X[row_index], X[i])
    elif kernel == 'poly':
        for i in range(len(X)):
            if i != row_index:
                ans += alpha[i] * Y[i] * (np.dot(X[row_index], X[i])) ** kernel_param
    else:
        for i in range(len(X)):
            if i != row_index:
                ans += alpha[i] * Y[i] * np.exp(-kernel_param * np.linalg.norm(X[row_index] - X[i]) ** 2)
    return ans


def predict(point, X, Y, alpha, b, kernel, kernel_param):
    ans = b
    if kernel == 'linear':
        for i in range(len(X)):
            ans += alpha[i] * Y[i] * np.dot(point, X[i])
    elif kernel == 'poly':
        for i in range(len(X)):
            ans += alpha[i] * Y[i] * (np.dot(point, X[i])) ** kernel_param
    else:
        for i in range(len(X)):
            ans += alpha[i] * Y[i] * np.exp(-kernel_param * np.linalg.norm(point - X[i]) ** 2)
    return ans


def SMO(X, Y, C, kernel, kernel_param=1, max_passes=4):
    alpha = np.zeros(X.shape[0])
    b = 0
    passes = 0
    num_changed_alphas = 0
    while passes < max_passes:
        for i in range(len(X)):
            num_changed_alphas = 0
            E_i = f(i, X, Y, alpha, b, kernel, kernel_param) - Y[i]
            if alpha[i] < C or alpha[i] > 0:
                j = randint(0, len(X) - 1)
                while j == i:
                    j = randint(0, len(X) - 1)
                E_j = f(j, X, Y, alpha, b, kernel, kernel_param) - Y[j]
                alpha_i_old = alpha[i]
                alpha_j_old = alpha[j]
                if Y[i] != Y[j]:
                    L = max(0, alpha[j] - alpha[i])
                    H = min(C, C + alpha[j] - alpha[i])
                else:
                    L = max(0, alpha[j] + alpha[i] - C)
                    H = min(C, alpha[j] + alpha[i])
                if L == H:
                    continue
                eta = 2 * np.dot(X[i], X[j]) - np.dot(X[i], X[i]) - np.dot(X[j], X[j])
                if eta >= 0:
                    continue
                alpha[j] -= Y[j] * (E_i - E_j) / eta
                if alpha[j] > H:
                    alpha[j] = H
                elif alpha[j] < L:
                    alpha[j] = L
                if abs(alpha[j] - alpha_j_old) < 1e-5:
                    continue
                alpha[i] += Y[i] * Y[j] * (alpha_j_old - alpha[j])
                b_1 = b - E_i - Y[i] * (alpha[i] - alpha_i_old) * np.dot(X[i], X[i]) \
                      - Y[j] * (alpha[j] - alpha_j_old) * np.dot(X[i], X[j])
                b_2 = b - E_j - Y[i] * (alpha[i] - alpha_i_old) * np.dot(X[i], X[j]) \
                      - Y[j] * (alpha[j] - alpha_j_old) * np.dot(X[j], X[j])
                if 0 < alpha[i] < C:
                    b = b_1
                elif 0 < alpha[j] < C:
                    b = b_2
                else:
                    b = (b_1 + b_2) / 2
                num_changed_alphas += 1
        if num_changed_alphas == 0:
            passes += 1
        else:
            passes = 0
    return alpha, b


def draw_plot(X, Y, C, kernel, kernel_param):
    alpha, b = SMO(X, Y, C, kernel, kernel_param)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x_step = (x_max - x_min) / 100
    y_step = (y_max - y_min) / 100
    for x in np.arange(x_min, x_max, x_step):
        for y in np.arange(y_min, y_max, y_step):
            prediction = predict([x, y], X, Y, alpha, b, kernel, kernel_param)
            if prediction == 0:
                plt.scatter(x, y, c='black')
            elif -1 <= prediction <= 1:
                plt.scatter(x, y, c='yellow')
            elif prediction >= 0:
                plt.scatter(x, y, c='pink')
            else:
                plt.scatter(x, y, c='lime')
    d = {-1: 'green', 1: 'red'}
    plt.scatter(X[:, 0], X[:, 1], c=[d[y] for y in Y])
    plt.show()


datasets = ['datasets/SVM_datasets/chips.csv', 'datasets/SVM_datasets/geyser.csv']
for dataset in datasets:
    print(dataset.split('/')[-1])
    X, Y = read_dataset(dataset)
    kernels = ['linear', 'poly', 'rbf']
    C_all = [0.05, 0.1, 0.5, 1, 5, 10, 50, 100]
    folds = 5
    for kernel in kernels:
        best_acc = -1
        best_C = -1
        best_param = -1
        info = ''
        for C in C_all:
            for kernel_param in range(1, 6):
                acc = 0
                for train_index, test_index in KFold(n_splits=folds, shuffle=True).split(X):
                    X_train, X_test = X[train_index], X[test_index]
                    Y_train, Y_test = Y[train_index], Y[test_index]
                    alpha, b = SMO(X_train, Y_train, C, kernel, kernel_param)
                    ok = 0
                    for i in range(len(X_test)):
                        prediction = 1 if f(i, X_test, Y_test, alpha, b, kernel, kernel_param) >= 0 else -1
                        if prediction == Y_test[i]:
                            ok += 1
                    acc += ok / len(Y_test)
                acc /= folds
                if acc > best_acc:
                    info = 'Kernel ' + kernel + ' C = ' + str(C) + ' parameter = ' + str(
                        kernel_param) + ' with accuracy = ' + str(acc)
                    best_acc = acc
                    best_C = C
                    best_param = kernel_param
        print(info)
        draw_plot(X, Y, best_C, kernel, best_param)

# chips.csv
# Kernel linear C = 10 with accuracy = 0.5503623188405797
# Kernel poly C = 1 parameter = 2 with accuracy = 0.9579710144927537
# Kernel rbf C = 0.5 parameter = 1 with accuracy = 0.975
# geyser.csv
# Kernel linear C = 0.05 with accuracy = 0.6125252525252525
# Kernel poly C = 5 parameter = 3 with accuracy = 0.6036363636363636
# Kernel rbf C = 10 parameter = 5 with accuracy = 0.883030303030303
# Kernel rbf C = 0.5 parameter = 5 with accuracy = 0.8243434343434343
