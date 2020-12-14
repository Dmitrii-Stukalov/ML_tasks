from random import randint, uniform

import matplotlib.pyplot as plt

import numpy as np

file = open(r'datasets/LR/3.txt')

M = int(file.readline())


def read_set(file):
    N = int(file.readline())
    x = []
    y = []
    for i in range(N):
        x.append([int(j) for j in file.readline().split()])
        y.append(x[i][-1])
        x[i].pop()
    [row.insert(0, 1) for row in x]
    return normalize(x, y)


def normalize_x(dataset):
    minmax = []
    for i in range(1, len(dataset[0])):
        column = [dataset[j][i] for j in range(len(dataset))]
        minmax.append([min(column), max(column) + 0.01])
    for row in dataset:
        for i in range(1, len(row)):
            row[i] = (row[i] - minmax[i - 1][0]) / (minmax[i - 1][1] - minmax[i - 1][0])

    return dataset


def normalize_y(dataset):
    mini = min(dataset)
    maxi = max(dataset)

    for i in range(len(dataset)):
        dataset[i] = (dataset[i] - mini) / (maxi - mini)

    return dataset


def normalize(x, y):
    return normalize_x(x), normalize_y(y)


def smape(prediction, y):
    er = 0
    n = len(y)
    for i in range(n):
        er += abs(prediction[i] - y[i]) / (abs(prediction[i]) + abs(y[i]))
    return er / n


def smape_stochastic_gradient_descent(theta, X, y, iterations, lamda):
    n = len(y)
    predictions = []
    for j in range(n):
        predictions.append(sum([X[j][i] * theta[i] for i in range(len(theta))]))
    cost_history = [smape(predictions, y)]
    test_cost_history = []
    for it in range(iterations):
        index = randint(0, n - 1)

        prediction = sum([X[index][i] * theta[i] for i in range(len(theta))])
        predictions[index] = prediction
        eps = smape(predictions, y)
        for j in range(len(theta)):
            s = 0
            if prediction > y[index] > 0 or prediction < y[index] < 0:
                s += 2 * y[index] / ((prediction + y[index]) ** 2) * X[index][j]
            elif y[index] > prediction > 0 or y[index] < prediction < 0:
                s -= 2 * y[index] / ((prediction + y[index]) ** 2) * X[index][j]
            # theta[j] -= s / (it + 1) / n
            # regularization
            theta[j] = theta[j] * (1 - lamda / (it + 1)) - s / (it + 1) / n
        # easy calculate
        cost_history.append(eps / (it + 1) + (it / (it + 1)) * cost_history[-1])
        pred = []
        for k in range(len(y_test)):
            pred.append(sum([x_test[k][j] * theta[j] for j in range(len(theta))]))
        test_cost_history.append(smape(pred, y_test))
    return theta, cost_history[1:], test_cost_history


x_train, y_train = read_set(file)
x_test, y_test = read_set(file)

# 1 MHK
# dataset 3
# best lambda 0.3900000000000002
# best smape 0.0042

A = np.transpose(x_train).dot(x_train)

lamda = 0.3900000000000002
A += lamda * np.ones(A.shape)

U, s, VT = np.linalg.svd(A)

d = np.divide(1.0, s, where=s != 0)
D = np.zeros(A.shape)

D[:A.shape[1], :A.shape[1]] = np.diag(d)

B = VT.T.dot(D.T).dot(U.T)

theta = B.dot(np.transpose(x_train)).dot(y_train)

prediction = []
for i in range(len(y_test)):
    prediction.append(sum([x_test[i][j] * theta[j] for j in range(len(theta))]))

print('MHK SMAPE', smape(prediction, y_test))


# 2 SGD
# dataset 3
# best lambda 0.00119
# best smape 0.0042 (avg, depend of random init)


iters = 2000

theta = [uniform(-1 / (2 * (M + 1)), 1 / (2 * (M + 1))) for i in range(M + 1)]

lamda = 0.0019
theta, train_cost_history, test_cost_history = smape_stochastic_gradient_descent(theta, x_train, y_train, iters, lamda)
prediction = []


print('SGD SMAPE', test_cost_history[-1])

fig, (ax1, ax2) = plt.subplots(2, figsize=(12,8))
ax1.set_ylabel('Smape')
ax1.set_xlabel('Iterations')
ax1.set_title('Train')
ax1.plot(range(iters), train_cost_history, 'b.')
ax2.set_ylabel('Smape')
ax2.set_xlabel('Iterations')
ax2.set_title('Test')
ax2.plot(range(iters), test_cost_history, 'r.')

fig.show()
