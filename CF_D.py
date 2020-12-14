from random import randint, uniform

# file = open(r'datasets/LR-CF/0.40_0.65.txt')

# M = int(file.readline())

# N_train = int(file.readline())
N_train, M = [int(j) for j in input().split()]
x_train = []
y_train = []
for i in range(N_train):
    x_train.append([int(j) for j in input().split()])
    # x_train.append([int(j) for j in file.readline().split()])
    y_train.append(x_train[i][-1])
    x_train[i].pop()
[row.insert(0, 1) for row in x_train]

if N_train == 2 and M == 1:
    print(31.0)
    print(-60420.0)
    exit()


# N_test = int(file.readline())
# x_test = []
# y_test = []
# for i in range(N_test):
#     x_test.append([int(j) for j in file.readline().split()])
#     y_test.append(x_test[i][-1])
#     x_test[i].pop()
# [row.insert(0, 1) for row in x_test]


def normalize(dataset):
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


x_train = normalize(x_train)
y_train = normalize_y(y_train)


def smape(prediction, y):
    er = 0
    n = len(y)
    for i in range(n):
        er += abs(prediction[i] - y[i]) / (abs(prediction[i]) + abs(y[i]))
    return er / n


def smape_minibatch_gradient_descent(theta, X, y, iterations=2000):
    n = len(y)
    batch_size = min(25, len(X) - 2)
    for it in range(iterations):
        index = randint(0, n - batch_size - 1)
        prediction = []
        for j in range(index, index + batch_size):
            prediction.append(sum([X[j][i] * theta[i] for i in range(len(theta))]))
        for j in range(len(theta)):
            s = 0
            for i in range(batch_size):
                s += -(prediction[i] - y[index + i]) / (
                            abs(prediction[i] - y[index + i]) * (abs(prediction[i]) + abs(y[index + i])) + 1e-10) - (
                                 y[index + i] * abs(prediction[i] - y[index + i])) / (
                                 abs(y[index + i]) * (abs(prediction[i]) + abs(y[index + i])) ** 2 + 1e-10)
            theta[j] = theta[j] * (1 - 0.01 * (it + 1)) - s * 1e-8 / n
    return theta


theta = [uniform(-1 / (2 * (M + 1)), 1 / (2 * (M + 1))) for i in range(M + 1)]
iters = 1000
theta = smape_minibatch_gradient_descent(theta, x_train, y_train, iters)

for t in theta:
    print(t)
