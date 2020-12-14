from math import sqrt, exp, pi, cos


def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)):
        distance += (row1[i] - row2[i]) ** 2
    return sqrt(distance)


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


def get_distances(distance_type, dataset, test_row):
    if distance_type == 'euclidean':
        return [euclidean_distance(train_row[:-1], test_row) for train_row in dataset]
    elif distance_type == 'manhattan':
        return [manhattan_distance(train_row[:-1], test_row) for train_row in dataset]
    elif distance_type == 'chebyshev':
        return [chebyshev_distance(train_row[:-1], test_row) for train_row in dataset]


def uniform_kernel(u):
    if abs(u) < 1:
        return 0.5
    return 0


def triangular_kernel(u):
    if abs(u) < 1:
        return 1 - abs(u)
    return 0


def epanechnikov_kernel(u):
    if abs(u) < 1:
        return 3 / 4 * (1 - u ** 2)
    return 0


def quartic_kernel(u):
    if abs(u) < 1:
        return 15 / 16 * (1 - u ** 2) ** 2
    return 0


def triweight_kernel(u):
    if abs(u) < 1:
        return 35 / 32 * (1 - u ** 2) ** 3
    return 0


def tricube_kernel(u):
    if abs(u) < 1:
        return 70 / 81 * (1 - abs(u) ** 3) ** 3
    return 0


def gaussian_kernel(u):
    return exp(-u ** 2 / 2) / sqrt(2 * pi)


def cosine_kernel(u):
    if abs(u) < 1:
        return pi / 4 * cos(pi * u / 2)
    return 0


def logistic_kernel(u):
    return 1 / (exp(u) + 2 + exp(-u))


def sigmoid_kernel(u):
    return 2 / (pi * (exp(u) + exp(-u)))


def get_kernels(kernel_type, window_size, distances):
    if kernel_type == 'uniform':
        return [uniform_kernel(distance / window_size) for distance in distances]
    elif kernel_type == 'triangular':
        return [triangular_kernel(distance / window_size) for distance in distances]
    elif kernel_type == 'epanechnikov':
        return [epanechnikov_kernel(distance / window_size) for distance in distances]
    elif kernel_type == 'quartic':
        return [quartic_kernel(distance / window_size) for distance in distances]
    elif kernel_type == 'triweight':
        return [triweight_kernel(distance / window_size) for distance in distances]
    elif kernel_type == 'tricube':
        return [tricube_kernel(distance / window_size) for distance in distances]
    elif kernel_type == 'gaussian':
        return [gaussian_kernel(distance / window_size) for distance in distances]
    elif kernel_type == 'cosine':
        return [cosine_kernel(distance / window_size) for distance in distances]
    elif kernel_type == 'logistic':
        return [logistic_kernel(distance / window_size) for distance in distances]
    elif kernel_type == 'sigmoid':
        return [sigmoid_kernel(distance / window_size) for distance in distances]


n, _ = [int(i) for i in input().split()]
dataset = []
for i in range(n):
    dataset.append([int(i) for i in input().split()])

test = [int(i) for i in input().split()]
distance_type = input()

d = get_distances(distance_type, dataset, test)
d_s = sorted(d)

kernel_type = input()
window_type = input()

window_size = int(input()) if window_type == 'fixed' else d_s[int(input())]

if window_size == 0:
    window_size = 1

kernels = get_kernels(kernel_type, window_size, d)
if sum(kernels) != 0:
    ans = sum([dataset[i][-1] * kernels[i] for i in range(n)]) / sum(kernels) if sum(kernels) != 0 else 0
    print(ans)
else:
    ans = 0.0
    cnt = 0
    for i in range(n):
        if dataset[i][:-1] == test:
            ans += dataset[i][-1]
            cnt += 1
    if cnt != 0:
        print(ans / cnt)
    else:
        print(sum([row[-1] for row in dataset]) / n)
