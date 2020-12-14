import sys
from math import sqrt, pi, exp, cos

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

    # for row in dataset:
    #     if row[-1] == "'1'":
    #         row[-1] = 0
    #     elif row[-1] == "'2'":
    #         row[-1] = 1
    #     elif row[-1] == "'3'":
    #         row[-1] = 2
    #     elif row[-1] == "'4'":
    #         row[-1] = 3
    #     elif row[-1] == "'5'":
    #         row[-1] = 4
    #     elif row[-1] == "'6'":
    #         row[-1] = 5

    for i in range(len(dataset[0]) - 1):
        minmax.append([dataset[:][i].min(), dataset[:][i].max()])
    for row in dataset:
        for i in range(len(row) - 1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
        row[-1] -= 1

    return dataset


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
    try:
        return exp(-u ** 2 / 2) / sqrt(2 * pi)
    except OverflowError:
        return 0


def cosine_kernel(u):
    if abs(u) < 1:
        return pi / 4 * cos(pi * u / 2)
    return 0


def logistic_kernel(u):
    try:
        return 1 / (exp(u) + 2 + exp(-u))
    except OverflowError:
        return 0


def sigmoid_kernel(u):
    try:
        return 2 / (pi * (exp(u) + exp(-u)))
    except OverflowError:
        return 0


dataset = pd.read_csv('datasets/KNN_datasets/wine.csv')
# sns.pairplot(dataset, hue='Class')
# plt.show()
dataset = dataset[[c for c in dataset if c != 'class'] + ['class']]
num_classes = 3
normalized = normalize(dataset.values)
onehot_labels = pd.get_dummies(dataset.values[:, -1])

dist_funcs = [euclidean_distance, manhattan_distance, chebyshev_distance]
kernel_funcs = [uniform_kernel, triangular_kernel, epanechnikov_kernel, quartic_kernel, triweight_kernel,
                tricube_kernel, gaussian_kernel, cosine_kernel, logistic_kernel, sigmoid_kernel]

# result = ''
# max_F1 = 0
# for dist in dist_funcs:
#     for kernel in kernel_funcs:
#         for neighbours in range(len(normalized) // num_classes):
#             CM = np.zeros((num_classes, num_classes))
#             for j in range(len(normalized)):
#                 test = normalized[j][:-1]
#                 distances = [dist(train_row[:-1], test) for train_row in normalized]
#                 distances[j] = sys.float_info.max
#                 distances_sorted = sorted(distances)
#
#                 window_size = neighbours / 20
#                 if window_size == 0:
#                     window_size = 1
#
#                 kernels = [kernel(distance / window_size) for distance in distances]
#
#                 ans = sum([normalized[i][-1] * kernels[i] for i in range(len(normalized) - 1)]) / sum(kernels) if sum(
#                     kernels) != 0 else 0
#                 CM[round(ans)][round(normalized[j][-1])] += 1
#
#             T = [CM[i][i] for i in range(num_classes)]
#             C = [sum(i) for i in CM]
#             P = [sum(i) for i in zip(*CM)]
#             All = sum(C)
#
#             Precision_w = sum([(T[i] * C[i] / P[i]) if P[i] != 0 else 0 for i in range(num_classes)]) / All
#             Recall_w = sum(T) / All
#             F1_macro = 2 * Precision_w * Recall_w / (Precision_w + Recall_w)
#
#             print('Macro:', F1_macro, 'Distance:', dist.__name__, 'Kernel:', kernel.__name__, 'Neighbours:',
#                   neighbours + 1)
#             if F1_macro > max_F1:
#                 max_F1 = F1_macro
#                 result = 'Macro:' + str(
#                     F1_macro) + 'Distance:' + dist.__name__ + 'Kernel:' + kernel.__name__ + 'Neighbours:' + str(
#                     neighbours + 1)
#
# print(result)
# print('!!!!!!!!!IMPORTANT!!!!!!!!!!!!!NAIVE_RESULT_ABOVE')
#
# onehot_result = ''
# onehot_max_F1 = 0
# for dist in dist_funcs:
#     for kernel in kernel_funcs:
#         CM = np.zeros((num_classes, num_classes))
#         for neighbours in range(len(normalized) // num_classes):
#             for j in range(len(normalized)):
#                 test = normalized[j][:-1]
#                 distances = [dist(train_row[:-1], test) for train_row in normalized]
#                 distances[j] = sys.float_info.max
#                 distances_sorted = sorted(distances)
#
#                 window_size = neighbours / 20
#                 if window_size == 0:
#                     window_size = 1
#
#                 kernels = [kernel(distance / window_size) for distance in distances]
#
#                 ans = []
#                 for k in range(num_classes):
#                     ans.append(
#                         sum([onehot_labels.values[i][-(k + 1)] * kernels[i] for i in range(len(normalized))]) / sum(
#                             kernels) if sum(kernels) != 0 else 0)
#                 ans.reverse()
#                 CM[ans.index(max(ans))][round(normalized[j][-1])] += 1
#
#             T = [CM[i][i] for i in range(num_classes)]
#             C = [sum(i) for i in CM]
#             P = [sum(i) for i in zip(*CM)]
#             All = sum(C)
#
#             Precision_w = sum([(T[i] * C[i] / P[i]) if P[i] != 0 else 0 for i in range(num_classes)]) / All
#             Recall_w = sum(T) / All
#             F1_macro = 2 * Precision_w * Recall_w / (Precision_w + Recall_w)
#
#             print('Macro:', F1_macro, 'Distance:', dist.__name__, 'Kernel:', kernel.__name__, 'Neighbours:',
#                   neighbours + 1)
#             if F1_macro > onehot_max_F1:
#                 onehot_max_F1 = F1_macro
#                 onehot_result = 'Macro:' + str(
#                     F1_macro) + 'Distance:' + dist.__name__ + 'Kernel:' + kernel.__name__ + 'Neighbours:' + str(
#                     neighbours + 1)
#
# print(onehot_result)

plot_data = []
for neighbours in range(len(normalized) // num_classes):
    CM = np.zeros((num_classes, num_classes))
    CM_hot = np.zeros((num_classes, num_classes))
    for j in range(len(normalized)):
        test = normalized[j][:-1]
        distances = [manhattan_distance(train_row[:-1], test) for train_row in normalized]
        # distances[j] = sys.float_info.max
        distances.pop(j)
        distances_sorted = sorted(distances)

        window_size = distances_sorted[neighbours + 1]
        if window_size == 0:
            window_size = 1

        kernels = [triweight_kernel(distance / window_size) for distance in distances]

        ans = sum([normalized[i][-1] * kernels[i] for i in range(len(normalized) - 1)]) / sum(kernels) if sum(
            kernels) != 0 else 0
        CM[round(ans)][round(normalized[j][-1])] += 1

        ans_hot = []
        for k in range(num_classes):
            ans_hot.append(
                sum([onehot_labels.values[i][-(k + 1)] * kernels[i] for i in range(len(normalized) - 1)]) / sum(
                    kernels) if sum(kernels) != 0 else 0)
        ans_hot.reverse()
        CM_hot[ans_hot.index(max(ans_hot))][round(normalized[j][-1])] += 1

    T = [CM[i][i] for i in range(num_classes)]
    C = [sum(i) for i in CM]
    P = [sum(i) for i in zip(*CM)]
    All = sum(C)

    T_hot = [CM_hot[i][i] for i in range(num_classes)]
    C_hot = [sum(i) for i in CM_hot]
    P_hot = [sum(i) for i in zip(*CM_hot)]
    All_hot = sum(C_hot)

    Precision_w = sum([(T[i] * C[i] / P[i]) if P[i] != 0 else 0 for i in range(num_classes)]) / All
    Recall_w = sum(T) / All
    F1_macro = 2 * Precision_w * Recall_w / (Precision_w + Recall_w)

    Precision_w_hot = sum(
        [(T_hot[i] * C_hot[i] / P_hot[i]) if P_hot[i] != 0 else 0 for i in range(num_classes)]) / All_hot
    Recall_w_hot = sum(T_hot) / All_hot
    F1_macro_hot = 2 * Precision_w_hot * Recall_w_hot / (Precision_w_hot + Recall_w_hot)

    print('Naive:', F1_macro, 'Hot:', F1_macro_hot, 'Neighbours:', neighbours + 1)
    plot_data.append([neighbours + 1, F1_macro, F1_macro_hot])

x = [plot_data[i][0] for i in range(len(plot_data))]
y_1 = [plot_data[i][1] for i in range(len(plot_data))]
y_2 = [plot_data[i][2] for i in range(len(plot_data))]
plt.plot(x, y_1, y_2)
plt.legend(['F1_naive', 'F1_onehot'])
plt.show()

# breast
# variable
# manhattan + tricube + 2
# euclidean + gaussian + 1

# fixed
# euclidean + uniform + 0.4
# euclidean + logistic + 0.5

# wine
# variable
# manhattan + triweight + 3
# manhattan + triweight + 1