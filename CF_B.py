n = int(input())

CM = []
for i in range(n):
    CM.append([int(j) for j in input().split()])

T = [CM[i][i] for i in range(n)]
C = [sum(i) for i in CM]
P = [sum(i) for i in zip(*CM)]
All = sum(C)

Precision_w = sum([(T[i] * C[i] / P[i]) if P[i] != 0 else 0 for i in range(n)]) / All
Recall_w = sum(T) / All
F1_macro = 2 * Precision_w * Recall_w / (Precision_w + Recall_w)
print(F1_macro)

Precision = [T[i] / C[i] if C[i] != 0 else 0 for i in range(n)]
Recall = [T[i] / P[i] if P[i] != 0 else 0 for i in range(n)]
F1 = [2 * Precision[i] * Recall[i] / (Precision[i] + Recall[i]) if Precision[i] != 0 and Recall[i] != 0 else 0 for i in range(n)]
F1_micro = sum(F1[i] * C[i] for i in range(n)) / All
print(F1_micro)
