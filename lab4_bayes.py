import os
from math import log

import matplotlib.pyplot as plt


class Message:
    def __init__(self, subject, text, is_legit):
        self.subject = subject
        self.text = text
        self.is_legit = is_legit

    def split_ngrams(self, n):
        full_message = self.subject + self.text
        grams = []
        for i in range(len(full_message) - n + 1):
            gram = full_message[i: i + n]
            str_gram = ''
            for word in gram:
                str_gram += str(word) + ' '
            grams.append(str_gram[:-1])
        return grams


def calculate_probabilities(message, num_spam, num_legit, spam_data, legit_data, alpha, n, lambda_legit=1, lambda_spam=1):
    spam_probability = log(lambda_spam * num_spam / (num_spam + num_legit))
    legit_probability = log(lambda_legit * num_legit / (num_spam + num_legit))

    for gram in message.split_ngrams(n):
        if gram in spam_data.keys():
            spam_probability += log((spam_data[gram] + alpha) / (num_spam + 2 * alpha))
        else:
            spam_probability += log(alpha / (num_spam + 2 * alpha))
        if gram in legit_data.keys():
            legit_probability += log((legit_data[gram] + alpha) / (num_legit + 2 * alpha))
        else:
            legit_probability += log(alpha / (num_legit + 2 * alpha))
    return spam_probability, legit_probability


def calculate_accuracy(messages, n, alpha, lambda_legit=1, lambda_spam=1):
    legit_in_spam = False
    roc_tmp = []
    accuracy = 0
    for test_index in range(10):
        spam_data = {}
        legit_data = {}
        num_legit = 0
        num_spam = 0
        for i in range(10):
            if i == test_index:
                continue
            for msg in messages[i]:
                if msg.is_legit:
                    for gram in msg.split_ngrams(n):
                        legit_data[gram] = (legit_data[gram] + 1 if gram in legit_data else 1)
                    num_legit += 1
                else:
                    for gram in msg.split_ngrams(n):
                        spam_data[gram] = (spam_data[gram] + 1 if gram in spam_data else 1)
                    num_spam += 1

        ok = 0
        for msg in messages[test_index]:
            spam_probability, legit_probability = calculate_probabilities(msg, num_spam, num_legit, spam_data,
                                                                          legit_data, alpha, n, lambda_legit, lambda_spam)
            if msg.is_legit:
                roc_tmp.append((spam_probability - legit_probability, 1))
            else:
                roc_tmp.append((spam_probability - legit_probability, -1))

            if msg.is_legit and legit_probability >= spam_probability or \
                    not msg.is_legit and legit_probability < spam_probability:
                ok += 1
            if legit_probability < spam_probability and msg.is_legit:
                legit_in_spam = True
        accuracy += ok / len(messages[test_index])
    accuracy /= 10
    return accuracy, roc_tmp, legit_in_spam

messages = []
for i in range(1, 11):
    part = []
    for name in (os.listdir('datasets/messages/part' + str(i))):
        file = open(r'datasets/messages/part' + str(i) + '/' + name)
        subject = [int(item) for item in file.readline().split(':')[1].split()]
        text = [int(item) for item in file.read().split()]
        message = Message(subject, text, 'legit' in name)
        part.append(message)
    messages.append(part)

alphas = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
best_accuracy = 0
best_alpha = 0
best_n = 0
roc_tmp_best = []
info = ''
for n in range(1, 4):
    for alpha in alphas:
        accuracy, roc_tmp, _ = calculate_accuracy(messages, n, alpha)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_n = n
            best_alpha = alpha
            roc_tmp_best = roc_tmp
            info = 'alpha ' + str(alpha) + ' ' + str(n) + 'gram accuracy ' + str(accuracy)

roc = list(reversed(sorted(roc_tmp_best, key=lambda x: x[0])))
num_legit = 0
num_spam = 0
for item in roc:
    if item[1] == 1:
        num_legit += 1
    else:
        num_spam += 1
FPR = [0] * len(roc)
TPR = [0] * len(roc)
for i in range(len(roc) - 1):
    if roc[i][1] == -1:
        FPR[i + 1] = FPR[i] + 1 / num_spam
        TPR[i + 1] = TPR[i]
    else:
        FPR[i + 1] = FPR[i]
        TPR[i + 1] = TPR[i] + 1 / num_legit
plt.plot(TPR, FPR)
plt.show()
print('best', info)

# best_n = 2
# best_alpha = 1e-10
lambda_spam = 1
lambda_legit = 1
legit_in_spam = True
accs = []
lambdas = []
while legit_in_spam:
    accuracy, _, legit_in_spam = calculate_accuracy(messages, best_n, best_alpha, lambda_legit, lambda_spam)
    accs.append(accuracy)
    lambdas.append(log(lambda_legit))
    lambda_legit *= 1e10
plt.plot(lambdas, accs)
plt.show()
print(lambdas[-1])
