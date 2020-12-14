def read_train(K):
    n = int(input())
    words = set()
    data = {}
    classes = [0] * K
    for i in range(n):
        line = input().split()
        k = int(line[0]) - 1
        classes[k] += 1
        words = words.union(set(line[2:]))
        if k in data:
            data[k] += list(set(line[2:]))
        else:
            data[k] = list(set(line[2:]))
    return data, classes, words


def count_p_words(words, data, classes, alpha):
    p_words = {}
    for word in words:
        for k in data.keys():
            if k not in p_words:
                p_words[k] = []
            class_words = data[k]
            p_words[k].append((word, (class_words.count(word) + alpha) / (classes[k] + alpha * 2)))
    return p_words


def read_test():
    M = int(input())
    data = []
    for i in range(M):
        data.append(list(set(input().split()[1:])))
    return data


K = int(input())

lam = [int(i) for i in input().split()]

alpha = int(input())

train_data, num_classes, all_words = read_train(K)

p_words = count_p_words(all_words, train_data, num_classes, alpha)
test_data = read_test()

for line in test_data:
    class_probability = [0] * K
    for k in range(K):
        counter = lam[k] * num_classes[k] / sum(num_classes)
        for pair in p_words[k]:
            if pair[0] in line:
                counter *= pair[1]
            else:
                counter *= (1 - pair[1])
        class_probability[k] = counter
    ans = []
    for i in range(K):
        ans.append(class_probability[i] / sum(class_probability))
    print(*ans)
