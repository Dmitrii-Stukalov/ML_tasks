from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


def read_dataset(number, type):
    if type == 'train':
        file = open(r'datasets/DT_txt/' + str(number).zfill(2) + '_train.txt')
    else:
        file = open(r'datasets/DT_txt/' + str(number).zfill(2) + '_test.txt')
    m, k = [int(i) for i in file.readline().split()]
    n = int(file.readline())
    X = []
    Y = []
    for i in range(n):
        line = [int(i) for i in file.readline().split()]
        X.append(line[:-1])
        Y.append(line[-1])
    return X, Y


def get_plot_data(index, criterion, splitter):
    x = []
    y_test = []
    y_train = []
    for depth in range(1, 35):
        X_train, Y_train = read_dataset(index, 'train')
        X_test, Y_test = read_dataset(index, 'test')

        clf = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=depth)
        clf.fit(X_train, Y_train)
        y_test.append(clf.score(X_test, Y_test))
        y_train.append(clf.score(X_train, Y_train))
        x.append(clf.get_depth())
    return x, y_test, y_train


criterions = ['gini', 'entropy']
splitters = ['best', 'random']
min_depth = 100
min_criterion = ''
min_splitter = ''
index_min_depth = -1

max_depth = 0
max_criterion = ''
max_splitter = ''
index_max_depth = -1

for i in range(1, 22):
    X_train, Y_train = read_dataset(i, 'train')
    X_test, Y_test = read_dataset(i, 'test')
    best_score = 0
    best_depth = 0
    best_criterion = ''
    best_splitter = ''
    for criterion in criterions:
        for splitter in splitters:
            for depth in range(1, 30):
                clf = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=depth)
                clf.fit(X_train, Y_train)
                current_score = clf.score(X_test, Y_test)
                if current_score > best_score:
                    best_score = current_score
                    best_depth = clf.get_depth()
                    best_criterion = criterion
                    best_splitter = splitter
    print('Dataset', i, 'has score', best_score, 'with depth', best_depth, ', criterion', best_criterion, ', splitter',
          best_splitter)
    if best_depth > max_depth:
        max_depth = best_depth
        index_max_depth = i
        max_criterion = best_criterion
        max_splitter = best_splitter
    if best_depth < min_depth:
        min_depth = best_depth
        index_min_depth = i
        min_criterion = best_criterion
        min_splitter = best_splitter

print('Dataset', index_max_depth, 'has max depth', max_depth)
print('Dataset', index_min_depth, 'has min depth', min_depth)

x, y_test, y_train = get_plot_data(index_max_depth, max_criterion, max_splitter)
plt.plot(x, y_train, label='max train')
plt.plot(x, y_test, label='max test')
plt.legend()
plt.show()

x, y_test, y_train = get_plot_data(index_min_depth, min_criterion, min_splitter)
plt.plot(x, y_train, label='min train')
plt.plot(x, y_test, label='min test')
plt.legend()
plt.show()
