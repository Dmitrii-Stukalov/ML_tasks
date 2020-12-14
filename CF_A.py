n, m, k = [int(i) for i in input().split()]
lst = [int(i) for i in input().split()]
srt_lst = sorted(lst)

ans = [0] * k
for i in range(k):
    ans[i] = []

for i in range(n):
    index = lst.index(srt_lst[i])
    lst[index] = -1
    ans[i % k].append(index + 1)

for items in ans:
    print(len(items), end=' ')
    for item in items:
        print(item, end=' ')
    print()
