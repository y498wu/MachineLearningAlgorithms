def fib(n):
    dict = {}
    for k in range(1, n+1):
        if k <= 2:
           f = 1
        else:
            f = dict[k-1] + dict[k-2]
        dict[k] = f
    return dict[n]
