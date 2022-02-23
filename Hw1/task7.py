def process(l):
    b=[]
    for i in set([i for i in l]):
        b += [i*i]
    b.sort(reverse=1)
    return b