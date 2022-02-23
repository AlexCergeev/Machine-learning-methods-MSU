def is_palindrome(x):
    a = x
    count = 0
    while a > 0:
        a = a // 10
        count += 1
    answer = 1
    leth = count
    for i in range(count):
        left = x % 10 ** (count - i) // 10 ** (count - i - 1)
        right = x % 10 ** (i + 1) // 10 ** (i)
        if left != right:
            answer = 0
        leth -= 2
        if leth <= 0: break

    if answer:
        return 'YES'
    else:
        return 'NO'


