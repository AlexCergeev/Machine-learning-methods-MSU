def primes():
    yield 2

if __name__ == "__main__":
    for i in primes():
        print(i)
        if i == 2:
            break
