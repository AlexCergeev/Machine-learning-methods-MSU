def primes():
    for n in range(2,10000000000000000000):
        count = 0
        for i in range(2,int(n**0.5)+1):
            if n % i == 0: 
                count += 1
                break

        if count == 0: yield n