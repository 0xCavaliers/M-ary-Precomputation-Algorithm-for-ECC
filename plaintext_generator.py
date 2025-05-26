import random

seed = 998244353

if __name__ == "__main__":
    random.seed(seed)
    n = 50
    with open(r"plain_text.txt", "wb") as f:
        random_bytes = bytes([random.randint(ord("a"), ord("z")) for i in range(n)])
        f.write(random_bytes)
