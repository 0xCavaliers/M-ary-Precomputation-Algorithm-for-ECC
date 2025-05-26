from Crypto.Util.number import *
from math import *
from sympy.ntheory.residue_ntheory import *
import random
from gmpy2 import iroot, log2
from scipy.special import lambertw

import hmac
import hashlib
import secrets
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes


zero = None


def W(x):
    return lambertw(x).real


def lowbit(x):
    return x & (-x)


def countr_zero(x):
    assert x != 0
    return lowbit(x).bit_length() - 1


eps = 0.1


class Elliptic_Curve:

    def __init__(self, p, a, b, G, n):
        self.p = p
        self.a = a
        self.b = b
        self.G = G
        self.n = n
        self.zero = zero

        self.l = floor((1 / 2 - eps) * log2(p))
        # print(f"{self.l=}")

        assert self.is_on_curve(G)
        # print("Yes")

    def is_on_curve(self, P):
        p = self.p
        a = self.a
        b = self.b
        x, y = P
        return y**2 % p == (x**3 + a * x + b) % p

    def add(self, p1, p2):
        p = self.p
        a = self.a

        if p1 == zero:
            return p2
        if p2 == zero:
            return p1
        (p1x, p1y), (p2x, p2y) = p1, p2
        if p1x == p2x and (p1y != p2y or p1y == 0):
            return zero
        if p1x == p2x:
            lam = (3 * p1x**2 + a) * inverse(2 * p1y, p) % p
        else:
            lam = (p2y - p1y) * inverse(p2x - p1x, p) % p
        x = (lam**2 - p1x - p2x) % p
        y = (lam * (p1x - x) - p1y) % p

        # print(type(x),type(3))

        return (x, y)

    def neg(self, p1):
        p = self.p
        x, y = p1
        return (x, (-y) % p)

    def sub(self, p1, p2):
        return self.add(p1, self.neg(p2))

    def qmul1(self, n, P):
        x, ans = P, zero
        while n:
            if n & 1:
                ans = self.add(ans, x)
            x = self.add(x, x)
            n >>= 1
        return ans

    def qmul2(self, n, P):
        x, ans = P, zero
        while n:
            if n & 1:
                a = 2 - n % 4
                if a == 1:
                    ans = self.add(ans, x)
                else:
                    ans = self.sub(ans, x)
                n -= a
            x = self.add(x, x)
            n >>= 1
        return ans

    def qmul3(self, N, P):
        K = 5

        mul_x = [zero] * (2**K)
        mul_x[0] = zero
        mul_x[1] = P
        mul_x[2] = self.add(P, P)

        for i in range(3, 2**K, 2):
            mul_x[i] = self.add(mul_x[i - 2], mul_x[2])

        ans = []
        for n in N:
            S = zero
            a = []
            while n:
                a.append(n % (2**K))
                n >>= K
            a.reverse()

            for ai in a:
                (u, s) = (0, 0)
                if ai:
                    s = countr_zero(ai)
                    u = ai >> s

                for j in range(K - s):
                    S = self.add(S, S)

                S = self.add(S, mul_x[u])

                for j in range(s):
                    S = self.add(S, S)

            ans.append(S)

        return ans

    # def qmul_opt(self, N, P):
    #     n = self.n
    #     Q = len(N)
    #     d = int(ceil(log(n) / (W(Q / e) + 1)))
    #     B, isInteger = gmpy2.iroot(n, d)
    #     B = int(B)
    #     if not isInteger:
    #         B += 1

    #     # print(f"d={d}\nB={B}")

    #     M = [[zero for j in range(B + 1)] for i in range(d)]
    #     for i in range(d):
    #         M[i][0] = zero
    #         for j in range(1, B + 1):
    #             if i == 0:
    #                 M[i][j] = self.add(M[i][j - 1], P)
    #             else:
    #                 M[i][j] = self.add(M[i][j - 1], M[i - 1][B])

    #     ans = []
    #     for k in N:
    #         S = zero
    #         for i in range(d):
    #             S = self.add(S, M[i][k % B])
    #             k //= B
    #         ans.append(S)

    #     return ans

    # def precompute_for_opt(self, P):。
    #     """
    #     n = self.n
    #     Q = len([P]) 
    #     d = int(ceil(log(n) / (W(Q / e) + 1)))
    #     B, isInteger = gmpy2.iroot(n, d)
    #     B = int(B)
    #     if not isInteger:
    #         B += 1

    #     M = [[None for j in range(B + 1)] for i in range(d)]
        
    #     for i in range(d):
    #         M[i][0] = zero  
    #         for j in range(1, B + 1):
    #             if i == 0:
    #                 M[i][j] = self.add(M[i][j - 1], P)
    #             else:
    #                 M[i][j] = self.add(M[i][j - 1], M[i - 1][B])
        
    #     self.precomputed_M = M
    #     self.precomputed_B = B
    #     self.precomputed_d = d
    #     # print(f"Precomputation completed: d={d}, B={B}")

    # def qmul_opt(self, N, P):
    #     n = self.n
    #     Q = len(N)
    #     d = self.precomputed_d
    #     B = self.precomputed_B
    #     M = self.precomputed_M

    #     ans = []
    #     for k in N:
    #         S = zero
    #         for i in range(d):
    #             S = self.add(S, M[i][k % B])
    #             k //= B
    #         ans.append(S)

    #     return ans

    def precompute_for_opt(self, P):

        n = self.n
        Q = len([P])
        d = int(ceil(log(n) / (W(Q / e) + 1)))
        B, isInteger = iroot(n, d)
        B = int(B)
        if not isInteger:
            B += 1

        M = [[None for j in range(B + 1)] for i in range(d)]
        
        for i in range(d):
            M[i][0] = zero 
            for j in range(1, B + 1):
                if i == 0:
                    M[i][j] = self.add_constant_time(M[i][j - 1], P)
                else:
                    M[i][j] = self.add_constant_time(M[i][j - 1], M[i - 1][B])
                
                if not self.is_on_curve(M[i][j]):
                    raise Exception("Fault detected in precomputed table")
        
        self.precomputed_M = self.randomize_matrix(M)
        self.precomputed_B = B
        self.precomputed_d = d

    def precompute_for_opt_without_W(self, P):
        """
        基于 M-ary 的预计算，并进行“只存 2^r 处关键点”的改良。
        """
        n = self.n
        Q = len([P])  
        d = int(ceil(log(n))) 

        B, isInteger = iroot(n, d)
        B = int(B)
        if not isInteger:
            B += 1

        big_M = [[None for j in range(B + 1)] for i in range(d)]

        big_M[0][0] = self.zero
        for j in range(1, B + 1):
            big_M[0][j] = self.add_constant_time(big_M[0][j - 1], P)

        for i in range(1, d):
            big_M[i][0] = self.zero
            # big_M[i][j] = big_M[i][j-1] + big_M[i-1][B]
            for j in range(1, B + 1):
                big_M[i][j] = self.add_constant_time(big_M[i][j - 1], big_M[i - 1][B])

        self.precomputed_M = []
        for i in range(d):
            row_i_powers = []
            max_r = int(floor(log2(B)))  # 2^max_r <= B
            for r in range(max_r + 1):
                power = 1 << r  # 2^r
                if power <= B:
                    row_i_powers.append( big_M[i][power] )
            self.precomputed_M.append(row_i_powers)

        self.precomputed_B = B
        self.precomputed_d = d

    def add_constant_time(self, P1, P2):
        return self.add(P1, P2)

    def randomize_matrix(self, matrix):
        randomized_matrix = [row[:] for row in matrix]
        random.shuffle(randomized_matrix) 
        for row in randomized_matrix:
            random.shuffle(row)
        return randomized_matrix

    def qmul_opt(self, N, P):
        n = self.n
        Q = len(N)
        d = self.precomputed_d
        B = self.precomputed_B
        M = self.precomputed_M

        ans = []
        for k in N:
            S = zero
            for i in range(d):
                S = self.add_constant_time(S, M[i][k % B])
                k //= B
            ans.append(S)

        return ans
    
    def qmul_opt_without_W(self, N_list, P):
        d = self.precomputed_d
        B = self.precomputed_B

        results = []
        for k in N_list:
            S = self.zero

            for i in range(d):
                digit_i = k % B 
                k //= B

                row_powers = self.precomputed_M[i]

                tmp = self.zero
                bit_length = len(row_powers)
                for r in range(bit_length):
                    bit_r = (digit_i >> r) & 1
                    if bit_r == 1:
                        tmp = self.add_constant_time(tmp, row_powers[r])

                S = self.add_constant_time(S, tmp)

            results.append(S)
        return results
    
    def precompute_for_comb(self, P, w, d):
        B = 2 ** w
        M = [[self.zero for _ in range(B)] for _ in range(d)]

        base = [P]
        for i in range(1, d):
            b = base[i - 1]
            for _ in range(w):
                b = self.add(b, b)
            base.append(b)

        for i in range(d):
            for j in range(1, B):
                M[i][j] = self.qmul1(j, base[i])

        self.comb_M = M
        self.comb_w = w
        self.comb_d = d


    def qmul_comb(self, N_list):
        M = self.comb_M
        w = self.comb_w
        d = self.comb_d
        B = 2 ** w

        results = []
        for k in N_list:
            k_bin = bin(k)[2:].zfill(w * d)
            a = []
            for i in range(d):
                start = len(k_bin) - (i + 1) * w
                end = len(k_bin) - i * w
                chunk = k_bin[start:end]
                a.append(int(chunk, 2))

            S = self.zero
            for i in range(d - 1, -1, -1):
                for _ in range(w):
                    S = self.add(S, S)
                if a[i] != 0:
                    S = self.add(S, M[i][a[i]])
            results.append(S)

        return results

    def precompute_for_fixed_window(self, P, w=5):
        self.precomputed_points = [self.zero] * (2 ** w)
        self.precomputed_points[1] = P
        for i in range(2, 2 ** w):
            self.precomputed_points[i] = self.add(self.precomputed_points[i - 1], P)
        
    def encode_tau_NAF(self, k, w):
        naf = []
        while k > 0:
            if k & 1:
                naf.append(1)
            else:
                naf.append(0)
            k >>= 1
        naf.reverse()

        i = 0
        while i < len(naf):
            if naf[i] == 1 and i + 1 < len(naf) and naf[i + 1] == 1:
                naf[i] = -1 
                naf[i + 1] = 0 
                i += 1
            i += 1
        return naf

    def qmul_fixed_window(self, k, P, w=5):
        naf = self.encode_tau_NAF(k, w)
        
        # Precompute [P, 2P, 3P, ..., (2^w-1)P]
        self.precompute_for_fixed_window(P, w)

        R = self.zero

        for i in range(len(naf) - 1, -1, -1):
            R = self.add(R, R)
            if naf[i] != 0:
                index = (naf[i] + (1 << (w - 1))) // 2
                R = self.add(R, self.precomputed_points[index])

        return R

    def sliding_window_mul(self, k, P, w):
        # Precompute: P, 2P, ..., (2^w - 1)P
        precomputed = [self.zero] + [P]
        for i in range(2, 2**w):
            precomputed.append(self.add(precomputed[i - 1], P))

        k_bin = bin(k)[2:]
        i = 0
        S = self.zero
        while i < len(k_bin):
            if k_bin[i] == '0':
                S = self.add(S, S)
                i += 1
            else:
                j = i + 1
                while j < len(k_bin) and (j - i) < w and k_bin[j] != '1':
                    j += 1
                j = min(i + w, len(k_bin))
                window = k_bin[i:j]
                window_value = int(window, 2)

                for _ in range(len(window)):
                    S = self.add(S, S)

                S = self.add(S, precomputed[window_value])
                i = j
        return S


    def montgomery_ladder(self, k, P):
        R0 = self.zero  
        R1 = P        
        k_bin = bin(k)[2:]

        for bit in k_bin:
            if bit == '0':
                R1 = self.add(R0, R1)  
                R0 = self.add(R0, R0)  
            else:
                R0 = self.add(R0, R1)  
                R1 = self.add(R1, R1) 

        return R0

    def encode_point(self, m: bytes):
        p = self.p
        a = self.a
        b = self.b
        l = self.l

        m = bytes_to_long(m)
        assert m < (1 << l)
        # x = 2**l * t + m
        # 0 <= x < p
        # 0 <= t <= (p - m - 1) // 2**l
        while 1:
            t = random.randint(0, (p - m - 1) >> l)
            x = (t << l) + m
            assert x < p
            rhs = (x**3 + a * x + b) % p
            if legendre_symbol(rhs, p) == 1:
                break
        y = sqrt_mod(rhs, p)
        return (x, y)

    def split_message(self, m):
        l = self.l
        k = l >> 3
        return [m[i : i + k] for i in range(0, len(m), k)]

    def encrypt_ECC(self, func, message: bytes, P_A):
        """
        Encrypt the whole message `m` (which can be very long)
        with the public key `P_A` (a point on the curve).
        """

        n = self.n
        G = self.G
        P_m_list = [self.encode_point(m) for m in self.split_message(message)]
        Q = len(P_m_list)

        # for m in self.split_message(message):
        #     print(m)
        # print()

        k_list = [random.randint(2, n - 1) for i in range(Q)]

        if func.__name__ in ["qmul3", "qmul_opt"]:
            C_1_list = func(k_list, G)
            C_2_list = func(k_list, P_A)
        else:
            C_1_list = [func(k, G) for k in k_list]
            C_2_list = [func(k, P_A) for k in k_list]

        # C_1_list = func(k_list, G)
        # C_2_list = func(k_list, P_A)

        # print(f"{func=}")
        # print(f"{func.__name__=}")

        """
        if func is self.qmul_opt: # False
            print("func is self.qmul_opt")
        
        if func.__name__ == "qmul_opt": # True
            print("func.__name__ == qmul_opt")
        """

        for i in range(Q):
            C_2_list[i] = self.add(C_2_list[i], P_m_list[i])
        C_m_list = list(zip(C_1_list, C_2_list))
        return C_m_list

    def decode_point(self, P) -> bytes:
        l = self.l
        x, y = P
        m = x & ((1 << l) - 1)
        m = long_to_bytes(m)
        return m

    def decrypt_ECC(self, C_m_list, d_A) -> bytes:
        """
        Decrypt the entire ciphertext (which can be very long)
        with the private key `d_A` to obtain the whole message `m`.
        """

        message = b""
        for C_1, C_2 in C_m_list:
            P_m = self.sub(C_2, self.qmul2(d_A, C_1))
            m = self.decode_point(P_m)
            message += m

            # print(m)

        return message

    """
    def keyDerivator(self, point, salt=b""):
        pointBytes = bytes(str(point[0]) + str(point[1]), "utf-8")
        randomKey = hmac.new(salt, pointBytes, hashlib.sha256).digest()
        return randomKey

    def encrypt_ECC(self, basePoint, msg, pubKey):

        ciphertextPrivKey = secrets.randbelow(self.p)
        sharedECCKey = self.scalarMultiplication(ciphertextPrivKey, pubKey)
        secretKey = self.keyDerivator(sharedECCKey)
        cipher = AES.new(secretKey, AES.MODE_EAX)
        ciphertext, tag = cipher.encrypt_and_digest(msg)
        ciphertextPubKey = self.scalarMultiplication(ciphertextPrivKey, basePoint)
        return (ciphertext, cipher.nonce, tag, ciphertextPubKey)

    def decrypt_ECC(self, encryptedMsg, privKey):
        ciphertext = encryptedMsg[0]
        nonce = encryptedMsg[1]
        tag = encryptedMsg[2]
        ciphertextPubKey = encryptedMsg[3]
        sharedECCKey = self.scalarMultiplication(privKey, ciphertextPubKey)
        secretKey = self.keyDerivator(sharedECCKey)
        cipher = AES.new(secretKey, AES.MODE_EAX, nonce)
        plaintext = cipher.decrypt_and_verify(ciphertext, tag)
        return plaintext
    """
