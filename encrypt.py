import time
import random
from elliptic_curve import Elliptic_Curve
import curve_parameters
import tracemalloc

curve_name = "secp256k1" # secp256k1 secp384r1 secp521r1
param = curve_parameters.curves[curve_name]
ecc = Elliptic_Curve(*param.values())
G = ecc.G
n = ecc.n
d_A = random.randint(2, n - 1) 
P_A = ecc.qmul2(d_A, G) 


def run_bin(N):
    return ecc.qmul1(N, G)

def run_naf(N):
    return ecc.qmul2(N, G)

def run_ary(N):
    return ecc.qmul3([N], G)[0]

def run_opt(N):
    return ecc.qmul_opt([N], G)[0]

def run_opt_without_W(N):
    return ecc.qmul_opt_without_W([N], G)[0]

def run_sliding_window(N):
    window_size = 4
    return ecc.sliding_window_mul(N, G, window_size)

def run_montgomery(N):
    return ecc.montgomery_ladder(N, G)

def run_comb(N):
    return ecc.qmul_comb([N])[0]

def run_fixed_window(N):
    return ecc.qmul_fixed_window(N, G)

algo_funcs = {
    "bin": run_bin,
    "naf": run_naf,
    "ary": run_ary,
    "opt": run_opt,
    "opt_without_W": run_opt_without_W,
    "sliding": run_sliding_window,
    "montgomery": run_montgomery,  
    "comb": run_comb,
    "fixed": run_fixed_window,
}


def main():
    with open("plain_text.txt", "rb") as f:
        message = f.read()

    N = int.from_bytes(message, byteorder='big') % n

    algo = "opt"  # "bin", "naf", "ary", "opt", "opt_without_W", "sliding", "montgomery", "comb", "fixed"
    encrypt_func = algo_funcs[algo]

    if algo == "opt":
        # print("Performing precomputation for M-ary algorithm...")
        precompute_start = time.perf_counter()
        # tracemalloc.start()
        ecc.precompute_for_opt(G)
        # current, peak = tracemalloc.get_traced_memory()
        # tracemalloc.stop()
        precompute_end = time.perf_counter()
        precompute_duration = precompute_end - precompute_start
        # memory_peak_mb = peak / (1024 * 1024)
        print(f"Precomputation time for opt: {precompute_duration:.6f} seconds")
        # print(f"Memory used during opt precomputation: {memory_peak_mb:.6f} MB")
    elif algo == "opt_without_W":
        precompute_start = time.perf_counter()
        tracemalloc.start()
        ecc.precompute_for_opt_without_W(G)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        precompute_end = time.perf_counter()
        # precompute_duration = precompute_end - precompute_start
        memory_peak_mb = peak / (1024 * 1024)
        # print(f"Precomputation time for opt without W: {precompute_duration:.6f} seconds")
        print(f"Memory used during opt precomputation: {memory_peak_mb:.6f} MB")
    elif algo == "comb":
        precompute_start = time.perf_counter()
        tracemalloc.start()
        ecc.precompute_for_comb(G, w=4, d=48) 
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        precompute_end = time.perf_counter()
        # precompute_duration = precompute_end - precompute_start
        memory_peak_mb = peak / (1024 * 1024)
        # print(f"Precomputation time for comb: {precompute_duration:.6f} seconds")
        print(f"Memory used during opt precomputation: {memory_peak_mb:.6f} MB")
    elif algo == "fixed":
        precompute_start = time.perf_counter()
        tracemalloc.start()
        ecc.precompute_for_fixed_window(G, w=4) 
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        precompute_end = time.perf_counter()
        # precompute_duration = precompute_end - precompute_start
        memory_peak_mb = peak / (1024 * 1024)
        # print(f"Precomputation time for fixed: {precompute_duration:.6f} seconds")
        print(f"Memory used during opt precomputation: {memory_peak_mb:.6f} MB")
    elif algo == "sliding":
        tracemalloc.start()
        run_sliding_window(N)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        precompute_end = time.perf_counter()
        memory_peak_mb = peak / (1024 * 1024)
        print(f"Memory used during opt precomputation: {memory_peak_mb:.6f} MB")

    total_encryption_time = 0.0

    for i in range(10):
        start_time = time.perf_counter()
        encrypted_message = encrypt_func(N)
        end_time = time.perf_counter()

        encryption_time = end_time - start_time
        # print(f"Encryption time using {algo} (iteration {i+1}): {encryption_time:.6f} seconds")

        total_encryption_time += encryption_time

    # print(f"Total encryption time using {algo}: {total_encryption_time:.6f} seconds")

    encrypted_data = str(encrypted_message)
    with open("encrypted.txt", "w") as f:
        f.write(encrypted_data)

if __name__ == "__main__":
    main()
