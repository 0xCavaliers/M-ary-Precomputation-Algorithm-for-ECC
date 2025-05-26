import tracemalloc
import random
from elliptic_curve import Elliptic_Curve
import curve_parameters

# Initialize ECC
curve_name = "secp256k1" # secp256k1 secp384r1 secp521r1
param = curve_parameters.curves[curve_name]
ecc = Elliptic_Curve(*param.values())
G = ecc.G
n = ecc.n

# Define algorithms
algo_funcs = {
    "bin": lambda k_list: [ecc.qmul1(k, G) for k in k_list],
    "naf": lambda k_list: [ecc.qmul2(k, G) for k in k_list],
    "ary": lambda k_list: ecc.qmul3(k_list, G),
    "opt": lambda k_list: ecc.qmul_opt(k_list, G),
    "opt_without_W": lambda k_list: ecc.qmul_opt_without_W(k_list, G),
    "sliding": lambda k_list: [ecc.sliding_window_mul(k, G, 8) for k in k_list],
    "montgomery": lambda k_list: [ecc.montgomery_ladder(k, G) for k in k_list],
    "comb": lambda k_list: ecc.qmul_comb(k_list),
    "fixed": lambda k_list: [ecc.qmul_fixed_window(k, G) for k in k_list],
}

# Test Number
num_tests = 10000
k_list = [random.randint(2, n - 1) for _ in range(num_tests)]

# Store Result
memory_usage = {}

for algo_name, func in algo_funcs.items():
    print(f"Running memory test for: {algo_name}")

    if algo_name == "opt":
        ecc.precompute_for_opt(G)
    elif algo_name == "opt_without_W":
        ecc.precompute_for_opt_without_W(G)
    elif algo_name == "comb":
        ecc.precompute_for_comb(G, w=4, d=64)
    elif algo_name == "fixed":
        ecc.precompute_for_fixed_window(G, w=4)

    tracemalloc.start()
    _ = func(k_list)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    memory_usage[algo_name] = peak / (1024 * 1024)

print("\nMemory usage comparison:")
for algo, mem in memory_usage.items():
    print(f"{algo:<15} : {mem:.6f} MB")
