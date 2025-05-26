# M-ary Precomputation-Based Accelerated Scalar Multiplication Algorithms for Enhanced Elliptic Curve Cryptography

This repository contains a partial implementation of our M-ary Precomputation-Based ECC implementation with side-channel protection mechanisms. The full implementation, including the complete verification framework, will be released upon paper acceptance.

## Overview

This implementation includes several scalar multiplication algorithms with different side-channel protection strategies:

- Basic binary method (`qmul1`)
- Improved binary method (`qmul2`)
- M-ary method (`qmul3`)
- Optimized M-ary method (`qmul_opt`)
- M-ary with binary-sparse optimization (`qmul_opt_without_W`)
- Sliding window method (`sliding_window_mul`)
- Montgomery ladder (`montgomery_ladder`)
- Comb method (`qmul_comb`)
- Fixed window method (`qmul_fixed_window`)

## Requirements

```bash
pip install numpy matplotlib pycryptodome tqdm sympy
```

## Usage Examples

### 1. TVLA Analysis

To perform TVLA (Test Vector Leakage Assessment) analysis on different algorithms:

```bash
python test_qmul_opt.py
```

This will generate TVLA results for the specified algorithm and save the plot in the `results` directory.

### 2. Memory Usage Analysis

To compare memory usage across different algorithms:

```bash
python ecc_memory.py
```

This will output memory usage statistics for each algorithm.

### 3. Generate Test Data

To generate test plaintext:

```bash
python plaintext_generator.py
```

This will create a `plain_text.txt` file with random test data.

## Supported Curves

The implementation supports several standard curves:

- secp256k1
- secp384r1
- secp521r1

## Code Structure

- `elliptic_curve.py`: Core ECC implementation with various scalar multiplication algorithms
- `test_qmul_opt.py`: TVLA analysis framework
- `ecc_memory.py`: Memory usage analysis tool
- `curve_parameters.py`: Curve parameters for different standards
- `plaintext_generator.py`: Test data generator

## Side-Channel Protection Features

The implementation includes several side-channel protection mechanisms:

1. Constant-time operations
2. Randomization techniques
3. Fault injection protection
4. Power analysis countermeasures

## Future Work

Upon paper acceptance, we will release:

1. Complete verification framework
2. Additional side-channel protection mechanisms
3. Performance optimization tools
4. Comprehensive test suite
5. Detailed documentation

## Citation

If you use this code in your research, please cite our paper (to be added upon acceptance).

## License

This code is released under the MIT License. See the LICENSE file for details.
