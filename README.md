# numbt

NumPy-style numerical computing library for MoonBit.

Built on BLAS (Apple Accelerate framework on macOS) for high-performance matrix operations.

## Features

- Vec/Mat views over Float arrays (zero-copy)
- BLAS-accelerated matrix multiplication (`cblas_sgemm`)
- LAPACK SVD decomposition
- Element-wise operations
- Softmax, ReLU activation functions

## Requirements

- MoonBit native backend
- BLAS/LAPACK library:
  - **macOS**: Apple Accelerate (built-in)
  - **Linux**: OpenBLAS + LAPACK (`sudo apt-get install libopenblas-dev liblapack-dev`)

## Platform Configuration

Add the appropriate link flags to your package's `moon.pkg`:

**macOS:**
```
options(
  link: { "native": { "cc-link-flags": "-framework Accelerate" } },
)
```

**Linux:**
```
options(
  link: { "native": { "cc-link-flags": "-lopenblas -llapack -lm" } },
)
```

## Installation

Add to `moon.mod.json`:

```json
{
  "deps": {
    "mizchi/numbt": "0.1.0"
  }
}
```

Then run:

```bash
moon update
```

## Usage

```moonbit
// Create views over arrays
let data : Array[Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
let mat = @numbt.mat_view(data, rows=2, cols=3)
let vec = @numbt.vec_view(data, offset=0, len=3)

// Matrix multiplication
let a = @numbt.mat_view([1.0, 2.0, 3.0, 4.0], 2, 2)
let b = @numbt.mat_view([5.0, 6.0, 7.0, 8.0], 2, 2)
let c = a.matmul(b)

// Softmax
let logits = @numbt.vec_view([1.0, 2.0, 3.0], 0, 3)
let probs = @numbt.vec_view(Array::make(3, 0.0), 0, 3)
@numbt.softmax_into(input=logits, output=probs)
```

## API

### Vec operations

- `vec_view(data, offset, len)` - Create a view
- `vec_add_into(left, right, output~)` - Element-wise addition
- `vec_sub_into(left, right, output~)` - Element-wise subtraction
- `vec_mul_into(left, right, output~)` - Element-wise multiplication
- `softmax_into(input~, output~)` - Softmax activation
- `relu_into(input~, output~)` - ReLU activation

### Mat operations

- `mat_view(data, rows, cols)` - Create a view
- `mat_matmul(a, b)` - Matrix multiplication
- `Mat::matmul(self, other)` - Method syntax
- `matmul_vec_bias_into(weight, input, bias, output~)` - Linear layer forward

### LAPACK (LapackMat)

`LapackMat` is the `FixedArray[Byte]`-backed matrix type. Its raw byte
layout matches what BLAS / LAPACK / vDSP expect, so calls into the
native side are zero-copy.

- `fmat_zeros(rows, cols)` / `fmat_eye(n)` / `fmat_randn(rows, cols)` - constructors
- `fmat_from_mat(m)` / `fmat_to_mat(fm)` - conversion to / from `Mat`
- `fmat_matmul(a, b)` - BLAS sgemm (matrix multiply)
- `fmat_svd` / `fmat_eig` / `fmat_cholesky` / `fmat_qr` / `fmat_det` / `fmat_lstsq` - LAPACK
- `fmat_inv` / `fmat_solve` - LU-based linear solve

### LapackMat: Accelerate vDSP element-wise + reductions (SIMD)

Zero-copy SIMD via Apple Accelerate's vDSP. ~10-30x faster than the
equivalent scalar implementation on the same storage:

- `fmat_add` / `fmat_sub` / `fmat_mul` / `fmat_div` - element-wise binary ops
- `fmat_add_into` / etc. - in-place variants (no allocation)
- `fmat_add_scalar` / `fmat_mul_scalar` - broadcast scalar ops
- `fmat_sum` / `fmat_mean` / `fmat_max` / `fmat_min` - reductions

Note: the equivalent ops on the `Array[Float]`-backed `Vec` / `Mat`
types stay scalar by default. MoonBit's C FFI requires the buffer to
be `FixedArray[Byte]`, and the `Array[Float]` -> bytes round-trip cost
erases the SIMD win even at N = 1M+. For hot inner loops, convert
once with `fmat_from_mat` and stay on `LapackMat`.

## License

Apache-2.0
