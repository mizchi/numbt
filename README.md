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

- macOS with Accelerate framework (native target only)
- MoonBit native backend

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

- `lapack_mat_new(rows, cols)` - Create a new matrix
- `lapack_mat_svd(mat)` - SVD decomposition

## License

Apache-2.0
