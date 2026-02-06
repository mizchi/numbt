// ============================================================================
// FFI declarations for Accelerate framework optimization
// ============================================================================

///|
extern "C" fn numbt_vsort_asc(data : FixedArray[Byte], n : Int) -> Unit = "numbt_vsort_asc"

///|
extern "C" fn numbt_vsort_desc(data : FixedArray[Byte], n : Int) -> Unit = "numbt_vsort_desc"

///|
extern "C" fn numbt_vexp(
  src : FixedArray[Byte],
  dst : FixedArray[Byte],
  n : Int,
) -> Unit = "numbt_vexp"

///|
extern "C" fn numbt_outer(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  m : Int,
  n : Int,
) -> Unit = "numbt_outer"

///|
extern "C" fn numbt_seed(seed : Int) -> Unit = "numbt_seed"

///|
extern "C" fn numbt_rand(out : FixedArray[Byte], n : Int) -> Unit = "numbt_rand"

///|
extern "C" fn numbt_randn(out : FixedArray[Byte], n : Int) -> Unit = "numbt_randn"

///|
extern "C" fn numbt_inv(a : FixedArray[Byte], n : Int) -> Int = "numbt_inv"

///|
extern "C" fn numbt_solve(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  n : Int,
) -> Int = "numbt_solve"

///|
extern "C" fn numbt_svd(
  a : FixedArray[Byte],
  u : FixedArray[Byte],
  s : FixedArray[Byte],
  vt : FixedArray[Byte],
  m : Int,
  n : Int,
) -> Int = "numbt_svd"

///|
extern "C" fn numbt_eig_symmetric(
  a : FixedArray[Byte],
  w : FixedArray[Byte],
  n : Int,
) -> Int = "numbt_eig_symmetric"

///|
extern "C" fn numbt_cholesky(a : FixedArray[Byte], n : Int) -> Int = "numbt_cholesky"

///|
extern "C" fn numbt_qr(
  a : FixedArray[Byte],
  q : FixedArray[Byte],
  r : FixedArray[Byte],
  m : Int,
  n : Int,
) -> Int = "numbt_qr"

///|
extern "C" fn numbt_det(
  a : FixedArray[Byte],
  det : FixedArray[Byte],
  n : Int,
) -> Int = "numbt_det"

///|
extern "C" fn numbt_lstsq(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  m : Int,
  n : Int,
) -> Int = "numbt_lstsq"

///|
fn float_arr_to_bytes(arr : Array[Float]) -> FixedArray[Byte] {
  let bytes : FixedArray[Byte] = FixedArray::make(arr.length() * 4, b'\x00')
  for i, v in arr {
    let bits = v.reinterpret_as_int()
    bytes[i * 4] = (bits & 0xFF).to_byte()
    bytes[i * 4 + 1] = ((bits >> 8) & 0xFF).to_byte()
    bytes[i * 4 + 2] = ((bits >> 16) & 0xFF).to_byte()
    bytes[i * 4 + 3] = ((bits >> 24) & 0xFF).to_byte()
  }
  bytes
}

///|
fn bytes_to_float_arr(bytes : FixedArray[Byte], count : Int) -> Array[Float] {
  let arr : Array[Float] = Array::make(count, Float::from_int(0))
  for i in 0..<count {
    let bits = bytes[i * 4].to_int() |
      (bytes[i * 4 + 1].to_int() << 8) |
      (bytes[i * 4 + 2].to_int() << 16) |
      (bytes[i * 4 + 3].to_int() << 24)
    arr[i] = Float::reinterpret_from_int(bits)
  }
  arr
}

// ============================================================================
// LapackMat - FixedArray[Byte] based for LAPACK operations
// ============================================================================

///|
/// Matrix type optimized for LAPACK operations (SVD, eigenvalues, etc).
///
/// Uses FixedArray[Byte] for zero-copy interop with C LAPACK functions.
/// Convert from Mat using `fmat_from_mat`, back to Mat using `fmat_to_mat`.
pub struct LapackMat {
  data : FixedArray[Byte]
  rows : Int
  cols : Int
}

// Note: FMat has been renamed to LapackMat for clarity.
// The old name is kept as a deprecated alias in the documentation.

///|
pub fn fmat_zeros(rows : Int, cols : Int) -> LapackMat {
  { data: FixedArray::make(rows * cols * 4, b'\x00'), rows, cols }
}

///|
pub fn fmat_from_mat(m : Mat) -> LapackMat {
  let bytes = float_arr_to_bytes(m.data)
  { data: bytes, rows: m.rows, cols: m.cols }
}

///|
pub fn fmat_to_mat(fm : LapackMat) -> Mat {
  let arr = bytes_to_float_arr(fm.data, fm.rows * fm.cols)
  mat_view(arr, fm.rows, fm.cols)
}

///|
pub fn fmat_clone(fm : LapackMat) -> LapackMat {
  let new_data : FixedArray[Byte] = FixedArray::make(fm.data.length(), b'\x00')
  for i = 0; i < fm.data.length(); i = i + 1 {
    new_data[i] = fm.data[i]
  }
  { data: new_data, rows: fm.rows, cols: fm.cols }
}

///|
pub fn fmat_rows(fm : LapackMat) -> Int {
  fm.rows
}

///|
pub fn fmat_cols(fm : LapackMat) -> Int {
  fm.cols
}

///|
/// Matrix inverse on LapackMat (in-place, returns success)
pub fn fmat_inv(fm : LapackMat) -> Bool {
  if fm.rows != fm.cols {
    return false
  }
  numbt_inv(fm.data, fm.rows) == 0
}

///|
/// Solve A @ x = b on LapackMat (modifies both a and b in-place)
/// After call, b contains solution x
pub fn fmat_solve(a : LapackMat, b : FixedArray[Byte]) -> Bool {
  if a.rows != a.cols {
    return false
  }
  numbt_solve(a.data, b, a.rows) == 0
}

///|
/// Create LapackMat identity matrix
pub fn fmat_eye(n : Int) -> LapackMat {
  let fm = fmat_zeros(n, n)
  let one_bits = (1.0 : Float).reinterpret_as_int()
  for i = 0; i < n; i = i + 1 {
    let idx = (i * n + i) * 4
    fm.data[idx] = (one_bits & 0xFF).to_byte()
    fm.data[idx + 1] = ((one_bits >> 8) & 0xFF).to_byte()
    fm.data[idx + 2] = ((one_bits >> 16) & 0xFF).to_byte()
    fm.data[idx + 3] = ((one_bits >> 24) & 0xFF).to_byte()
  }
  fm
}

///|
/// Create LapackMat from random values
pub fn fmat_randn(rows : Int, cols : Int) -> LapackMat {
  let n = rows * cols
  let data : FixedArray[Byte] = FixedArray::make(n * 4, b'\x00')
  numbt_randn(data, n)
  { data, rows, cols }
}

///|
/// LapackMat matrix multiply using BLAS (C = A @ B)
extern "C" fn numbt_sgemm(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  c : FixedArray[Byte],
  m : Int,
  n : Int,
  k : Int,
) -> Unit = "numbt_sgemm"

///|
pub fn fmat_matmul(a : LapackMat, b : LapackMat) -> LapackMat {
  if a.cols != b.rows {
    panic()
  }
  let c = fmat_zeros(a.rows, b.cols)
  numbt_sgemm(a.data, b.data, c.data, a.rows, b.cols, a.cols)
  c
}

///|
/// LapackMat add (C = A + B)
pub fn fmat_add(a : LapackMat, b : LapackMat) -> LapackMat {
  if a.rows != b.rows || a.cols != b.cols {
    panic()
  }
  let c = fmat_zeros(a.rows, a.cols)
  let n = a.rows * a.cols
  for i = 0; i < n; i = i + 1 {
    let idx = i * 4
    let a_bits = a.data[idx].to_int() |
      (a.data[idx + 1].to_int() << 8) |
      (a.data[idx + 2].to_int() << 16) |
      (a.data[idx + 3].to_int() << 24)
    let b_bits = b.data[idx].to_int() |
      (b.data[idx + 1].to_int() << 8) |
      (b.data[idx + 2].to_int() << 16) |
      (b.data[idx + 3].to_int() << 24)
    let sum = Float::reinterpret_from_int(a_bits) +
      Float::reinterpret_from_int(b_bits)
    let sum_bits = sum.reinterpret_as_int()
    c.data[idx] = (sum_bits & 0xFF).to_byte()
    c.data[idx + 1] = ((sum_bits >> 8) & 0xFF).to_byte()
    c.data[idx + 2] = ((sum_bits >> 16) & 0xFF).to_byte()
    c.data[idx + 3] = ((sum_bits >> 24) & 0xFF).to_byte()
  }
  c
}

///|
/// LapackMat transpose
pub fn fmat_transpose(a : LapackMat) -> LapackMat {
  let t = fmat_zeros(a.cols, a.rows)
  for i = 0; i < a.rows; i = i + 1 {
    for j = 0; j < a.cols; j = j + 1 {
      let src_idx = (i * a.cols + j) * 4
      let dst_idx = (j * a.rows + i) * 4
      t.data[dst_idx] = a.data[src_idx]
      t.data[dst_idx + 1] = a.data[src_idx + 1]
      t.data[dst_idx + 2] = a.data[src_idx + 2]
      t.data[dst_idx + 3] = a.data[src_idx + 3]
    }
  }
  t
}

///|
/// Helper: create FixedArray[Byte] for n floats
fn fvec_zeros(n : Int) -> FixedArray[Byte] {
  FixedArray::make(n * 4, b'\x00')
}

///|
/// Helper: read float from FixedArray[Byte] at index i
fn fvec_get(data : FixedArray[Byte], i : Int) -> Float {
  let idx = i * 4
  let bits = data[idx].to_int() |
    (data[idx + 1].to_int() << 8) |
    (data[idx + 2].to_int() << 16) |
    (data[idx + 3].to_int() << 24)
  Float::reinterpret_from_int(bits)
}

///|
/// Helper: set float in FixedArray[Byte] at index i
fn fvec_set(data : FixedArray[Byte], i : Int, v : Float) -> Unit {
  let idx = i * 4
  let bits = v.reinterpret_as_int()
  data[idx] = (bits & 0xFF).to_byte()
  data[idx + 1] = ((bits >> 8) & 0xFF).to_byte()
  data[idx + 2] = ((bits >> 16) & 0xFF).to_byte()
  data[idx + 3] = ((bits >> 24) & 0xFF).to_byte()
}

///|
/// Helper: convert FixedArray[Byte] to Vec
fn fvec_to_vec(data : FixedArray[Byte], n : Int) -> Vec {
  let arr = bytes_to_float_arr(data, n)
  vec_from_array(arr)
}

///|
/// SVD decomposition: A = U @ diag(S) @ Vt
/// Returns (U, S, Vt) or None if failed
pub fn fmat_svd(a : LapackMat) -> (LapackMat, Vec, LapackMat)? {
  let m = a.rows
  let n = a.cols
  let k = if m < n { m } else { n }
  let a_copy = fmat_clone(a)
  let u = fmat_zeros(m, m)
  let s = fvec_zeros(k)
  let vt = fmat_zeros(n, n)
  let ret = numbt_svd(a_copy.data, u.data, s, vt.data, m, n)
  if ret != 0 {
    return None
  }
  Some((u, fvec_to_vec(s, k), vt))
}

///|
/// Eigenvalue decomposition for symmetric matrix
/// Returns (eigenvalues, eigenvectors) or None if failed
/// Note: input matrix must be symmetric
pub fn fmat_eig(a : LapackMat) -> (Vec, LapackMat)? {
  if a.rows != a.cols {
    return None
  }
  let n = a.rows
  let a_copy = fmat_clone(a)
  let w = fvec_zeros(n)
  let ret = numbt_eig_symmetric(a_copy.data, w, n)
  if ret != 0 {
    return None
  }
  // After ssyev, a_copy contains eigenvectors
  Some((fvec_to_vec(w, n), a_copy))
}

///|
/// Cholesky decomposition: A = L @ L^T
/// Returns lower triangular L or None if failed
/// Note: input matrix must be symmetric positive definite
pub fn fmat_cholesky(a : LapackMat) -> LapackMat? {
  if a.rows != a.cols {
    return None
  }
  let n = a.rows
  let a_copy = fmat_clone(a)
  let ret = numbt_cholesky(a_copy.data, n)
  if ret != 0 {
    return None
  }
  // Zero out upper triangle (LAPACK returns L in lower triangle)
  for i = 0; i < n; i = i + 1 {
    for j = i + 1; j < n; j = j + 1 {
      fvec_set(a_copy.data, i * n + j, 0.0)
    }
  }
  Some(a_copy)
}

///|
/// QR decomposition: A = Q @ R
/// Returns (Q, R) or None if failed
pub fn fmat_qr(a : LapackMat) -> (LapackMat, LapackMat)? {
  let m = a.rows
  let n = a.cols
  let a_copy = fmat_clone(a)
  let q = fmat_zeros(m, m)
  let r = fmat_zeros(m, n)
  let ret = numbt_qr(a_copy.data, q.data, r.data, m, n)
  if ret != 0 {
    return None
  }
  Some((q, r))
}

///|
/// Compute determinant of square matrix
/// Returns determinant or None if failed
pub fn fmat_det(a : LapackMat) -> Float? {
  if a.rows != a.cols {
    return None
  }
  let n = a.rows
  let a_copy = fmat_clone(a)
  let det_buf = fvec_zeros(1)
  let ret = numbt_det(a_copy.data, det_buf, n)
  if ret != 0 {
    return None
  }
  Some(fvec_get(det_buf, 0))
}

///|
/// Least squares solution: minimize ||A @ x - b||
/// Returns solution x or None if failed
pub fn fmat_lstsq(a : LapackMat, b : Vec) -> Vec? {
  let m = a.rows
  let n = a.cols
  if b.len != m {
    return None
  }
  let a_copy = fmat_clone(a)
  let b_data = fvec_zeros(m)
  for i = 0; i < m; i = i + 1 {
    fvec_set(b_data, i, vec_at(b, i))
  }
  let ret = numbt_lstsq(a_copy.data, b_data, m, n)
  if ret != 0 {
    return None
  }
  // Solution is in first n elements of b_data
  Some(fvec_to_vec(b_data, n))
}

// ============================================================================
// Core data structures
// ============================================================================

///|
pub struct Vec {
  data : Array[Float]
  offset : Int
  len : Int
}

///|
pub struct Mat {
  data : Array[Float]
  rows : Int
  cols : Int
}

///|
fn vec_check_range(data : Array[Float], offset : Int, len : Int) -> Unit {
  if offset < 0 || len < 0 || offset + len > data.length() {
    panic()
  }
}

///|
fn vec_check_same_len(left : Vec, right : Vec) -> Unit {
  if left.len != right.len {
    panic()
  }
}

///|
fn vec_check_non_empty(vec : Vec) -> Unit {
  if vec.len <= 0 {
    panic()
  }
}

///|
pub fn vec_view(data : Array[Float], offset : Int, len : Int) -> Vec {
  vec_check_range(data, offset, len)
  { data, offset, len }
}

///|
pub fn vec_from_array(data : Array[Float]) -> Vec {
  vec_view(data, 0, data.length())
}

///|
pub fn vec_new(len : Int, value : Float) -> Vec {
  if len < 0 {
    panic()
  }
  vec_from_array(Array::make(len, value))
}

///|
pub fn vec_zeros(len : Int) -> Vec {
  vec_new(len, Float::from_int(0))
}

///|
pub fn vec_ones(len : Int) -> Vec {
  vec_new(len, Float::from_int(1))
}

///|
/// Create vec from initializer function: out[i] = f(i)
pub fn vec_from_fn(len : Int, f : (Int) -> Float) -> Vec {
  if len < 0 {
    panic()
  }
  vec_from_array(Array::makei(len, f))
}

///|
/// Create vec like numpy.arange: [start, stop) with step
pub fn vec_arange(start : Float, stop : Float, step : Float) -> Vec {
  if step == Float::from_int(0) {
    panic()
  }
  let len = ((stop - start) / step).to_int()
  if len <= 0 {
    return vec_from_array([])
  }
  vec_from_fn(len, fn(i) { start + Float::from_int(i) * step })
}

///|
/// Create vec like numpy.linspace: num evenly spaced values in [start, stop]
pub fn vec_linspace(start : Float, stop : Float, num : Int) -> Vec {
  if num <= 0 {
    panic()
  }
  if num == 1 {
    return vec_from_array([start])
  }
  let step = (stop - start) / Float::from_int(num - 1)
  vec_from_fn(num, fn(i) { start + Float::from_int(i) * step })
}

///|
pub fn vec_len(vec : Vec) -> Int {
  vec.len
}

///|
pub fn vec_at(vec : Vec, index : Int) -> Float {
  vec.data[vec.offset + index]
}

///|
pub fn vec_set(vec : Vec, index : Int, value : Float) -> Unit {
  vec.data[vec.offset + index] = value
}

///|
/// Fill all elements with a constant value (in-place modification).
///
/// Example: `vec_fill_inplace(v, 0.0)` sets all elements to zero.
pub fn vec_fill_inplace(vec : Vec, value : Float) -> Unit {
  for i = 0; i < vec.len; i = i + 1 {
    vec_set(vec, i, value)
  }
}

///|
/// @deprecated Use `vec_fill_inplace` instead.
pub fn vec_fill(vec : Vec, value : Float) -> Unit {
  vec_fill_inplace(vec, value)
}

///|
pub fn vec_to_array(vec : Vec) -> Array[Float] {
  Array::makei(vec.len, fn(i) { vec_at(vec, i) })
}

///|
/// Element-wise addition: output[i] = left[i] + right[i]
pub fn vec_add_into(left : Vec, right : Vec, output~ : Vec) -> Unit {
  vec_check_same_len(output, left)
  vec_check_same_len(left, right)
  for i = 0; i < left.len; i = i + 1 {
    vec_set(output, i, vec_at(left, i) + vec_at(right, i))
  }
}

///|
/// Element-wise subtraction: output[i] = left[i] - right[i]
pub fn vec_sub_into(left : Vec, right : Vec, output~ : Vec) -> Unit {
  vec_check_same_len(output, left)
  vec_check_same_len(left, right)
  for i = 0; i < left.len; i = i + 1 {
    vec_set(output, i, vec_at(left, i) - vec_at(right, i))
  }
}

///|
/// Element-wise multiplication: output[i] = left[i] * right[i]
pub fn vec_mul_into(left : Vec, right : Vec, output~ : Vec) -> Unit {
  vec_check_same_len(output, left)
  vec_check_same_len(left, right)
  for i = 0; i < left.len; i = i + 1 {
    vec_set(output, i, vec_at(left, i) * vec_at(right, i))
  }
}

///|
/// Element-wise division: output[i] = left[i] / right[i]
pub fn vec_div_into(left : Vec, right : Vec, output~ : Vec) -> Unit {
  vec_check_same_len(output, left)
  vec_check_same_len(left, right)
  for i = 0; i < left.len; i = i + 1 {
    vec_set(output, i, vec_at(left, i) / vec_at(right, i))
  }
}

///|
pub fn vec_add(left : Vec, right : Vec) -> Vec {
  vec_check_same_len(left, right)
  let output = vec_new(left.len, Float::from_int(0))
  vec_add_into(left, right, output~)
  output
}

///|
pub fn vec_sub(left : Vec, right : Vec) -> Vec {
  vec_check_same_len(left, right)
  let output = vec_new(left.len, Float::from_int(0))
  vec_sub_into(left, right, output~)
  output
}

///|
pub fn vec_mul(left : Vec, right : Vec) -> Vec {
  vec_check_same_len(left, right)
  let output = vec_new(left.len, Float::from_int(0))
  vec_mul_into(left, right, output~)
  output
}

///|
pub fn vec_div(left : Vec, right : Vec) -> Vec {
  vec_check_same_len(left, right)
  let output = vec_new(left.len, Float::from_int(0))
  vec_div_into(left, right, output~)
  output
}

///|
pub fn vec_sum(vec : Vec) -> Float {
  let mut sum = Float::from_int(0)
  for i = 0; i < vec.len; i = i + 1 {
    sum = sum + vec_at(vec, i)
  }
  sum
}

///|
pub fn vec_max(vec : Vec) -> Float {
  vec_check_non_empty(vec)
  let mut max = vec_at(vec, 0)
  for i = 1; i < vec.len; i = i + 1 {
    let v = vec_at(vec, i)
    if v > max {
      max = v
    }
  }
  max
}

///|
pub fn vec_argmax(vec : Vec) -> Int {
  vec_check_non_empty(vec)
  let mut max = vec_at(vec, 0)
  let mut index = 0
  for i = 1; i < vec.len; i = i + 1 {
    let v = vec_at(vec, i)
    if v > max {
      max = v
      index = i
    }
  }
  index
}

///|
pub fn vec_min(vec : Vec) -> Float {
  vec_check_non_empty(vec)
  let mut min = vec_at(vec, 0)
  for i = 1; i < vec.len; i = i + 1 {
    let v = vec_at(vec, i)
    if v < min {
      min = v
    }
  }
  min
}

///|
pub fn vec_argmin(vec : Vec) -> Int {
  vec_check_non_empty(vec)
  let mut min = vec_at(vec, 0)
  let mut index = 0
  for i = 1; i < vec.len; i = i + 1 {
    let v = vec_at(vec, i)
    if v < min {
      min = v
      index = i
    }
  }
  index
}

///|
pub fn vec_mean(vec : Vec) -> Float {
  vec_check_non_empty(vec)
  vec_sum(vec) / Float::from_int(vec.len)
}

///|
/// Variance: E[(X - mean)^2]
pub fn vec_var(vec : Vec) -> Float {
  vec_check_non_empty(vec)
  let mean = vec_mean(vec)
  let mut sum_sq = Float::from_int(0)
  for i = 0; i < vec.len; i = i + 1 {
    let diff = vec_at(vec, i) - mean
    sum_sq = sum_sq + diff * diff
  }
  sum_sq / Float::from_int(vec.len)
}

///|
/// Standard deviation: sqrt(variance)
pub fn vec_std(vec : Vec) -> Float {
  vec_var(vec).sqrt()
}

///|
/// Product of all elements
pub fn vec_prod(vec : Vec) -> Float {
  vec_check_non_empty(vec)
  let mut prod = Float::from_int(1)
  for i = 0; i < vec.len; i = i + 1 {
    prod = prod * vec_at(vec, i)
  }
  prod
}

///|
/// Multiply all elements by a scalar (in-place modification).
///
/// Example: `vec_scale_inplace(v, 2.0)` doubles all elements.
pub fn vec_scale_inplace(vec : Vec, scale : Float) -> Unit {
  for i = 0; i < vec.len; i = i + 1 {
    vec_set(vec, i, vec_at(vec, i) * scale)
  }
}

///|
/// Element-wise exponential: output[i] = exp(input[i])
///
/// Note: For BLAS-accelerated version, use `vec_exp` which uses vvexpf.
pub fn vec_exp_into(output~ : Vec, input : Vec) -> Unit {
  vec_check_same_len(output, input)
  for i = 0; i < input.len; i = i + 1 {
    vec_set(output, i, @math.expf(vec_at(input, i)))
  }
}

///|
/// Uses vvexpf from Accelerate framework for optimal performance
pub fn vec_exp(input : Vec) -> Vec {
  let src_arr = vec_to_array(input)
  let src_bytes = float_arr_to_bytes(src_arr)
  let dst_bytes : FixedArray[Byte] = FixedArray::make(input.len * 4, b'\x00')
  numbt_vexp(src_bytes, dst_bytes, input.len)
  let result = bytes_to_float_arr(dst_bytes, input.len)
  vec_from_array(result)
}

///|
/// Element-wise natural logarithm: output[i] = ln(input[i])
pub fn vec_log_into(output~ : Vec, input : Vec) -> Unit {
  vec_check_same_len(output, input)
  for i = 0; i < input.len; i = i + 1 {
    vec_set(output, i, @math.lnf(vec_at(input, i)))
  }
}

///|
pub fn vec_log(input : Vec) -> Vec {
  let out = vec_zeros(input.len)
  vec_log_into(output=out, input)
  out
}

///|
/// Element-wise square root: output[i] = sqrt(input[i])
pub fn vec_sqrt_into(input : Vec, output~ : Vec) -> Unit {
  vec_check_same_len(output, input)
  for i = 0; i < input.len; i = i + 1 {
    vec_set(output, i, vec_at(input, i).sqrt())
  }
}

///|
pub fn vec_sqrt(input : Vec) -> Vec {
  let output = vec_zeros(input.len)
  vec_sqrt_into(input, output~)
  output
}

///|
/// Element-wise absolute value: output[i] = |input[i]|
pub fn vec_abs_into(input : Vec, output~ : Vec) -> Unit {
  vec_check_same_len(output, input)
  for i = 0; i < input.len; i = i + 1 {
    vec_set(output, i, vec_at(input, i).abs())
  }
}

///|
pub fn vec_abs(input : Vec) -> Vec {
  let output = vec_zeros(input.len)
  vec_abs_into(input, output~)
  output
}

///|
/// Element-wise power: output[i] = input[i]^exp
pub fn vec_pow_into(input : Vec, exp : Float, output~ : Vec) -> Unit {
  vec_check_same_len(output, input)
  for i = 0; i < input.len; i = i + 1 {
    vec_set(output, i, @math.powf(vec_at(input, i), exp))
  }
}

///|
pub fn vec_pow(input : Vec, exp : Float) -> Vec {
  let output = vec_zeros(input.len)
  vec_pow_into(input, exp, output~)
  output
}

///|
/// Clip values to [min, max] range: output[i] = clamp(input[i], min, max)
pub fn vec_clip_into(
  input : Vec,
  min : Float,
  max : Float,
  output~ : Vec,
) -> Unit {
  vec_check_same_len(output, input)
  for i = 0; i < input.len; i = i + 1 {
    let v = vec_at(input, i)
    let clipped = if v < min { min } else if v > max { max } else { v }
    vec_set(output, i, clipped)
  }
}

///|
pub fn vec_clip(input : Vec, min : Float, max : Float) -> Vec {
  let output = vec_zeros(input.len)
  vec_clip_into(input, min, max, output~)
  output
}

///|
/// Element-wise sine: output[i] = sin(input[i])
pub fn vec_sin_into(input : Vec, output~ : Vec) -> Unit {
  vec_check_same_len(output, input)
  for i = 0; i < input.len; i = i + 1 {
    vec_set(output, i, @math.sinf(vec_at(input, i)))
  }
}

///|
pub fn vec_sin(input : Vec) -> Vec {
  let output = vec_zeros(input.len)
  vec_sin_into(input, output~)
  output
}

///|
/// Element-wise cosine: output[i] = cos(input[i])
pub fn vec_cos_into(input : Vec, output~ : Vec) -> Unit {
  vec_check_same_len(output, input)
  for i = 0; i < input.len; i = i + 1 {
    vec_set(output, i, @math.cosf(vec_at(input, i)))
  }
}

///|
pub fn vec_cos(input : Vec) -> Vec {
  let output = vec_zeros(input.len)
  vec_cos_into(input, output~)
  output
}

///|
/// Element-wise hyperbolic tangent: output[i] = tanh(input[i])
pub fn vec_tanh_into(input : Vec, output~ : Vec) -> Unit {
  vec_check_same_len(output, input)
  for i = 0; i < input.len; i = i + 1 {
    vec_set(output, i, @math.tanhf(vec_at(input, i)))
  }
}

///|
pub fn vec_tanh(input : Vec) -> Vec {
  let output = vec_zeros(input.len)
  vec_tanh_into(input, output~)
  output
}

///|
pub fn relu(value : Float) -> Float {
  let zero = Float::from_int(0)
  if value < zero {
    zero
  } else {
    value
  }
}

///|
pub fn relu_grad(value : Float) -> Float {
  let zero = Float::from_int(0)
  if value > zero {
    Float::from_int(1)
  } else {
    zero
  }
}

///|
/// Compute softmax probabilities from logits (writes to output buffer).
///
/// Parameters:
/// - `input`: Raw logits (unnormalized scores)
/// - `output`: Output buffer for normalized probabilities (same length as input)
///
/// The softmax is computed with numerical stability (subtracting max).
/// Result: `output[i] = exp(input[i] - max) / sum(exp(input - max))`
pub fn softmax_into(input~ : Vec, output~ : Vec) -> Unit {
  vec_check_same_len(input, output)
  let max = vec_max(input)
  let mut sum = Float::from_int(0)
  for i = 0; i < input.len; i = i + 1 {
    let v = @math.expf(vec_at(input, i) - max)
    vec_set(output, i, v)
    sum = sum + v
  }
  for i = 0; i < input.len; i = i + 1 {
    vec_set(output, i, vec_at(output, i) / sum)
  }
}

///|
pub fn cross_entropy_loss(probs : Vec, label : Int) -> Float {
  let eps = Float::from_int(1) / Float::from_int(1000000)
  let p = vec_at(probs, label) + eps
  -@math.lnf(p)
}

///|
fn mat_check_shape(mat : Mat, data : Array[Float]) -> Unit {
  if mat.rows <= 0 || mat.cols <= 0 {
    panic()
  }
  if mat.rows * mat.cols != data.length() {
    panic()
  }
}

///|
fn mat_check_vec_dims(mat : Mat, vec : Vec, out : Vec) -> Unit {
  if vec.len != mat.rows || out.len != mat.cols {
    panic()
  }
}

///|
fn mat_check_same_shape(a : Mat, b : Mat) -> Unit {
  if a.rows != b.rows || a.cols != b.cols {
    panic()
  }
}

///|
pub fn mat_view(data : Array[Float], rows : Int, cols : Int) -> Mat {
  let mat : Mat = { data, rows, cols }
  mat_check_shape(mat, data)
  mat
}

///|
pub fn mat_new(rows : Int, cols : Int, value : Float) -> Mat {
  if rows <= 0 || cols <= 0 {
    panic()
  }
  mat_view(Array::make(rows * cols, value), rows, cols)
}

///|
pub fn mat_zeros(rows : Int, cols : Int) -> Mat {
  mat_new(rows, cols, Float::from_int(0))
}

///|
pub fn mat_ones(rows : Int, cols : Int) -> Mat {
  mat_new(rows, cols, Float::from_int(1))
}

///|
pub fn mat_eye(n : Int) -> Mat {
  if n <= 0 {
    panic()
  }
  let data = Array::make(n * n, Float::from_int(0))
  for i = 0; i < n; i = i + 1 {
    data[i * n + i] = Float::from_int(1)
  }
  mat_view(data, n, n)
}

///|
pub fn mat_rows(mat : Mat) -> Int {
  mat.rows
}

///|
pub fn mat_cols(mat : Mat) -> Int {
  mat.cols
}

///|
pub fn mat_at(mat : Mat, row : Int, col : Int) -> Float {
  mat.data[row * mat.cols + col]
}

///|
pub fn mat_set(mat : Mat, row : Int, col : Int, value : Float) -> Unit {
  mat.data[row * mat.cols + col] = value
}

///|
/// Fill all elements with a constant value (in-place modification).
pub fn mat_fill_inplace(mat : Mat, value : Float) -> Unit {
  for i = 0; i < mat.data.length(); i = i + 1 {
    mat.data[i] = value
  }
}

///|
/// @deprecated Use `mat_fill_inplace` instead.
pub fn mat_fill(mat : Mat, value : Float) -> Unit {
  mat_fill_inplace(mat, value)
}

///|
pub fn mat_to_array(mat : Mat) -> Array[Float] {
  Array::makei(mat.data.length(), fn(i) { mat.data[i] })
}

///|
pub fn mat_clone(mat : Mat) -> Mat {
  mat_view(mat_to_array(mat), mat.rows, mat.cols)
}

///|
/// Get row as Vec (view into original data)
pub fn mat_row(mat : Mat, row : Int) -> Vec {
  vec_view(mat.data, row * mat.cols, mat.cols)
}

///|
/// Transpose matrix (creates new matrix)
pub fn mat_transpose(mat : Mat) -> Mat {
  let out = mat_zeros(mat.cols, mat.rows)
  for i = 0; i < mat.rows; i = i + 1 {
    for j = 0; j < mat.cols; j = j + 1 {
      mat_set(out, j, i, mat_at(mat, i, j))
    }
  }
  out
}

///|
/// Element-wise matrix addition: output[i,j] = a[i,j] + b[i,j]
pub fn mat_add_into(a : Mat, b : Mat, output~ : Mat) -> Unit {
  mat_check_same_shape(output, a)
  mat_check_same_shape(a, b)
  for i = 0; i < a.data.length(); i = i + 1 {
    output.data[i] = a.data[i] + b.data[i]
  }
}

///|
/// Element-wise matrix subtraction: output[i,j] = a[i,j] - b[i,j]
pub fn mat_sub_into(a : Mat, b : Mat, output~ : Mat) -> Unit {
  mat_check_same_shape(output, a)
  mat_check_same_shape(a, b)
  for i = 0; i < a.data.length(); i = i + 1 {
    output.data[i] = a.data[i] - b.data[i]
  }
}

///|
/// Element-wise matrix multiplication: output[i,j] = a[i,j] * b[i,j]
pub fn mat_mul_into(a : Mat, b : Mat, output~ : Mat) -> Unit {
  mat_check_same_shape(output, a)
  mat_check_same_shape(a, b)
  for i = 0; i < a.data.length(); i = i + 1 {
    output.data[i] = a.data[i] * b.data[i]
  }
}

///|
/// Element-wise matrix division: output[i,j] = a[i,j] / b[i,j]
pub fn mat_div_into(a : Mat, b : Mat, output~ : Mat) -> Unit {
  mat_check_same_shape(output, a)
  mat_check_same_shape(a, b)
  for i = 0; i < a.data.length(); i = i + 1 {
    output.data[i] = a.data[i] / b.data[i]
  }
}

///|
pub fn mat_add(a : Mat, b : Mat) -> Mat {
  mat_check_same_shape(a, b)
  let output = mat_zeros(a.rows, a.cols)
  mat_add_into(a, b, output~)
  output
}

///|
pub fn mat_sub(a : Mat, b : Mat) -> Mat {
  mat_check_same_shape(a, b)
  let output = mat_zeros(a.rows, a.cols)
  mat_sub_into(a, b, output~)
  output
}

///|
pub fn mat_mul(a : Mat, b : Mat) -> Mat {
  mat_check_same_shape(a, b)
  let output = mat_zeros(a.rows, a.cols)
  mat_mul_into(a, b, output~)
  output
}

///|
pub fn mat_div(a : Mat, b : Mat) -> Mat {
  mat_check_same_shape(a, b)
  let output = mat_zeros(a.rows, a.cols)
  mat_div_into(a, b, output~)
  output
}

///|
pub impl Add for Mat with add(self : Mat, other : Mat) -> Mat {
  mat_add(self, other)
}

///|
pub impl Sub for Mat with sub(self : Mat, other : Mat) -> Mat {
  mat_sub(self, other)
}

///|
pub impl Mul for Mat with mul(self : Mat, other : Mat) -> Mat {
  mat_mul(self, other)
}

///|
pub impl Div for Mat with div(self : Mat, other : Mat) -> Mat {
  mat_div(self, other)
}

///|
pub impl Neg for Mat with neg(self : Mat) -> Mat {
  mat_neg(self)
}

///|
pub fn mat_neg(mat : Mat) -> Mat {
  let out = mat_zeros(mat.rows, mat.cols)
  for i = 0; i < mat.data.length(); i = i + 1 {
    out.data[i] = -mat.data[i]
  }
  out
}

///|
pub fn mat_add_scalar(mat : Mat, scalar : Float) -> Mat {
  let out = mat_zeros(mat.rows, mat.cols)
  for i = 0; i < mat.data.length(); i = i + 1 {
    out.data[i] = mat.data[i] + scalar
  }
  out
}

///|
pub fn mat_sub_scalar(mat : Mat, scalar : Float) -> Mat {
  let out = mat_zeros(mat.rows, mat.cols)
  for i = 0; i < mat.data.length(); i = i + 1 {
    out.data[i] = mat.data[i] - scalar
  }
  out
}

///|
pub fn mat_mul_scalar(mat : Mat, scalar : Float) -> Mat {
  let out = mat_zeros(mat.rows, mat.cols)
  for i = 0; i < mat.data.length(); i = i + 1 {
    out.data[i] = mat.data[i] * scalar
  }
  out
}

///|
pub fn mat_div_scalar(mat : Mat, scalar : Float) -> Mat {
  let out = mat_zeros(mat.rows, mat.cols)
  for i = 0; i < mat.data.length(); i = i + 1 {
    out.data[i] = mat.data[i] / scalar
  }
  out
}

///|
pub fn mat_scale_inplace(mat : Mat, scalar : Float) -> Unit {
  for i = 0; i < mat.data.length(); i = i + 1 {
    mat.data[i] = mat.data[i] * scalar
  }
}

///|
/// Matrix multiplication using BLAS: C = A @ B
///
/// Equivalent to NumPy's `@` operator or `np.matmul`.
/// Requires: a.cols == b.rows
pub fn mat_matmul(a : Mat, b : Mat) -> Mat {
  if a.cols != b.rows {
    panic()
  }
  let out = mat_zeros(a.rows, b.cols)
  @blas.sgemm(a.data, b.data, out.data, a.rows, b.cols, a.cols)
  out
}

///|
/// Matrix multiplication method: self @ other
///
/// Example: `let c = a.matmul(b)` is equivalent to `mat_matmul(a, b)`
pub fn Mat::matmul(self : Mat, other : Mat) -> Mat {
  mat_matmul(self, other)
}

///|
/// Sum of all elements
pub fn mat_sum(mat : Mat) -> Float {
  let mut sum = Float::from_int(0)
  for i = 0; i < mat.data.length(); i = i + 1 {
    sum = sum + mat.data[i]
  }
  sum
}

///|
/// Mean of all elements
pub fn mat_mean(mat : Mat) -> Float {
  mat_sum(mat) / Float::from_int(mat.data.length())
}

///|
/// Max of all elements
pub fn mat_max(mat : Mat) -> Float {
  if mat.data.length() == 0 {
    panic()
  }
  let mut max = mat.data[0]
  for i = 1; i < mat.data.length(); i = i + 1 {
    if mat.data[i] > max {
      max = mat.data[i]
    }
  }
  max
}

///|
/// Min of all elements
pub fn mat_min(mat : Mat) -> Float {
  if mat.data.length() == 0 {
    panic()
  }
  let mut min = mat.data[0]
  for i = 1; i < mat.data.length(); i = i + 1 {
    if mat.data[i] < min {
      min = mat.data[i]
    }
  }
  min
}

///|
/// Sum along axis 0 (columns): result[j] = sum_i(mat[i,j])
pub fn mat_sum_axis0(mat : Mat) -> Vec {
  let out = vec_zeros(mat.cols)
  for i = 0; i < mat.rows; i = i + 1 {
    for j = 0; j < mat.cols; j = j + 1 {
      vec_set(out, j, vec_at(out, j) + mat_at(mat, i, j))
    }
  }
  out
}

///|
/// Sum along axis 1 (rows): result[i] = sum_j(mat[i,j])
pub fn mat_sum_axis1(mat : Mat) -> Vec {
  let out = vec_zeros(mat.rows)
  for i = 0; i < mat.rows; i = i + 1 {
    let mut sum = Float::from_int(0)
    for j = 0; j < mat.cols; j = j + 1 {
      sum = sum + mat_at(mat, i, j)
    }
    vec_set(out, i, sum)
  }
  out
}

///|
/// Mean along axis 0
pub fn mat_mean_axis0(mat : Mat) -> Vec {
  let sums = mat_sum_axis0(mat)
  vec_div_scalar(sums, Float::from_int(mat.rows))
}

///|
/// Mean along axis 1
pub fn mat_mean_axis1(mat : Mat) -> Vec {
  let sums = mat_sum_axis1(mat)
  vec_div_scalar(sums, Float::from_int(mat.cols))
}

// ============================================================================
// Comparison functions
// ============================================================================

///|
/// Check if all elements are equal within tolerance
pub fn vec_allclose(a : Vec, b : Vec, atol : Float) -> Bool {
  if a.len != b.len {
    return false
  }
  for i = 0; i < a.len; i = i + 1 {
    if (vec_at(a, i) - vec_at(b, i)).abs() > atol {
      return false
    }
  }
  true
}

///|
/// Check if all elements are equal within tolerance
pub fn mat_allclose(a : Mat, b : Mat, atol : Float) -> Bool {
  if a.rows != b.rows || a.cols != b.cols {
    return false
  }
  for i = 0; i < a.data.length(); i = i + 1 {
    if (a.data[i] - b.data[i]).abs() > atol {
      return false
    }
  }
  true
}

///|
/// Check if any element is NaN
pub fn vec_has_nan(v : Vec) -> Bool {
  for i = 0; i < v.len; i = i + 1 {
    if vec_at(v, i).is_nan() {
      return true
    }
  }
  false
}

///|
/// Check if any element is infinite
pub fn vec_has_inf(v : Vec) -> Bool {
  for i = 0; i < v.len; i = i + 1 {
    if vec_at(v, i).is_inf() {
      return true
    }
  }
  false
}

// ============================================================================
// Boolean operations
// ============================================================================

///|
/// Returns true if any element is non-zero (truthy)
pub fn vec_any(v : Vec) -> Bool {
  for i = 0; i < v.len; i = i + 1 {
    if vec_at(v, i) != 0.0 {
      return true
    }
  }
  false
}

///|
/// Returns true if all elements are non-zero (truthy)
pub fn vec_all(v : Vec) -> Bool {
  for i = 0; i < v.len; i = i + 1 {
    if vec_at(v, i) == 0.0 {
      return false
    }
  }
  true
}

// ============================================================================
// Element-wise operations
// ============================================================================

///|
/// Element-wise sign: -1 for negative, 0 for zero, 1 for positive
pub fn vec_sign_into(input : Vec, output~ : Vec) -> Unit {
  vec_check_same_len(output, input)
  let one : Float = 1.0
  let neg_one : Float = -1.0
  let zero : Float = 0.0
  for i = 0; i < input.len; i = i + 1 {
    let v = vec_at(input, i)
    let s = if v > zero { one } else if v < zero { neg_one } else { zero }
    vec_set(output, i, s)
  }
}

///|
pub fn vec_sign(input : Vec) -> Vec {
  let output = vec_zeros(input.len)
  vec_sign_into(input, output~)
  output
}

///|
/// Element-wise maximum of two vectors: output[i] = max(a[i], b[i])
pub fn vec_maximum_into(a : Vec, b : Vec, output~ : Vec) -> Unit {
  vec_check_same_len(output, a)
  vec_check_same_len(a, b)
  for i = 0; i < a.len; i = i + 1 {
    let va = vec_at(a, i)
    let vb = vec_at(b, i)
    vec_set(output, i, if va > vb { va } else { vb })
  }
}

///|
pub fn vec_maximum(a : Vec, b : Vec) -> Vec {
  let output = vec_zeros(a.len)
  vec_maximum_into(a, b, output~)
  output
}

///|
/// Element-wise minimum of two vectors: output[i] = min(a[i], b[i])
pub fn vec_minimum_into(a : Vec, b : Vec, output~ : Vec) -> Unit {
  vec_check_same_len(output, a)
  vec_check_same_len(a, b)
  for i = 0; i < a.len; i = i + 1 {
    let va = vec_at(a, i)
    let vb = vec_at(b, i)
    vec_set(output, i, if va < vb { va } else { vb })
  }
}

///|
pub fn vec_minimum(a : Vec, b : Vec) -> Vec {
  let output = vec_zeros(a.len)
  vec_minimum_into(a, b, output~)
  output
}

// ============================================================================
// Slice and reshape
// ============================================================================

///|
/// Create a slice of vec [start, end)
pub fn vec_slice(v : Vec, start : Int, end : Int) -> Vec {
  if start < 0 || end > v.len || start >= end {
    panic()
  }
  vec_view(v.data, v.offset + start, end - start)
}

///|
/// Flatten mat to vec (row-major order)
pub fn mat_flatten(mat : Mat) -> Vec {
  vec_from_array(mat_to_array(mat))
}

///|
/// Reshape vec to mat (must have matching element count)
pub fn vec_reshape(v : Vec, rows : Int, cols : Int) -> Mat {
  if rows * cols != v.len {
    panic()
  }
  mat_view(vec_to_array(v), rows, cols)
}

///|
/// Matrix-vector multiply: output = weight^T @ input
///
/// Parameters:
/// - `weight`: Matrix of shape (in_dim, out_dim)
/// - `input`: Input vector of length in_dim
/// - `output`: Output buffer of length out_dim
///
/// Computes: `output[j] = sum_i(input[i] * weight[i, j])`
pub fn matmul_vec_into(weight : Mat, input : Vec, output~ : Vec) -> Unit {
  mat_check_vec_dims(weight, input, output)
  for j = 0; j < weight.cols; j = j + 1 {
    let mut acc = Float::from_int(0)
    for i = 0; i < weight.rows; i = i + 1 {
      acc = acc + vec_at(input, i) * weight.data[i * weight.cols + j]
    }
    vec_set(output, j, acc)
  }
}

///|
/// Matrix-vector multiply with bias: output = weight^T @ input + bias
///
/// Parameters:
/// - `weight`: Matrix of shape (in_dim, out_dim)
/// - `input`: Input vector of length in_dim
/// - `bias`: Bias vector of length out_dim
/// - `output`: Output buffer of length out_dim
///
/// Computes: `output[j] = bias[j] + sum_i(input[i] * weight[i, j])`
pub fn matmul_vec_bias_into(
  weight : Mat,
  input : Vec,
  bias : Vec,
  output~ : Vec,
) -> Unit {
  mat_check_vec_dims(weight, input, output)
  vec_check_same_len(bias, output)
  for j = 0; j < weight.cols; j = j + 1 {
    let mut acc = vec_at(bias, j)
    for i = 0; i < weight.rows; i = i + 1 {
      acc = acc + vec_at(input, i) * weight.data[i * weight.cols + j]
    }
    vec_set(output, j, acc)
  }
}

///|
/// Matrix-vector multiply with bias and ReLU: output = relu(weight^T @ input + bias)
///
/// Parameters:
/// - `weight`: Matrix of shape (in_dim, out_dim)
/// - `input`: Input vector of length in_dim
/// - `bias`: Bias vector of length out_dim
/// - `output`: Output buffer of length out_dim
///
/// Computes: `output[j] = max(0, bias[j] + sum_i(input[i] * weight[i, j]))`
pub fn matmul_vec_bias_relu_into(
  weight : Mat,
  input : Vec,
  bias : Vec,
  output~ : Vec,
) -> Unit {
  mat_check_vec_dims(weight, input, output)
  vec_check_same_len(bias, output)
  for j = 0; j < weight.cols; j = j + 1 {
    let mut acc = vec_at(bias, j)
    for i = 0; i < weight.rows; i = i + 1 {
      acc = acc + vec_at(input, i) * weight.data[i * weight.cols + j]
    }
    vec_set(output, j, relu(acc))
  }
}

///|
pub impl Add for Vec with add(self : Vec, other : Vec) -> Vec {
  vec_add(self, other)
}

///|
pub impl Sub for Vec with sub(self : Vec, other : Vec) -> Vec {
  vec_sub(self, other)
}

///|
pub impl Mul for Vec with mul(self : Vec, other : Vec) -> Vec {
  vec_mul(self, other)
}

///|
pub impl Div for Vec with div(self : Vec, other : Vec) -> Vec {
  vec_div(self, other)
}

///|
pub impl Neg for Vec with neg(self : Vec) -> Vec {
  vec_neg(self)
}

// ============================================================================
// Scalar operations
// ============================================================================

///|
/// Negate all elements: out = -input
pub fn vec_neg(input : Vec) -> Vec {
  let out = vec_new(input.len, Float::from_int(0))
  for i = 0; i < input.len; i = i + 1 {
    vec_set(out, i, -vec_at(input, i))
  }
  out
}

///|
/// Add scalar to all elements: out = vec + scalar
pub fn vec_add_scalar(vec : Vec, scalar : Float) -> Vec {
  let out = vec_new(vec.len, Float::from_int(0))
  for i = 0; i < vec.len; i = i + 1 {
    vec_set(out, i, vec_at(vec, i) + scalar)
  }
  out
}

///|
/// Subtract scalar from all elements: out = vec - scalar
pub fn vec_sub_scalar(vec : Vec, scalar : Float) -> Vec {
  let out = vec_new(vec.len, Float::from_int(0))
  for i = 0; i < vec.len; i = i + 1 {
    vec_set(out, i, vec_at(vec, i) - scalar)
  }
  out
}

///|
/// Multiply all elements by scalar: out = vec * scalar
pub fn vec_mul_scalar(vec : Vec, scalar : Float) -> Vec {
  let out = vec_new(vec.len, Float::from_int(0))
  for i = 0; i < vec.len; i = i + 1 {
    vec_set(out, i, vec_at(vec, i) * scalar)
  }
  out
}

///|
/// Divide all elements by scalar: out = vec / scalar
pub fn vec_div_scalar(vec : Vec, scalar : Float) -> Vec {
  let out = vec_new(vec.len, Float::from_int(0))
  for i = 0; i < vec.len; i = i + 1 {
    vec_set(out, i, vec_at(vec, i) / scalar)
  }
  out
}

///|
/// In-place negate: vec = -vec
pub fn vec_neg_inplace(vec : Vec) -> Unit {
  for i = 0; i < vec.len; i = i + 1 {
    vec_set(vec, i, -vec_at(vec, i))
  }
}

///|
/// In-place add scalar: vec = vec + scalar
pub fn vec_add_scalar_inplace(vec : Vec, scalar : Float) -> Unit {
  for i = 0; i < vec.len; i = i + 1 {
    vec_set(vec, i, vec_at(vec, i) + scalar)
  }
}

///|
/// In-place sub scalar: vec = vec - scalar
pub fn vec_sub_scalar_inplace(vec : Vec, scalar : Float) -> Unit {
  for i = 0; i < vec.len; i = i + 1 {
    vec_set(vec, i, vec_at(vec, i) - scalar)
  }
}

///|
/// In-place mul scalar: vec = vec * scalar
pub fn vec_mul_scalar_inplace(vec : Vec, scalar : Float) -> Unit {
  for i = 0; i < vec.len; i = i + 1 {
    vec_set(vec, i, vec_at(vec, i) * scalar)
  }
}

///|
/// In-place div scalar: vec = vec / scalar
pub fn vec_div_scalar_inplace(vec : Vec, scalar : Float) -> Unit {
  for i = 0; i < vec.len; i = i + 1 {
    vec_set(vec, i, vec_at(vec, i) / scalar)
  }
}

// ============================================================================
// BLAS Level 1 (vector-vector)
// ============================================================================

///|
/// Dot product: x · y
pub fn vec_dot(x : Vec, y : Vec) -> Float {
  vec_check_same_len(x, y)
  @blas.sdot(x.data, y.data)
}

///|
/// L2 norm: ||x||_2
pub fn vec_nrm2(x : Vec) -> Float {
  @blas.snrm2(x.data)
}

///|
/// AXPY: y = α*x + y
pub fn vec_axpy(alpha : Float, x : Vec, y : Vec) -> Unit {
  vec_check_same_len(x, y)
  @blas.saxpy(alpha, x.data, y.data)
}

// ============================================================================
// BLAS Level 2 (matrix-vector)
// ============================================================================

///|
/// Matrix-vector multiply using BLAS sgemv: output = weight^T @ input
///
/// Uses Accelerate framework's BLAS for optimal performance.
pub fn matmul_vec_blas_into(weight : Mat, input : Vec, output~ : Vec) -> Unit {
  mat_check_vec_dims(weight, input, output)
  @blas.sgemv_trans(
    weight.data,
    input.data,
    output.data,
    weight.rows,
    weight.cols,
  )
}

///|
/// Matrix-vector multiply + bias using BLAS: output = weight^T @ input + bias
///
/// Uses Accelerate framework's BLAS for optimal performance.
pub fn matmul_vec_bias_blas_into(
  weight : Mat,
  input : Vec,
  bias : Vec,
  output~ : Vec,
) -> Unit {
  matmul_vec_blas_into(weight, input, output~)
  for i = 0; i < output.len; i = i + 1 {
    vec_set(output, i, vec_at(output, i) + vec_at(bias, i))
  }
}

///|
/// Matrix-vector multiply + bias + ReLU using BLAS
///
/// Computes: `output = relu(weight^T @ input + bias)`
/// Uses Accelerate framework's BLAS for optimal performance.
pub fn matmul_vec_bias_relu_blas_into(
  weight : Mat,
  input : Vec,
  bias : Vec,
  output~ : Vec,
) -> Unit {
  matmul_vec_blas_into(weight, input, output~)
  for i = 0; i < output.len; i = i + 1 {
    vec_set(output, i, relu(vec_at(output, i) + vec_at(bias, i)))
  }
}

// ============================================================================
// BLAS Level 3 (matrix-matrix) - Batch operations
// ============================================================================

///|
/// Batch matrix multiply using BLAS sgemm: Out = Input @ Weight
/// Input: batch x in_dim, Weight: in_dim x out_dim, Out: batch x out_dim
pub fn batch_matmul(
  input : Array[Float],
  weight : Array[Float],
  out : Array[Float],
  batch : Int,
  in_dim : Int,
  out_dim : Int,
) -> Unit {
  @blas.sgemm(input, weight, out, batch, out_dim, in_dim)
}

///|
/// Batch matrix multiply + bias
pub fn batch_matmul_bias(
  input : Array[Float],
  weight : Array[Float],
  bias : Array[Float],
  out : Array[Float],
  batch : Int,
  in_dim : Int,
  out_dim : Int,
) -> Unit {
  batch_matmul(input, weight, out, batch, in_dim, out_dim)
  for b = 0; b < batch; b = b + 1 {
    for j = 0; j < out_dim; j = j + 1 {
      out[b * out_dim + j] = out[b * out_dim + j] + bias[j]
    }
  }
}

///|
/// Batch matrix multiply + bias + ReLU
pub fn batch_matmul_bias_relu(
  input : Array[Float],
  weight : Array[Float],
  bias : Array[Float],
  out : Array[Float],
  batch : Int,
  in_dim : Int,
  out_dim : Int,
) -> Unit {
  batch_matmul(input, weight, out, batch, in_dim, out_dim)
  for b = 0; b < batch; b = b + 1 {
    for j = 0; j < out_dim; j = j + 1 {
      let idx = b * out_dim + j
      out[idx] = relu(out[idx] + bias[j])
    }
  }
}

// ============================================================================
// Sort operations
// ============================================================================

///|
/// Sort vector in ascending order (returns new vector)
/// Uses vDSP_vsort from Accelerate framework for optimal performance
pub fn vec_sort(v : Vec) -> Vec {
  let arr = vec_to_array(v)
  let bytes = float_arr_to_bytes(arr)
  numbt_vsort_asc(bytes, arr.length())
  let result = bytes_to_float_arr(bytes, arr.length())
  vec_from_array(result)
}

///|
/// Sort vector in descending order (returns new vector)
/// Uses vDSP_vsort from Accelerate framework for optimal performance
pub fn vec_sort_desc(v : Vec) -> Vec {
  let arr = vec_to_array(v)
  let bytes = float_arr_to_bytes(arr)
  numbt_vsort_desc(bytes, arr.length())
  let result = bytes_to_float_arr(bytes, arr.length())
  vec_from_array(result)
}

///|
fn argsort_quicksort(
  indices : Array[Int],
  values : Vec,
  lo : Int,
  hi : Int,
  ascending : Bool,
) -> Unit {
  if lo < hi {
    let pivot_idx = argsort_partition(indices, values, lo, hi, ascending)
    argsort_quicksort(indices, values, lo, pivot_idx - 1, ascending)
    argsort_quicksort(indices, values, pivot_idx + 1, hi, ascending)
  }
}

///|
fn argsort_partition(
  indices : Array[Int],
  values : Vec,
  lo : Int,
  hi : Int,
  ascending : Bool,
) -> Int {
  let pivot_val = vec_at(values, indices[hi])
  let mut i = lo - 1
  for j = lo; j < hi; j = j + 1 {
    let cmp = if ascending {
      vec_at(values, indices[j]) <= pivot_val
    } else {
      vec_at(values, indices[j]) >= pivot_val
    }
    if cmp {
      i = i + 1
      let tmp = indices[i]
      indices[i] = indices[j]
      indices[j] = tmp
    }
  }
  let tmp = indices[i + 1]
  indices[i + 1] = indices[hi]
  indices[hi] = tmp
  i + 1
}

///|
/// Return indices that would sort the vector (ascending)
pub fn vec_argsort(v : Vec) -> Array[Int] {
  let n = v.len
  let indices = Array::makei(n, fn(i) { i })
  if n > 1 {
    argsort_quicksort(indices, v, 0, n - 1, true)
  }
  indices
}

///|
/// Return indices that would sort the vector (descending)
pub fn vec_argsort_desc(v : Vec) -> Array[Int] {
  let n = v.len
  let indices = Array::makei(n, fn(i) { i })
  if n > 1 {
    argsort_quicksort(indices, v, 0, n - 1, false)
  }
  indices
}

// ============================================================================
// Conditional operations (where / nonzero)
// ============================================================================

///|
/// Return indices of non-zero elements
pub fn vec_nonzero(v : Vec) -> Array[Int] {
  let indices : Array[Int] = []
  let zero = Float::from_int(0)
  for i = 0; i < v.len; i = i + 1 {
    if vec_at(v, i) != zero {
      indices.push(i)
    }
  }
  indices
}

///|
/// Element-wise greater than: out[i] = 1.0 if v[i] > threshold else 0.0
pub fn vec_gt(v : Vec, threshold : Float) -> Vec {
  let out = vec_zeros(v.len)
  let one = Float::from_int(1)
  for i = 0; i < v.len; i = i + 1 {
    if vec_at(v, i) > threshold {
      vec_set(out, i, one)
    }
  }
  out
}

///|
/// Element-wise greater than or equal
pub fn vec_ge(v : Vec, threshold : Float) -> Vec {
  let out = vec_zeros(v.len)
  let one = Float::from_int(1)
  for i = 0; i < v.len; i = i + 1 {
    if vec_at(v, i) >= threshold {
      vec_set(out, i, one)
    }
  }
  out
}

///|
/// Element-wise less than
pub fn vec_lt(v : Vec, threshold : Float) -> Vec {
  let out = vec_zeros(v.len)
  let one = Float::from_int(1)
  for i = 0; i < v.len; i = i + 1 {
    if vec_at(v, i) < threshold {
      vec_set(out, i, one)
    }
  }
  out
}

///|
/// Element-wise less than or equal
pub fn vec_le(v : Vec, threshold : Float) -> Vec {
  let out = vec_zeros(v.len)
  let one = Float::from_int(1)
  for i = 0; i < v.len; i = i + 1 {
    if vec_at(v, i) <= threshold {
      vec_set(out, i, one)
    }
  }
  out
}

///|
/// Select elements based on mask: out[i] = x[i] if mask[i] > 0 else y[i]
pub fn vec_where(mask : Vec, x : Vec, y : Vec) -> Vec {
  vec_check_same_len(mask, x)
  vec_check_same_len(x, y)
  let out = vec_zeros(mask.len)
  let zero = Float::from_int(0)
  for i = 0; i < mask.len; i = i + 1 {
    if vec_at(mask, i) > zero {
      vec_set(out, i, vec_at(x, i))
    } else {
      vec_set(out, i, vec_at(y, i))
    }
  }
  out
}

///|
/// Select elements based on mask with scalar fallback
pub fn vec_where_scalar(mask : Vec, x : Vec, y : Float) -> Vec {
  vec_check_same_len(mask, x)
  let out = vec_zeros(mask.len)
  let zero = Float::from_int(0)
  for i = 0; i < mask.len; i = i + 1 {
    if vec_at(mask, i) > zero {
      vec_set(out, i, vec_at(x, i))
    } else {
      vec_set(out, i, y)
    }
  }
  out
}

// ============================================================================
// Cumulative operations
// ============================================================================

///|
/// Cumulative sum: out[i] = sum(v[0..i+1])
pub fn vec_cumsum(v : Vec) -> Vec {
  let out = vec_zeros(v.len)
  if v.len == 0 {
    return out
  }
  let mut acc = Float::from_int(0)
  for i = 0; i < v.len; i = i + 1 {
    acc = acc + vec_at(v, i)
    vec_set(out, i, acc)
  }
  out
}

///|
/// Cumulative product: out[i] = prod(v[0..i+1])
pub fn vec_cumprod(v : Vec) -> Vec {
  let out = vec_zeros(v.len)
  if v.len == 0 {
    return out
  }
  let mut acc = Float::from_int(1)
  for i = 0; i < v.len; i = i + 1 {
    acc = acc * vec_at(v, i)
    vec_set(out, i, acc)
  }
  out
}

///|
/// Difference between consecutive elements: out[i] = v[i+1] - v[i]
pub fn vec_diff(v : Vec) -> Vec {
  if v.len <= 1 {
    return vec_zeros(0)
  }
  let out = vec_zeros(v.len - 1)
  for i = 0; i < v.len - 1; i = i + 1 {
    vec_set(out, i, vec_at(v, i + 1) - vec_at(v, i))
  }
  out
}

// ============================================================================
// Linear algebra extras (outer, diag, trace)
// ============================================================================

///|
/// Outer product: out[i,j] = a[i] * b[j]
/// Uses BLAS sger from Accelerate framework for optimal performance
pub fn vec_outer(a : Vec, b : Vec) -> Mat {
  let a_bytes = float_arr_to_bytes(vec_to_array(a))
  let b_bytes = float_arr_to_bytes(vec_to_array(b))
  let out_bytes : FixedArray[Byte] = FixedArray::make(
    a.len * b.len * 4,
    b'\x00',
  )
  numbt_outer(a_bytes, b_bytes, out_bytes, a.len, b.len)
  let out_arr = bytes_to_float_arr(out_bytes, a.len * b.len)
  mat_view(out_arr, a.len, b.len)
}

///|
/// Create diagonal matrix from vector
pub fn vec_diag(v : Vec) -> Mat {
  let n = v.len
  let out = mat_zeros(n, n)
  for i = 0; i < n; i = i + 1 {
    mat_set(out, i, i, vec_at(v, i))
  }
  out
}

///|
/// Extract diagonal elements from matrix
pub fn mat_diag(m : Mat) -> Vec {
  let n = if m.rows < m.cols { m.rows } else { m.cols }
  let out = vec_zeros(n)
  for i = 0; i < n; i = i + 1 {
    vec_set(out, i, mat_at(m, i, i))
  }
  out
}

///|
/// Trace of matrix (sum of diagonal elements)
pub fn mat_trace(m : Mat) -> Float {
  let n = if m.rows < m.cols { m.rows } else { m.cols }
  let mut sum = Float::from_int(0)
  for i = 0; i < n; i = i + 1 {
    sum = sum + mat_at(m, i, i)
  }
  sum
}

// ============================================================================
// Concatenate / Stack operations
// ============================================================================

///|
/// Concatenate multiple vectors into one
pub fn vec_concatenate(vecs : Array[Vec]) -> Vec {
  let mut total_len = 0
  for i = 0; i < vecs.length(); i = i + 1 {
    total_len = total_len + vecs[i].len
  }
  let data : Array[Float] = []
  for i = 0; i < vecs.length(); i = i + 1 {
    let v = vecs[i]
    for j = 0; j < v.len; j = j + 1 {
      data.push(vec_at(v, j))
    }
  }
  vec_from_array(data)
}

///|
/// Vertical stack: stack matrices along rows (axis=0)
pub fn mat_vstack(mats : Array[Mat]) -> Mat {
  if mats.length() == 0 {
    return mat_zeros(0, 0)
  }
  let cols = mats[0].cols
  let mut total_rows = 0
  for i = 0; i < mats.length(); i = i + 1 {
    if mats[i].cols != cols {
      panic()
    }
    total_rows = total_rows + mats[i].rows
  }
  let data : Array[Float] = []
  for i = 0; i < mats.length(); i = i + 1 {
    let m = mats[i]
    for j = 0; j < m.data.length(); j = j + 1 {
      data.push(m.data[j])
    }
  }
  mat_view(data, total_rows, cols)
}

///|
/// Horizontal stack: stack matrices along columns (axis=1)
pub fn mat_hstack(mats : Array[Mat]) -> Mat {
  if mats.length() == 0 {
    return mat_zeros(0, 0)
  }
  let rows = mats[0].rows
  let mut total_cols = 0
  for i = 0; i < mats.length(); i = i + 1 {
    if mats[i].rows != rows {
      panic()
    }
    total_cols = total_cols + mats[i].cols
  }
  let out = mat_zeros(rows, total_cols)
  let mut col_offset = 0
  for k = 0; k < mats.length(); k = k + 1 {
    let m = mats[k]
    for i = 0; i < m.rows; i = i + 1 {
      for j = 0; j < m.cols; j = j + 1 {
        mat_set(out, i, col_offset + j, mat_at(m, i, j))
      }
    }
    col_offset = col_offset + m.cols
  }
  out
}

///|
/// Repeat vector n times
pub fn vec_repeat(v : Vec, n : Int) -> Vec {
  if n <= 0 {
    return vec_zeros(0)
  }
  let data : Array[Float] = []
  for k = 0; k < n; k = k + 1 {
    for j = 0; j < v.len; j = j + 1 {
      data.push(vec_at(v, j))
    }
  }
  vec_from_array(data)
}

///|
/// Tile matrix: repeat along rows and columns
pub fn mat_tile(m : Mat, rows_repeat : Int, cols_repeat : Int) -> Mat {
  if rows_repeat <= 0 || cols_repeat <= 0 {
    return mat_zeros(0, 0)
  }
  let out = mat_zeros(m.rows * rows_repeat, m.cols * cols_repeat)
  for ri = 0; ri < rows_repeat; ri = ri + 1 {
    for ci = 0; ci < cols_repeat; ci = ci + 1 {
      let row_off = ri * m.rows
      let col_off = ci * m.cols
      for i = 0; i < m.rows; i = i + 1 {
        for j = 0; j < m.cols; j = j + 1 {
          mat_set(out, row_off + i, col_off + j, mat_at(m, i, j))
        }
      }
    }
  }
  out
}

// ============================================================================
// Math functions: floor, ceil, round
// ============================================================================

///|
/// Element-wise floor: output[i] = floor(input[i])
pub fn vec_floor_into(input : Vec, output~ : Vec) -> Unit {
  vec_check_same_len(output, input)
  for i = 0; i < input.len; i = i + 1 {
    vec_set(output, i, vec_at(input, i).floor())
  }
}

///|
pub fn vec_floor(input : Vec) -> Vec {
  let output = vec_zeros(input.len)
  vec_floor_into(input, output~)
  output
}

///|
/// Element-wise ceiling: output[i] = ceil(input[i])
pub fn vec_ceil_into(input : Vec, output~ : Vec) -> Unit {
  vec_check_same_len(output, input)
  for i = 0; i < input.len; i = i + 1 {
    vec_set(output, i, vec_at(input, i).ceil())
  }
}

///|
pub fn vec_ceil(input : Vec) -> Vec {
  let output = vec_zeros(input.len)
  vec_ceil_into(input, output~)
  output
}

///|
/// Element-wise rounding: output[i] = round(input[i])
pub fn vec_round_into(input : Vec, output~ : Vec) -> Unit {
  vec_check_same_len(output, input)
  for i = 0; i < input.len; i = i + 1 {
    vec_set(output, i, vec_at(input, i).round())
  }
}

///|
pub fn vec_round(input : Vec) -> Vec {
  let output = vec_zeros(input.len)
  vec_round_into(input, output~)
  output
}

// ============================================================================
// Statistics: median, percentile
// ============================================================================

///|
/// Median of vector elements
pub fn vec_median(v : Vec) -> Float {
  if v.len == 0 {
    return Float::from_int(0)
  }
  let sorted = vec_sort(v)
  let n = v.len
  if n % 2 == 1 {
    vec_at(sorted, n / 2)
  } else {
    (vec_at(sorted, n / 2 - 1) + vec_at(sorted, n / 2)) / Float::from_int(2)
  }
}

///|
/// Percentile (0-100) of vector elements
pub fn vec_percentile(v : Vec, p : Float) -> Float {
  if v.len == 0 {
    return Float::from_int(0)
  }
  let sorted = vec_sort(v)
  let n = v.len
  // Clamp p to [0, 100]
  let p_clamped = if p < Float::from_int(0) {
    Float::from_int(0)
  } else if p > Float::from_int(100) {
    Float::from_int(100)
  } else {
    p
  }
  // Linear interpolation
  let pos = Float::from_int(n - 1) * p_clamped / Float::from_int(100)
  let idx = pos.floor().to_int()
  let frac = pos - Float::from_int(idx)
  if idx >= n - 1 {
    vec_at(sorted, n - 1)
  } else {
    vec_at(sorted, idx) * (Float::from_int(1) - frac) +
    vec_at(sorted, idx + 1) * frac
  }
}

///|
/// Quantile (0-1) of vector elements
pub fn vec_quantile(v : Vec, q : Float) -> Float {
  vec_percentile(v, q * Float::from_int(100))
}

// ============================================================================
// Utility: unique, searchsorted
// ============================================================================

///|
/// Return sorted unique elements
pub fn vec_unique(v : Vec) -> Vec {
  if v.len == 0 {
    return vec_zeros(0)
  }
  let sorted = vec_sort(v)
  let result : Array[Float] = [vec_at(sorted, 0)]
  for i = 1; i < sorted.len; i = i + 1 {
    let curr = vec_at(sorted, i)
    let prev = vec_at(sorted, i - 1)
    if curr != prev {
      result.push(curr)
    }
  }
  vec_from_array(result)
}

///|
/// Binary search: find insertion point for value in sorted vector
/// Returns index where value should be inserted to maintain sorted order
pub fn vec_searchsorted(sorted : Vec, value : Float) -> Int {
  let mut lo = 0
  let mut hi = sorted.len
  while lo < hi {
    let mid = (lo + hi) / 2
    if vec_at(sorted, mid) < value {
      lo = mid + 1
    } else {
      hi = mid
    }
  }
  lo
}

///|
/// Binary search (right): find rightmost insertion point
pub fn vec_searchsorted_right(sorted : Vec, value : Float) -> Int {
  let mut lo = 0
  let mut hi = sorted.len
  while lo < hi {
    let mid = (lo + hi) / 2
    if vec_at(sorted, mid) <= value {
      lo = mid + 1
    } else {
      hi = mid
    }
  }
  lo
}

// ============================================================================
// Random number generation
// ============================================================================

///|
/// Set random seed
pub fn random_seed(seed : Int) -> Unit {
  numbt_seed(seed)
}

///|
/// Generate vector of uniform random values in [0, 1)
pub fn vec_rand(n : Int) -> Vec {
  let bytes : FixedArray[Byte] = FixedArray::make(n * 4, b'\x00')
  numbt_rand(bytes, n)
  let arr = bytes_to_float_arr(bytes, n)
  vec_from_array(arr)
}

///|
/// Generate vector of standard normal random values (mean=0, std=1)
pub fn vec_randn(n : Int) -> Vec {
  let bytes : FixedArray[Byte] = FixedArray::make(n * 4, b'\x00')
  numbt_randn(bytes, n)
  let arr = bytes_to_float_arr(bytes, n)
  vec_from_array(arr)
}

///|
/// Generate matrix of uniform random values in [0, 1)
pub fn mat_rand(rows : Int, cols : Int) -> Mat {
  let n = rows * cols
  let bytes : FixedArray[Byte] = FixedArray::make(n * 4, b'\x00')
  numbt_rand(bytes, n)
  let arr = bytes_to_float_arr(bytes, n)
  mat_view(arr, rows, cols)
}

///|
/// Generate matrix of standard normal random values
pub fn mat_randn(rows : Int, cols : Int) -> Mat {
  let n = rows * cols
  let bytes : FixedArray[Byte] = FixedArray::make(n * 4, b'\x00')
  numbt_randn(bytes, n)
  let arr = bytes_to_float_arr(bytes, n)
  mat_view(arr, rows, cols)
}

///|
/// Shuffle vector elements in place
pub fn vec_shuffle(v : Vec) -> Vec {
  let arr = vec_to_array(v)
  let n = arr.length()
  // Simple Fisher-Yates using vec_rand for random numbers
  let rand_vals = vec_rand(n)
  for i = n - 1; i > 0; i = i - 1 {
    let j = (vec_at(rand_vals, i) * Float::from_int(i + 1)).floor().to_int()
    let tmp = arr[i]
    arr[i] = arr[j]
    arr[j] = tmp
  }
  vec_from_array(arr)
}

// ============================================================================
// Linear algebra: inv, solve
// ============================================================================

///|
/// Matrix inverse using LAPACK sgetrf/sgetri
/// Returns None if matrix is singular
pub fn mat_inv(m : Mat) -> Mat? {
  if m.rows != m.cols {
    return None
  }
  let n = m.rows
  let a_arr = mat_to_array(m)
  let a_bytes = float_arr_to_bytes(a_arr)
  let info = numbt_inv(a_bytes, n)
  if info != 0 {
    return None
  }
  let result = bytes_to_float_arr(a_bytes, n * n)
  Some(mat_view(result, n, n))
}

///|
/// Solve linear system A @ x = b using LAPACK sgesv
/// Returns None if system is singular
pub fn mat_solve(a : Mat, b : Vec) -> Vec? {
  if a.rows != a.cols || a.rows != b.len {
    return None
  }
  let n = a.rows
  let a_arr = mat_to_array(a)
  let b_arr = vec_to_array(b)
  let a_bytes = float_arr_to_bytes(a_arr)
  let b_bytes = float_arr_to_bytes(b_arr)
  let info = numbt_solve(a_bytes, b_bytes, n)
  if info != 0 {
    return None
  }
  let result = bytes_to_float_arr(b_bytes, n)
  Some(vec_from_array(result))
}
