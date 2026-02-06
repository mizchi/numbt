///|
fn assert_near(actual : Float, expected : Float, tol : Float) -> Unit {
  let diff = (actual - expected).abs()
  if diff > tol {
    panic()
  }
}

///|
test "vec_ops_basic" {
  let left = vec_from_array([1.0, 2.0, 3.0])
  let right = vec_from_array([3.0, 2.0, 1.0])
  let added = left + right
  let subbed = left - right
  let mulled = left * right
  assert_eq(vec_to_array(added), [4.0, 4.0, 4.0])
  assert_eq(vec_to_array(subbed), [-2.0, 0.0, 2.0])
  assert_eq(vec_to_array(mulled), [3.0, 4.0, 3.0])
}

///|
test "matmul_vec_bias_relu" {
  let weight = mat_view([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3)
  let input = vec_from_array([10.0, 20.0])
  let bias = vec_from_array([1.0, 1.0, 1.0])
  let output = vec_new(3, 0.0)
  matmul_vec_bias_relu_into(weight, input, bias, output~)
  assert_eq(vec_to_array(output), [91.0, 121.0, 151.0])
}

///|
test "softmax_into" {
  let input = vec_from_array([0.0, 1.0])
  let output = vec_new(2, 0.0)
  softmax_into(input~, output~)
  let values = vec_to_array(output)
  assert_near(values[0], 0.26894, 0.0001)
  assert_near(values[1], 0.73105, 0.0001)
}

///|
test "argmax" {
  let values = vec_from_array([1.0, 3.0, 2.0])
  assert_eq(vec_argmax(values), 1)
}

// ============================================================================
// BLAS tests
// ============================================================================

///|
test "vec_dot" {
  let x = vec_from_array([1.0, 2.0, 3.0])
  let y = vec_from_array([4.0, 5.0, 6.0])
  let result = vec_dot(x, y)
  // 1*4 + 2*5 + 3*6 = 32
  assert_near(result, 32.0, 0.0001)
}

///|
test "vec_nrm2" {
  let x = vec_from_array([3.0, 4.0])
  let result = vec_nrm2(x)
  // sqrt(9 + 16) = 5
  assert_near(result, 5.0, 0.0001)
}

///|
test "matmul_vec_blas_into" {
  let weight = mat_view([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3)
  let input = vec_from_array([10.0, 20.0])
  let output = vec_new(3, 0.0)
  matmul_vec_blas_into(weight, input, output~)
  // output[j] = sum_i(input[i] * weight[i, j])
  // output[0] = 10*1 + 20*4 = 90
  // output[1] = 10*2 + 20*5 = 120
  // output[2] = 10*3 + 20*6 = 150
  assert_eq(vec_to_array(output), [90.0, 120.0, 150.0])
}

///|
test "matmul_vec_bias_relu_blas" {
  let weight = mat_view([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3)
  let input = vec_from_array([10.0, 20.0])
  let bias = vec_from_array([1.0, 1.0, 1.0])
  let output = vec_new(3, 0.0)
  matmul_vec_bias_relu_blas_into(weight, input, bias, output~)
  assert_eq(vec_to_array(output), [91.0, 121.0, 151.0])
}

///|
test "batch_matmul" {
  let f = Float::from_int
  // Input: 2x3 (2 samples, 3 features)
  // Weight: 3x2 (3 input, 2 output)
  // Out: 2x2
  let input : Array[Float] = [f(1), f(2), f(3), f(4), f(5), f(6)]
  let weight : Array[Float] = [f(1), f(2), f(0), f(1), f(1), f(0)]
  let out : Array[Float] = [f(0), f(0), f(0), f(0)]
  batch_matmul(input, weight, out, 2, 3, 2)
  // Row 0: [1,2,3] @ [[1,2],[0,1],[1,0]] = [4, 4]
  // Row 1: [4,5,6] @ [[1,2],[0,1],[1,0]] = [10, 13]
  assert_eq(out, [f(4), f(4), f(10), f(13)])
}

///|
test "batch_matmul_bias_relu" {
  let f = Float::from_int
  let input : Array[Float] = [f(1), f(2), f(3), f(4), f(5), f(6)]
  let weight : Array[Float] = [f(1), f(2), f(0), f(1), f(1), f(0)]
  let bias : Array[Float] = [
    Float::from_int(1) / Float::from_int(2),
    Float::from_int(-10),
  ]
  let out : Array[Float] = [f(0), f(0), f(0), f(0)]
  batch_matmul_bias_relu(input, weight, bias, out, 2, 3, 2)
  // Before ReLU: [4.5, -6, 10.5, 3]
  // After ReLU: [4.5, 0, 10.5, 3]
  let half = Float::from_int(1) / Float::from_int(2)
  assert_eq(out, [f(4) + half, f(0), f(10) + half, f(3)])
}

// ============================================================================
// Scalar operations tests
// ============================================================================

///|
test "vec_neg" {
  let v = vec_from_array([1.0, -2.0, 3.0])
  let neg_v = -v
  assert_eq(vec_to_array(neg_v), [-1.0, 2.0, -3.0])
}

///|
test "vec_add_scalar" {
  let v = vec_from_array([1.0, 2.0, 3.0])
  let result = vec_add_scalar(v, 10.0)
  assert_eq(vec_to_array(result), [11.0, 12.0, 13.0])
}

///|
test "vec_sub_scalar" {
  let v = vec_from_array([10.0, 20.0, 30.0])
  let result = vec_sub_scalar(v, 5.0)
  assert_eq(vec_to_array(result), [5.0, 15.0, 25.0])
}

///|
test "vec_mul_scalar" {
  let v = vec_from_array([1.0, 2.0, 3.0])
  let result = vec_mul_scalar(v, 2.0)
  assert_eq(vec_to_array(result), [2.0, 4.0, 6.0])
}

///|
test "vec_div_scalar" {
  let v = vec_from_array([10.0, 20.0, 30.0])
  let result = vec_div_scalar(v, 10.0)
  assert_eq(vec_to_array(result), [1.0, 2.0, 3.0])
}

///|
test "vec_scalar_inplace" {
  let v = vec_from_array([1.0, 2.0, 3.0])
  vec_mul_scalar_inplace(v, 2.0)
  assert_eq(vec_to_array(v), [2.0, 4.0, 6.0])
  vec_add_scalar_inplace(v, 1.0)
  assert_eq(vec_to_array(v), [3.0, 5.0, 7.0])
}

// ============================================================================
// Mat tests
// ============================================================================

///|
test "mat_new_and_access" {
  let m = mat_new(2, 3, 1.0)
  assert_eq(mat_rows(m), 2)
  assert_eq(mat_cols(m), 3)
  assert_eq(mat_at(m, 0, 0), 1.0)
  mat_set(m, 1, 2, 5.0)
  assert_eq(mat_at(m, 1, 2), 5.0)
}

///|
test "mat_zeros_ones" {
  let z = mat_zeros(2, 2)
  assert_eq(mat_to_array(z), [0.0, 0.0, 0.0, 0.0])
  let o = mat_ones(2, 2)
  assert_eq(mat_to_array(o), [1.0, 1.0, 1.0, 1.0])
}

///|
test "mat_eye" {
  let e = mat_eye(3)
  assert_eq(mat_to_array(e), [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
}

///|
test "mat_transpose" {
  let m = mat_view([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3)
  let t = mat_transpose(m)
  assert_eq(mat_rows(t), 3)
  assert_eq(mat_cols(t), 2)
  assert_eq(mat_to_array(t), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
}

///|
test "mat_ops_basic" {
  let a = mat_view([1.0, 2.0, 3.0, 4.0], 2, 2)
  let b = mat_view([4.0, 3.0, 2.0, 1.0], 2, 2)
  let added = a + b
  let subbed = a - b
  let mulled = a * b
  assert_eq(mat_to_array(added), [5.0, 5.0, 5.0, 5.0])
  assert_eq(mat_to_array(subbed), [-3.0, -1.0, 1.0, 3.0])
  assert_eq(mat_to_array(mulled), [4.0, 6.0, 6.0, 4.0])
}

///|
test "mat_neg" {
  let m = mat_view([1.0, -2.0, 3.0, -4.0], 2, 2)
  let neg_m = -m
  assert_eq(mat_to_array(neg_m), [-1.0, 2.0, -3.0, 4.0])
}

///|
test "mat_scalar_ops" {
  let m = mat_view([1.0, 2.0, 3.0, 4.0], 2, 2)
  let added = mat_add_scalar(m, 10.0)
  let mulled = mat_mul_scalar(m, 2.0)
  assert_eq(mat_to_array(added), [11.0, 12.0, 13.0, 14.0])
  assert_eq(mat_to_array(mulled), [2.0, 4.0, 6.0, 8.0])
}

///|
test "mat_row" {
  let m = mat_view([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3)
  let row0 = mat_row(m, 0)
  let row1 = mat_row(m, 1)
  assert_eq(vec_to_array(row0), [1.0, 2.0, 3.0])
  assert_eq(vec_to_array(row1), [4.0, 5.0, 6.0])
}

///|
test "mat_matmul" {
  // A: 2x3, B: 3x2 -> C: 2x2
  let a = mat_view([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3)
  let b = mat_view([1.0, 2.0, 0.0, 1.0, 1.0, 0.0], 3, 2)
  let c = mat_matmul(a, b)
  // Row 0: [1,2,3] @ [[1,2],[0,1],[1,0]] = [4, 4]
  // Row 1: [4,5,6] @ [[1,2],[0,1],[1,0]] = [10, 13]
  assert_eq(mat_rows(c), 2)
  assert_eq(mat_cols(c), 2)
  assert_eq(mat_to_array(c), [4.0, 4.0, 10.0, 13.0])
}

// ============================================================================
// Array creation tests
// ============================================================================

///|
test "vec_zeros_ones" {
  let z = vec_zeros(3)
  let o = vec_ones(3)
  assert_eq(vec_to_array(z), [0.0, 0.0, 0.0])
  assert_eq(vec_to_array(o), [1.0, 1.0, 1.0])
}

///|
test "vec_from_fn" {
  let v = vec_from_fn(5, fn(i) { Float::from_int(i * i) })
  assert_eq(vec_to_array(v), [0.0, 1.0, 4.0, 9.0, 16.0])
}

///|
test "vec_arange" {
  // np.arange(0, 5, 1) -> [0, 1, 2, 3, 4]
  let v1 = vec_arange(0.0, 5.0, 1.0)
  assert_eq(vec_to_array(v1), [0.0, 1.0, 2.0, 3.0, 4.0])
  // np.arange(0, 1, 0.25) -> [0, 0.25, 0.5, 0.75]
  let v2 = vec_arange(0.0, 1.0, 0.25)
  assert_eq(vec_len(v2), 4)
  assert_near(vec_at(v2, 0), 0.0, 0.0001)
  assert_near(vec_at(v2, 1), 0.25, 0.0001)
  assert_near(vec_at(v2, 2), 0.5, 0.0001)
  assert_near(vec_at(v2, 3), 0.75, 0.0001)
}

///|
test "vec_linspace" {
  // np.linspace(0, 1, 5) -> [0, 0.25, 0.5, 0.75, 1.0]
  let v = vec_linspace(0.0, 1.0, 5)
  assert_eq(vec_len(v), 5)
  assert_near(vec_at(v, 0), 0.0, 0.0001)
  assert_near(vec_at(v, 1), 0.25, 0.0001)
  assert_near(vec_at(v, 2), 0.5, 0.0001)
  assert_near(vec_at(v, 3), 0.75, 0.0001)
  assert_near(vec_at(v, 4), 1.0, 0.0001)
}

// ============================================================================
// Math function tests
// ============================================================================

///|
test "vec_exp" {
  let v = vec_from_array([0.0, 1.0, 2.0])
  let r = vec_exp(v)
  assert_near(vec_at(r, 0), 1.0, 0.0001)
  assert_near(vec_at(r, 1), 2.7182, 0.001)
  assert_near(vec_at(r, 2), 7.3890, 0.001)
}

///|
test "vec_log" {
  let v = vec_from_array([1.0, 2.7182818, 7.3890561])
  let r = vec_log(v)
  assert_near(vec_at(r, 0), 0.0, 0.0001)
  assert_near(vec_at(r, 1), 1.0, 0.0001)
  assert_near(vec_at(r, 2), 2.0, 0.0001)
}

///|
test "vec_sqrt" {
  let v = vec_from_array([0.0, 1.0, 4.0, 9.0])
  let r = vec_sqrt(v)
  assert_eq(vec_to_array(r), [0.0, 1.0, 2.0, 3.0])
}

///|
test "vec_abs" {
  let v = vec_from_array([-3.0, -1.0, 0.0, 2.0])
  let r = vec_abs(v)
  assert_eq(vec_to_array(r), [3.0, 1.0, 0.0, 2.0])
}

///|
test "vec_pow" {
  let v = vec_from_array([1.0, 2.0, 3.0])
  let r = vec_pow(v, 2.0)
  assert_eq(vec_to_array(r), [1.0, 4.0, 9.0])
}

///|
test "vec_clip" {
  let v = vec_from_array([-1.0, 0.5, 2.0])
  let r = vec_clip(v, 0.0, 1.0)
  assert_eq(vec_to_array(r), [0.0, 0.5, 1.0])
}

///|
test "vec_trig" {
  let pi : Float = 3.14159265
  let half_pi = pi / Float::from_int(2)
  let v = vec_from_array([Float::from_int(0), half_pi, pi])
  let s = vec_sin(v)
  let c = vec_cos(v)
  assert_near(vec_at(s, 0), 0.0, 0.0001)
  assert_near(vec_at(s, 1), 1.0, 0.0001)
  assert_near(vec_at(s, 2), 0.0, 0.0001)
  assert_near(vec_at(c, 0), 1.0, 0.0001)
  assert_near(vec_at(c, 1), 0.0, 0.0001)
  assert_near(vec_at(c, 2), -1.0, 0.0001)
}

///|
test "vec_tanh" {
  let v = vec_from_array([0.0, 1.0, -1.0])
  let r = vec_tanh(v)
  assert_near(vec_at(r, 0), 0.0, 0.0001)
  assert_near(vec_at(r, 1), 0.7615, 0.0001)
  assert_near(vec_at(r, 2), -0.7615, 0.0001)
}

// ============================================================================
// Statistics tests
// ============================================================================

///|
test "vec_min_max" {
  let v = vec_from_array([3.0, 1.0, 4.0, 1.0, 5.0])
  assert_eq(vec_min(v), 1.0)
  assert_eq(vec_max(v), 5.0)
}

///|
test "vec_argmin" {
  let v = vec_from_array([3.0, 1.0, 4.0, 0.5, 5.0])
  assert_eq(vec_argmin(v), 3)
  assert_eq(vec_argmax(v), 4)
}

///|
test "vec_mean" {
  let v = vec_from_array([1.0, 2.0, 3.0, 4.0, 5.0])
  assert_eq(vec_mean(v), 3.0)
}

///|
test "vec_var_std" {
  // np.var([1, 2, 3, 4, 5]) = 2.0
  // np.std([1, 2, 3, 4, 5]) = sqrt(2) ≈ 1.4142
  let v = vec_from_array([1.0, 2.0, 3.0, 4.0, 5.0])
  assert_near(vec_var(v), 2.0, 0.0001)
  assert_near(vec_std(v), 1.4142, 0.0001)
}

///|
test "vec_prod" {
  let v = vec_from_array([1.0, 2.0, 3.0, 4.0])
  assert_eq(vec_prod(v), 24.0)
}

// ============================================================================
// Mat statistics tests
// ============================================================================

///|
test "mat_sum_mean" {
  let m = mat_view([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3)
  // sum = 1+2+3+4+5+6 = 21
  assert_eq(mat_sum(m), 21.0)
  // mean = 21 / 6 = 3.5
  assert_eq(mat_mean(m), 3.5)
}

///|
test "mat_min_max" {
  let m = mat_view([3.0, 1.0, 4.0, 1.0, 5.0, 9.0], 2, 3)
  assert_eq(mat_min(m), 1.0)
  assert_eq(mat_max(m), 9.0)
}

///|
test "mat_sum_axis" {
  // [[1, 2, 3],
  //  [4, 5, 6]]
  let m = mat_view([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3)
  // axis=0: [5, 7, 9]
  let axis0 = mat_sum_axis0(m)
  assert_eq(vec_to_array(axis0), [5.0, 7.0, 9.0])
  // axis=1: [6, 15]
  let axis1 = mat_sum_axis1(m)
  assert_eq(vec_to_array(axis1), [6.0, 15.0])
}

///|
test "mat_mean_axis" {
  let m = mat_view([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3)
  // axis=0: [2.5, 3.5, 4.5]
  let axis0 = mat_mean_axis0(m)
  assert_eq(vec_to_array(axis0), [2.5, 3.5, 4.5])
  // axis=1: [2.0, 5.0]
  let axis1 = mat_mean_axis1(m)
  assert_eq(vec_to_array(axis1), [2.0, 5.0])
}

// ============================================================================
// Comparison tests
// ============================================================================

///|
test "vec_allclose" {
  let a = vec_from_array([1.0, 2.0, 3.0])
  let b = vec_from_array([1.0001, 1.9999, 3.0001])
  assert_true(vec_allclose(a, b, 0.001))
  assert_false(vec_allclose(a, b, 0.00001))
}

///|
test "mat_allclose" {
  let a = mat_view([1.0, 2.0, 3.0, 4.0], 2, 2)
  let b = mat_view([1.0001, 1.9999, 3.0001, 3.9999], 2, 2)
  assert_true(mat_allclose(a, b, 0.001))
  assert_false(mat_allclose(a, b, 0.00001))
}

// ============================================================================
// Slice and reshape tests
// ============================================================================

///|
test "vec_slice" {
  let v = vec_from_array([1.0, 2.0, 3.0, 4.0, 5.0])
  let s = vec_slice(v, 1, 4)
  assert_eq(vec_to_array(s), [2.0, 3.0, 4.0])
}

///|
test "mat_flatten" {
  let m = mat_view([1.0, 2.0, 3.0, 4.0], 2, 2)
  let v = mat_flatten(m)
  assert_eq(vec_to_array(v), [1.0, 2.0, 3.0, 4.0])
}

///|
test "vec_reshape" {
  let v = vec_from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  let m = vec_reshape(v, 2, 3)
  assert_eq(mat_rows(m), 2)
  assert_eq(mat_cols(m), 3)
  assert_eq(mat_at(m, 0, 2), 3.0)
  assert_eq(mat_at(m, 1, 0), 4.0)
}

// ============================================================================
// NumPy-style integration tests
// ============================================================================

///|
test "numpy_style_linear_algebra" {
  // Test: A @ B = C
  // A = [[1, 0], [0, 1]] (identity)
  // B = [[1, 2], [3, 4]]
  // A @ B = B
  let a = mat_eye(2)
  let b = mat_view([1.0, 2.0, 3.0, 4.0], 2, 2)
  let c = mat_matmul(a, b)
  assert_true(mat_allclose(b, c, 0.0001))
}

///|
test "numpy_style_broadcasting_scalar" {
  // Similar to np.array([1, 2, 3]) + 1 = [2, 3, 4]
  let v = vec_from_array([1.0, 2.0, 3.0])
  let result = vec_add_scalar(v, 1.0)
  assert_eq(vec_to_array(result), [2.0, 3.0, 4.0])
}

///|
test "numpy_style_chained_ops" {
  // np.exp(np.log([1, 2, 3])) ≈ [1, 2, 3]
  let v = vec_from_array([1.0, 2.0, 3.0])
  let result = vec_exp(vec_log(v))
  assert_true(vec_allclose(v, result, 0.0001))
}

///|
test "numpy_style_matrix_transpose_matmul" {
  // A^T @ A is symmetric positive definite
  let a = mat_view([1.0, 2.0, 3.0, 4.0], 2, 2)
  let at = mat_transpose(a)
  let ata = mat_matmul(at, a)
  // Check symmetry: ata[0,1] == ata[1,0]
  assert_eq(mat_at(ata, 0, 1), mat_at(ata, 1, 0))
}

///|
test "numpy_style_normalize" {
  // Normalize vector: v / ||v||
  let v = vec_from_array([3.0, 4.0])
  let norm = vec_nrm2(v)
  let normalized = vec_div_scalar(v, norm)
  // Should have unit length
  assert_near(vec_nrm2(normalized), 1.0, 0.0001)
}

// ============================================================================
// NumPy test_multiarray.py style tests
// ============================================================================

///|
test "np_zeros_all_zero" {
  // From NumPy: assert_equal(np.count_nonzero(d), 0); assert_(not d.any())
  let d = vec_zeros(13)
  assert_eq(vec_sum(d), 0.0)
  // Check no element is nonzero
  for i = 0; i < vec_len(d); i = i + 1 {
    assert_eq(vec_at(d, i), 0.0)
  }
}

///|
test "np_ones_construction" {
  // From NumPy: d = np.ones(6); r = np.array([d, d]); assert_equal(r, np.ones((2, 6)))
  let d = vec_ones(6)
  // Reshape to 2x6
  let row0 = vec_to_array(d)
  let row1 = vec_to_array(d)
  let combined : Array[Float] = []
  for i = 0; i < row0.length(); i = i + 1 {
    combined.push(row0[i])
  }
  for i = 0; i < row1.length(); i = i + 1 {
    combined.push(row1[i])
  }
  let m = mat_view(combined, 2, 6)
  let expected = mat_ones(2, 6)
  assert_true(mat_allclose(m, expected, 0.0001))
}

// ============================================================================
// NumPy test_function_base.py style tests (linspace, arange)
// ============================================================================

///|
test "np_linspace_default_50" {
  // From NumPy: y = linspace(0, 10); assert_(len(y) == 50)
  let y = vec_linspace(0.0, 10.0, 50)
  assert_eq(vec_len(y), 50)
}

///|
test "np_linspace_endpoint" {
  // From NumPy: y = linspace(2, 10, num=100); assert_(y[-1] == 10)
  let y = vec_linspace(2.0, 10.0, 100)
  assert_eq(vec_at(y, 99), 10.0)
}

///|
test "np_linspace_single" {
  // From NumPy: y = list(linspace(0, 1, 1)); assert_(y == [0.0])
  let y = vec_linspace(0.0, 1.0, 1)
  assert_eq(vec_len(y), 1)
  assert_eq(vec_at(y, 0), 0.0)
}

///|
test "np_arange_linspace_equivalence" {
  // From NumPy: assert_equal(linspace(0, j, j+1, dtype=int), arange(j+1, dtype=int))
  // Test for j = 10: linspace(0, 10, 11) should equal arange(0, 11, 1)
  let lin = vec_linspace(0.0, 10.0, 11)
  let ara = vec_arange(0.0, 11.0, 1.0)
  assert_eq(vec_len(lin), vec_len(ara))
  for i = 0; i < vec_len(lin); i = i + 1 {
    assert_near(vec_at(lin, i), vec_at(ara, i), 0.0001)
  }
}

// ============================================================================
// NumPy test_linalg.py style tests
// ============================================================================

///|
test "np_dot_negative" {
  // From NumPy: a = [1, 2, 3, 4]; b = [-1, -2, -3, -4]
  // Expected: 1*(-1) + 2*(-2) + 3*(-3) + 4*(-4) = -30
  let a = vec_from_array([1.0, 2.0, 3.0, 4.0])
  let b = vec_from_array([-1.0, -2.0, -3.0, -4.0])
  assert_near(vec_dot(a, b), -30.0, 0.0001)
}

///|
test "np_matmul_transpose" {
  // From NumPy: x = np.arange(6).reshape((2, 3))
  // actual = np.linalg.matmul(x, x.T); expected = np.array([[5, 14], [14, 50]])
  // x = [[0, 1, 2], [3, 4, 5]]
  let x = mat_view([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], 2, 3)
  let xt = mat_transpose(x)
  let actual = mat_matmul(x, xt)
  // x @ x.T:
  // [0,1,2] @ [0,3] = 0*0 + 1*1 + 2*2 = 5
  // [0,1,2] @ [1,4] = 0*1 + 1*4 + 2*5 = 14 (wait, wrong)
  // Actually x.T = [[0, 3], [1, 4], [2, 5]] (3x2)
  // So x @ x.T (2x3 @ 3x2 = 2x2):
  // [0,1,2] @ [0,1,2]^T = 0*0 + 1*1 + 2*2 = 5
  // [0,1,2] @ [3,4,5]^T = 0*3 + 1*4 + 2*5 = 14
  // [3,4,5] @ [0,1,2]^T = 3*0 + 4*1 + 5*2 = 14
  // [3,4,5] @ [3,4,5]^T = 3*3 + 4*4 + 5*5 = 50
  let expected = mat_view([5.0, 14.0, 14.0, 50.0], 2, 2)
  assert_true(mat_allclose(actual, expected, 0.0001))
}

///|
test "np_matrix_transpose_roundtrip" {
  // From NumPy: actual = np.linalg.matrix_transpose(x); expected = x.T
  let x = mat_view([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3)
  let xt = mat_transpose(x)
  let xtt = mat_transpose(xt)
  // x.T.T == x
  assert_true(mat_allclose(x, xtt, 0.0001))
}

///|
test "np_eye_identity" {
  // From NumPy: result = identity(4); expected = np.eye(4)
  let eye4 = mat_eye(4)
  assert_eq(mat_rows(eye4), 4)
  assert_eq(mat_cols(eye4), 4)
  // Check diagonal is 1, off-diagonal is 0
  for i = 0; i < 4; i = i + 1 {
    for j = 0; j < 4; j = j + 1 {
      if i == j {
        assert_eq(mat_at(eye4, i, j), 1.0)
      } else {
        assert_eq(mat_at(eye4, i, j), 0.0)
      }
    }
  }
}

///|
test "np_vector_norm_3x3" {
  // From NumPy: x = np.arange(9).reshape((3, 3))
  // actual = np.linalg.vector_norm(x) ≈ 14.2828
  // ||[0,1,2,3,4,5,6,7,8]|| = sqrt(0+1+4+9+16+25+36+49+64) = sqrt(204) ≈ 14.2828
  let x = mat_view([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 3, 3)
  let v = mat_flatten(x)
  let norm = vec_nrm2(v)
  assert_near(norm, 14.2828, 0.001)
}

// ============================================================================
// Additional NumPy-style edge case tests
// ============================================================================

///|
test "np_arange_negative_step" {
  // np.arange(5, 0, -1) -> [5, 4, 3, 2, 1]
  let v = vec_arange(5.0, 0.0, -1.0)
  assert_eq(vec_len(v), 5)
  assert_eq(vec_at(v, 0), 5.0)
  assert_eq(vec_at(v, 4), 1.0)
}

///|
test "np_sum_empty_like" {
  // Edge case: sum of zeros should be 0
  let z = vec_zeros(100)
  assert_eq(vec_sum(z), 0.0)
}

///|
test "np_mean_single" {
  // mean of single element array
  let v = vec_from_array([42.0])
  assert_eq(vec_mean(v), 42.0)
}

///|
test "np_std_uniform" {
  // std of uniform array should be 0
  let v = vec_from_array([5.0, 5.0, 5.0, 5.0])
  assert_eq(vec_std(v), 0.0)
}

///|
test "np_matmul_identity" {
  // A @ I = A
  let a = mat_view([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3, 3)
  let eye3 = mat_eye(3)
  let result = mat_matmul(a, eye3)
  assert_true(mat_allclose(a, result, 0.0001))
}

///|
test "np_transpose_square" {
  // For symmetric matrix, A.T == A
  let sym = mat_view([1.0, 2.0, 2.0, 3.0], 2, 2)
  let sym_t = mat_transpose(sym)
  assert_true(mat_allclose(sym, sym_t, 0.0001))
}

///|
test "np_clip_all_inside" {
  // If all values inside range, clip should return same
  let v = vec_from_array([0.2, 0.5, 0.8])
  let clipped = vec_clip(v, 0.0, 1.0)
  assert_true(vec_allclose(v, clipped, 0.0001))
}

///|
test "np_exp_log_inverse" {
  // exp(log(x)) == x for x > 0
  let v = vec_from_array([0.5, 1.0, 2.0, 10.0])
  let result = vec_exp(vec_log(v))
  assert_true(vec_allclose(v, result, 0.0001))
}

///|
test "np_sqrt_square" {
  // sqrt(x^2) == |x| for x >= 0
  let v = vec_from_array([1.0, 4.0, 9.0, 16.0])
  let squared = vec_pow(v, 0.5)
  let expected = vec_from_array([1.0, 2.0, 3.0, 4.0])
  assert_true(vec_allclose(squared, expected, 0.0001))
}

///|
test "np_sum_axis_consistency" {
  // sum(axis=0).sum() == sum(axis=1).sum() == sum()
  let m = mat_view([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3)
  let sum_all = mat_sum(m)
  let sum_axis0 = vec_sum(mat_sum_axis0(m))
  let sum_axis1 = vec_sum(mat_sum_axis1(m))
  assert_eq(sum_all, sum_axis0)
  assert_eq(sum_all, sum_axis1)
}

// ============================================================================
// Sort tests
// ============================================================================

///|
test "vec_sort" {
  let v = vec_from_array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
  let sorted = vec_sort(v)
  assert_eq(vec_to_array(sorted), [1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0])
}

///|
test "vec_sort_desc" {
  let v = vec_from_array([3.0, 1.0, 4.0, 1.0, 5.0])
  let sorted = vec_sort_desc(v)
  assert_eq(vec_to_array(sorted), [5.0, 4.0, 3.0, 1.0, 1.0])
}

///|
test "vec_argsort" {
  let v = vec_from_array([3.0, 1.0, 4.0, 1.0, 5.0])
  let indices = vec_argsort(v)
  // Verify sorted values via indices: should be 1, 1, 3, 4, 5
  let sorted_vals = Array::makei(5, fn(i) { vec_at(v, indices[i]) })
  assert_eq(sorted_vals, [1.0, 1.0, 3.0, 4.0, 5.0])
}

///|
test "vec_argsort_desc" {
  let v = vec_from_array([3.0, 1.0, 4.0, 1.0, 5.0])
  let indices = vec_argsort_desc(v)
  // Verify sorted values via indices: should be 5, 4, 3, 1, 1
  let sorted_vals = Array::makei(5, fn(i) { vec_at(v, indices[i]) })
  assert_eq(sorted_vals, [5.0, 4.0, 3.0, 1.0, 1.0])
}

// ============================================================================
// Conditional operations tests
// ============================================================================

///|
test "vec_nonzero" {
  let v = vec_from_array([0.0, 1.0, 0.0, 2.0, 0.0, 3.0])
  let indices = vec_nonzero(v)
  assert_eq(indices, [1, 3, 5])
}

///|
test "vec_gt" {
  let v = vec_from_array([1.0, 2.0, 3.0, 4.0, 5.0])
  let mask = vec_gt(v, 3.0)
  assert_eq(vec_to_array(mask), [0.0, 0.0, 0.0, 1.0, 1.0])
}

///|
test "vec_lt" {
  let v = vec_from_array([1.0, 2.0, 3.0, 4.0, 5.0])
  let mask = vec_lt(v, 3.0)
  assert_eq(vec_to_array(mask), [1.0, 1.0, 0.0, 0.0, 0.0])
}

///|
test "vec_ge_le" {
  let v = vec_from_array([1.0, 2.0, 3.0, 4.0, 5.0])
  let ge_mask = vec_ge(v, 3.0)
  let le_mask = vec_le(v, 3.0)
  assert_eq(vec_to_array(ge_mask), [0.0, 0.0, 1.0, 1.0, 1.0])
  assert_eq(vec_to_array(le_mask), [1.0, 1.0, 1.0, 0.0, 0.0])
}

///|
test "vec_where" {
  let mask = vec_from_array([1.0, 0.0, 1.0, 0.0])
  let x = vec_from_array([10.0, 20.0, 30.0, 40.0])
  let y = vec_from_array([1.0, 2.0, 3.0, 4.0])
  let result = vec_where(mask, x, y)
  assert_eq(vec_to_array(result), [10.0, 2.0, 30.0, 4.0])
}

///|
test "vec_where_scalar" {
  let v = vec_from_array([-1.0, 2.0, -3.0, 4.0])
  let mask = vec_gt(v, 0.0)
  let result = vec_where_scalar(mask, v, 0.0)
  // ReLU-like behavior
  assert_eq(vec_to_array(result), [0.0, 2.0, 0.0, 4.0])
}

// ============================================================================
// Cumulative operations tests
// ============================================================================

///|
test "vec_cumsum" {
  let v = vec_from_array([1.0, 2.0, 3.0, 4.0])
  let cs = vec_cumsum(v)
  assert_eq(vec_to_array(cs), [1.0, 3.0, 6.0, 10.0])
}

///|
test "vec_cumprod" {
  let v = vec_from_array([1.0, 2.0, 3.0, 4.0])
  let cp = vec_cumprod(v)
  assert_eq(vec_to_array(cp), [1.0, 2.0, 6.0, 24.0])
}

///|
test "vec_diff" {
  let v = vec_from_array([1.0, 3.0, 6.0, 10.0])
  let d = vec_diff(v)
  assert_eq(vec_to_array(d), [2.0, 3.0, 4.0])
}

///|
test "vec_cumsum_cdf" {
  // Use case: CDF from probabilities
  let probs = vec_from_array([0.2, 0.3, 0.5])
  let cdf = vec_cumsum(probs)
  assert_near(vec_at(cdf, 0), 0.2, 0.0001)
  assert_near(vec_at(cdf, 1), 0.5, 0.0001)
  assert_near(vec_at(cdf, 2), 1.0, 0.0001)
}

// ============================================================================
// Linear algebra extras tests
// ============================================================================

///|
test "vec_outer" {
  let a = vec_from_array([1.0, 2.0])
  let b = vec_from_array([3.0, 4.0, 5.0])
  let outer = vec_outer(a, b)
  // [[1*3, 1*4, 1*5], [2*3, 2*4, 2*5]] = [[3,4,5], [6,8,10]]
  assert_eq(mat_rows(outer), 2)
  assert_eq(mat_cols(outer), 3)
  assert_eq(mat_to_array(outer), [3.0, 4.0, 5.0, 6.0, 8.0, 10.0])
}

///|
test "vec_diag" {
  let v = vec_from_array([1.0, 2.0, 3.0])
  let m = vec_diag(v)
  assert_eq(mat_rows(m), 3)
  assert_eq(mat_cols(m), 3)
  assert_eq(mat_to_array(m), [1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0])
}

///|
test "mat_diag" {
  let m = mat_view([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3, 3)
  let d = mat_diag(m)
  assert_eq(vec_to_array(d), [1.0, 5.0, 9.0])
}

///|
test "mat_trace" {
  let m = mat_view([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3, 3)
  assert_eq(mat_trace(m), 15.0) // 1 + 5 + 9
}

///|
test "diag_trace_consistency" {
  // trace(diag(v)) == sum(v)
  let v = vec_from_array([1.0, 2.0, 3.0, 4.0])
  let m = vec_diag(v)
  assert_eq(mat_trace(m), vec_sum(v))
}

// ============================================================================
// Concatenate / Stack tests
// ============================================================================

///|
test "vec_concatenate" {
  let a = vec_from_array([1.0, 2.0])
  let b = vec_from_array([3.0, 4.0, 5.0])
  let c = vec_from_array([6.0])
  let result = vec_concatenate([a, b, c])
  assert_eq(vec_to_array(result), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
}

///|
test "mat_vstack" {
  let a = mat_view([1.0, 2.0, 3.0, 4.0], 2, 2)
  let b = mat_view([5.0, 6.0], 1, 2)
  let result = mat_vstack([a, b])
  assert_eq(mat_rows(result), 3)
  assert_eq(mat_cols(result), 2)
  assert_eq(mat_to_array(result), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
}

///|
test "mat_hstack" {
  let a = mat_view([1.0, 2.0, 3.0, 4.0], 2, 2)
  let b = mat_view([5.0, 6.0], 2, 1)
  let result = mat_hstack([a, b])
  assert_eq(mat_rows(result), 2)
  assert_eq(mat_cols(result), 3)
  assert_eq(mat_to_array(result), [1.0, 2.0, 5.0, 3.0, 4.0, 6.0])
}

///|
test "vec_repeat" {
  let v = vec_from_array([1.0, 2.0, 3.0])
  let r = vec_repeat(v, 3)
  assert_eq(vec_to_array(r), [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
}

///|
test "mat_tile" {
  let m = mat_view([1.0, 2.0, 3.0, 4.0], 2, 2)
  let t = mat_tile(m, 2, 2)
  assert_eq(mat_rows(t), 4)
  assert_eq(mat_cols(t), 4)
  // First row: 1, 2, 1, 2
  assert_eq(mat_at(t, 0, 0), 1.0)
  assert_eq(mat_at(t, 0, 2), 1.0)
  // Third row: 1, 2, 1, 2
  assert_eq(mat_at(t, 2, 0), 1.0)
  assert_eq(mat_at(t, 2, 2), 1.0)
}

// ============================================================================
// Math functions tests
// ============================================================================

///|
test "vec_floor" {
  let v = vec_from_array([1.5, 2.7, -1.3, -2.9])
  let result = vec_floor(v)
  assert_eq(vec_to_array(result), [1.0, 2.0, -2.0, -3.0])
}

///|
test "vec_ceil" {
  let v = vec_from_array([1.5, 2.7, -1.3, -2.9])
  let result = vec_ceil(v)
  assert_eq(vec_to_array(result), [2.0, 3.0, -1.0, -2.0])
}

///|
test "vec_round" {
  let v = vec_from_array([1.4, 1.5, 1.6, -1.4, -1.5, -1.6])
  let result = vec_round(v)
  // Round half to even or away from zero depends on implementation
  assert_eq(vec_at(result, 0), 1.0)
  assert_eq(vec_at(result, 2), 2.0)
}

// ============================================================================
// Statistics tests
// ============================================================================

///|
test "vec_median_odd" {
  let v = vec_from_array([3.0, 1.0, 4.0, 1.0, 5.0])
  let m = vec_median(v)
  assert_eq(m, 3.0)
}

///|
test "vec_median_even" {
  let v = vec_from_array([3.0, 1.0, 4.0, 2.0])
  let m = vec_median(v)
  assert_eq(m, 2.5)
}

///|
test "vec_percentile" {
  let v = vec_from_array([1.0, 2.0, 3.0, 4.0, 5.0])
  assert_eq(vec_percentile(v, 0.0), 1.0)
  assert_eq(vec_percentile(v, 100.0), 5.0)
  assert_eq(vec_percentile(v, 50.0), 3.0)
}

///|
test "vec_quantile" {
  let v = vec_from_array([1.0, 2.0, 3.0, 4.0, 5.0])
  assert_eq(vec_quantile(v, 0.0), 1.0)
  assert_eq(vec_quantile(v, 1.0), 5.0)
  assert_eq(vec_quantile(v, 0.5), 3.0)
}

// ============================================================================
// Utility tests
// ============================================================================

///|
test "vec_unique" {
  let v = vec_from_array([3.0, 1.0, 2.0, 1.0, 3.0, 2.0])
  let u = vec_unique(v)
  assert_eq(vec_to_array(u), [1.0, 2.0, 3.0])
}

///|
test "vec_unique_empty" {
  let v = vec_from_array([])
  let u = vec_unique(v)
  assert_eq(vec_len(u), 0)
}

///|
test "vec_searchsorted" {
  let sorted = vec_from_array([1.0, 2.0, 3.0, 4.0, 5.0])
  assert_eq(vec_searchsorted(sorted, 0.0), 0)
  assert_eq(vec_searchsorted(sorted, 1.0), 0)
  assert_eq(vec_searchsorted(sorted, 2.5), 2)
  assert_eq(vec_searchsorted(sorted, 5.0), 4)
  assert_eq(vec_searchsorted(sorted, 6.0), 5)
}

///|
test "vec_searchsorted_right" {
  let sorted = vec_from_array([1.0, 2.0, 2.0, 3.0])
  assert_eq(vec_searchsorted(sorted, 2.0), 1)
  assert_eq(vec_searchsorted_right(sorted, 2.0), 3)
}

// ============================================================================
// Random tests
// ============================================================================

///|
test "vec_rand" {
  random_seed(42)
  let v = vec_rand(100)
  assert_eq(vec_len(v), 100)
  // All values should be in [0, 1)
  for i = 0; i < 100; i = i + 1 {
    let val = vec_at(v, i)
    assert_true(val >= 0.0 && val < 1.0)
  }
}

///|
test "vec_randn" {
  random_seed(42)
  let v = vec_randn(1000)
  assert_eq(vec_len(v), 1000)
  // Mean should be close to 0
  let m = vec_mean(v)
  assert_true(m.abs() < 0.2)
}

///|
test "mat_rand" {
  random_seed(42)
  let m = mat_rand(10, 10)
  assert_eq(mat_rows(m), 10)
  assert_eq(mat_cols(m), 10)
}

///|
test "vec_shuffle" {
  random_seed(42)
  let v = vec_from_array([1.0, 2.0, 3.0, 4.0, 5.0])
  let shuffled = vec_shuffle(v)
  // Same length
  assert_eq(vec_len(shuffled), 5)
  // Same sum (same elements, different order)
  assert_eq(vec_sum(shuffled), vec_sum(v))
}

// ============================================================================
// Linear algebra tests
// ============================================================================

///|
test "mat_inv_2x2" {
  // [[1, 2], [3, 4]] -> [[-2, 1], [1.5, -0.5]]
  let m = mat_view([1.0, 2.0, 3.0, 4.0], 2, 2)
  match mat_inv(m) {
    Some(inv) => {
      // Check A @ A^-1 = I
      let prod = mat_matmul(m, inv)
      assert_near(mat_at(prod, 0, 0), 1.0, 0.001)
      assert_near(mat_at(prod, 0, 1), 0.0, 0.001)
      assert_near(mat_at(prod, 1, 0), 0.0, 0.001)
      assert_near(mat_at(prod, 1, 1), 1.0, 0.001)
    }
    None => panic()
  }
}

///|
test "mat_inv_singular" {
  // Singular matrix: [[1, 2], [2, 4]]
  let m = mat_view([1.0, 2.0, 2.0, 4.0], 2, 2)
  match mat_inv(m) {
    Some(_) => panic() // Should fail
    None => () // Expected
  }
}

///|
test "mat_solve_2x2" {
  // Solve: [[2, 1], [1, 3]] @ x = [5, 10]
  // Solution: x = [1, 3]
  let a = mat_view([2.0, 1.0, 1.0, 3.0], 2, 2)
  let b = vec_from_array([5.0, 10.0])
  match mat_solve(a, b) {
    Some(x) => {
      assert_near(vec_at(x, 0), 1.0, 0.001)
      assert_near(vec_at(x, 1), 3.0, 0.001)
    }
    None => panic()
  }
}

// ============================================================================
// FMat LAPACK tests
// ============================================================================

///|
test "fmat_svd_2x2" {
  // SVD of [[3, 0], [0, 2]]
  // Singular values should be 3, 2
  let m = mat_view([3.0, 0.0, 0.0, 2.0], 2, 2)
  let fm = fmat_from_mat(m)
  match fmat_svd(fm) {
    Some((u, s, vt)) => {
      // Check singular values (should be sorted descending)
      let s_arr = vec_to_array(s)
      assert_near(s_arr[0], 3.0, 0.001)
      assert_near(s_arr[1], 2.0, 0.001)
      // Verify U @ diag(S) @ Vt ≈ A
      let u_mat = fmat_to_mat(u)
      let vt_mat = fmat_to_mat(vt)
      // U and Vt should be orthogonal (U @ U^T = I)
      assert_near(
        mat_at(u_mat, 0, 0) * mat_at(u_mat, 0, 0) +
        mat_at(u_mat, 0, 1) * mat_at(u_mat, 0, 1),
        1.0,
        0.001,
      )
      ignore(vt_mat)
    }
    None => panic()
  }
}

///|
test "fmat_eig_symmetric" {
  // Symmetric matrix [[2, 1], [1, 2]]
  // Eigenvalues: 3, 1
  let m = mat_view([2.0, 1.0, 1.0, 2.0], 2, 2)
  let fm = fmat_from_mat(m)
  match fmat_eig(fm) {
    Some((w, _v)) => {
      let w_arr = vec_to_array(w)
      // Eigenvalues are sorted ascending by LAPACK
      assert_near(w_arr[0], 1.0, 0.001)
      assert_near(w_arr[1], 3.0, 0.001)
    }
    None => panic()
  }
}

///|
test "fmat_cholesky" {
  // Positive definite matrix [[4, 2], [2, 2]]
  // Cholesky: L = [[2, 0], [1, 1]]
  let m = mat_view([4.0, 2.0, 2.0, 2.0], 2, 2)
  let fm = fmat_from_mat(m)
  match fmat_cholesky(fm) {
    Some(l) => {
      let l_mat = fmat_to_mat(l)
      assert_near(mat_at(l_mat, 0, 0), 2.0, 0.001)
      assert_near(mat_at(l_mat, 0, 1), 0.0, 0.001)
      assert_near(mat_at(l_mat, 1, 0), 1.0, 0.001)
      assert_near(mat_at(l_mat, 1, 1), 1.0, 0.001)
    }
    None => panic()
  }
}

///|
test "fmat_cholesky_not_positive_definite" {
  // Not positive definite: [[-1, 0], [0, 1]]
  let m = mat_view([-1.0, 0.0, 0.0, 1.0], 2, 2)
  let fm = fmat_from_mat(m)
  match fmat_cholesky(fm) {
    Some(_) => panic() // Should fail
    None => () // Expected
  }
}

///|
test "fmat_qr_2x3" {
  // QR decomposition of [[1, 2, 3], [4, 5, 6]]
  let m = mat_view([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3)
  let fm = fmat_from_mat(m)
  match fmat_qr(fm) {
    Some((q, r)) => {
      let q_mat = fmat_to_mat(q)
      let r_mat = fmat_to_mat(r)
      // Q should be orthogonal (Q @ Q^T = I)
      let qt = mat_transpose(q_mat)
      let qqt = mat_matmul(q_mat, qt)
      assert_near(mat_at(qqt, 0, 0), 1.0, 0.001)
      assert_near(mat_at(qqt, 0, 1), 0.0, 0.001)
      assert_near(mat_at(qqt, 1, 0), 0.0, 0.001)
      assert_near(mat_at(qqt, 1, 1), 1.0, 0.001)
      // R should be upper triangular
      assert_near(mat_at(r_mat, 1, 0), 0.0, 0.001)
    }
    None => panic()
  }
}

///|
test "fmat_det_2x2" {
  // det([[1, 2], [3, 4]]) = 1*4 - 2*3 = -2
  let m = mat_view([1.0, 2.0, 3.0, 4.0], 2, 2)
  let fm = fmat_from_mat(m)
  match fmat_det(fm) {
    Some(det) => assert_near(det, -2.0, 0.001)
    None => panic()
  }
}

///|
test "fmat_det_3x3" {
  // det([[1, 2, 3], [4, 5, 6], [7, 8, 10]]) = -3
  let m = mat_view([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0], 3, 3)
  let fm = fmat_from_mat(m)
  match fmat_det(fm) {
    Some(det) => assert_near(det, -3.0, 0.001)
    None => panic()
  }
}

///|
test "fmat_lstsq_overdetermined" {
  // Least squares: overdetermined system
  // [[1], [1], [1]] @ x ≈ [1, 2, 3]
  // Solution: x = [2] (mean)
  let a = mat_view([1.0, 1.0, 1.0], 3, 1)
  let b = vec_from_array([1.0, 2.0, 3.0])
  let fa = fmat_from_mat(a)
  match fmat_lstsq(fa, b) {
    Some(x) => assert_near(vec_at(x, 0), 2.0, 0.001)
    None => panic()
  }
}

///|
test "fmat_lstsq_exact" {
  // Exact solution: [[2, 1], [1, 3]] @ x = [5, 10]
  // Solution: x = [1, 3]
  let a = mat_view([2.0, 1.0, 1.0, 3.0], 2, 2)
  let b = vec_from_array([5.0, 10.0])
  let fa = fmat_from_mat(a)
  match fmat_lstsq(fa, b) {
    Some(x) => {
      assert_near(vec_at(x, 0), 1.0, 0.001)
      assert_near(vec_at(x, 1), 3.0, 0.001)
    }
    None => panic()
  }
}

// ============================================================================
// Boolean operations tests
// ============================================================================

///|
test "vec_any" {
  assert_eq(vec_any(vec_from_array([0.0, 0.0, 0.0])), false)
  assert_eq(vec_any(vec_from_array([0.0, 1.0, 0.0])), true)
  assert_eq(vec_any(vec_from_array([1.0, 2.0, 3.0])), true)
  assert_eq(vec_any(vec_zeros(5)), false)
}

///|
test "vec_all" {
  assert_eq(vec_all(vec_from_array([1.0, 2.0, 3.0])), true)
  assert_eq(vec_all(vec_from_array([1.0, 0.0, 3.0])), false)
  assert_eq(vec_all(vec_from_array([0.0, 0.0, 0.0])), false)
  assert_eq(vec_all(vec_ones(5)), true)
}

// ============================================================================
// Element-wise operations tests
// ============================================================================

///|
test "vec_sign" {
  let v = vec_from_array([-3.0, -0.5, 0.0, 0.5, 3.0])
  let result = vec_sign(v)
  assert_eq(vec_to_array(result), [-1.0, -1.0, 0.0, 1.0, 1.0])
}

///|
test "vec_maximum" {
  let a = vec_from_array([1.0, 5.0, 3.0, 7.0])
  let b = vec_from_array([2.0, 3.0, 4.0, 6.0])
  let result = vec_maximum(a, b)
  assert_eq(vec_to_array(result), [2.0, 5.0, 4.0, 7.0])
}

///|
test "vec_minimum" {
  let a = vec_from_array([1.0, 5.0, 3.0, 7.0])
  let b = vec_from_array([2.0, 3.0, 4.0, 6.0])
  let result = vec_minimum(a, b)
  assert_eq(vec_to_array(result), [1.0, 3.0, 3.0, 6.0])
}

///|
test "vec_maximum_minimum_negative" {
  let a = vec_from_array([-1.0, -5.0, 0.0])
  let b = vec_from_array([-2.0, -3.0, -1.0])
  assert_eq(vec_to_array(vec_maximum(a, b)), [-1.0, -3.0, 0.0])
  assert_eq(vec_to_array(vec_minimum(a, b)), [-2.0, -5.0, -1.0])
}
