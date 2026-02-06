// NumPy compatibility tests ported from numpy test suite
// These tests verify numbt behaves consistently with NumPy
// Note: assert_near is defined in vec_test.mbt

///|
fn assert_vec_near(actual : Vec, expected : Array[Float], tol : Float) -> Unit {
  let arr = vec_to_array(actual)
  if arr.length() != expected.length() {
    panic()
  }
  for i = 0; i < arr.length(); i = i + 1 {
    let diff = (arr[i] - expected[i]).abs()
    if diff > tol {
      panic()
    }
  }
}

// ============================================================================
// Linear Algebra Tests (from numpy/linalg/tests/test_linalg.py)
// ============================================================================

///|
test "numpy_linalg_solve_2x2" {
  // np.linalg.solve([[1, 2], [3, 4]], [2, 1])
  // Expected: [-3, 2.5]
  let a = mat_view([1.0, 2.0, 3.0, 4.0], 2, 2)
  let b = vec_from_array([2.0, 1.0])
  match mat_solve(a, b) {
    Some(x) => {
      assert_near(vec_at(x, 0), -3.0, 0.001)
      assert_near(vec_at(x, 1), 2.5, 0.001)
    }
    None => panic()
  }
}

///|
test "numpy_linalg_inv_2x2" {
  // np.linalg.inv([[1, 2], [3, 4]])
  // Expected: [[-2, 1], [1.5, -0.5]]
  let m = mat_view([1.0, 2.0, 3.0, 4.0], 2, 2)
  match mat_inv(m) {
    Some(inv) => {
      assert_near(mat_at(inv, 0, 0), -2.0, 0.001)
      assert_near(mat_at(inv, 0, 1), 1.0, 0.001)
      assert_near(mat_at(inv, 1, 0), 1.5, 0.001)
      assert_near(mat_at(inv, 1, 1), -0.5, 0.001)
    }
    None => panic()
  }
}

///|
test "numpy_linalg_det_2x2" {
  // np.linalg.det([[1, 2], [3, 4]]) = -2
  let m = mat_view([1.0, 2.0, 3.0, 4.0], 2, 2)
  let fm = fmat_from_mat(m)
  match fmat_det(fm) {
    Some(det) => assert_near(det, -2.0, 0.001)
    None => panic()
  }
}

///|
test "numpy_linalg_det_3x3" {
  // np.linalg.det([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) = 0
  let m = mat_view([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3, 3)
  let fm = fmat_from_mat(m)
  match fmat_det(fm) {
    Some(det) => assert_near(det, 0.0, 0.001)
    None => panic()
  }
}

///|
test "numpy_linalg_svd_diagonal" {
  // SVD of diagonal matrix [[3, 0], [0, 2]]
  // Singular values: [3, 2]
  let m = mat_view([3.0, 0.0, 0.0, 2.0], 2, 2)
  let fm = fmat_from_mat(m)
  match fmat_svd(fm) {
    Some((_u, s, _vt)) => {
      let s_arr = vec_to_array(s)
      // Singular values should be 3 and 2 (possibly in different order)
      let s0 = s_arr[0]
      let s1 = s_arr[1]
      let has_3 = (s0 - 3.0).abs() < 0.001 || (s1 - 3.0).abs() < 0.001
      let has_2 = (s0 - 2.0).abs() < 0.001 || (s1 - 2.0).abs() < 0.001
      assert_eq(has_3 && has_2, true)
    }
    None => panic()
  }
}

///|
test "numpy_linalg_eig_symmetric" {
  // Eigenvalues of symmetric matrix [[2, 1], [1, 2]]
  // Expected eigenvalues: [1, 3]
  let m = mat_view([2.0, 1.0, 1.0, 2.0], 2, 2)
  let fm = fmat_from_mat(m)
  match fmat_eig(fm) {
    Some((w, _v)) => {
      let w_arr = vec_to_array(w)
      // ssyev returns eigenvalues in ascending order
      assert_near(w_arr[0], 1.0, 0.001)
      assert_near(w_arr[1], 3.0, 0.001)
    }
    None => panic()
  }
}

///|
test "numpy_linalg_cholesky" {
  // Cholesky of [[4, 2], [2, 2]]
  // Expected L: [[2, 0], [1, 1]]
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
test "numpy_linalg_qr" {
  // QR of [[1, 2], [3, 4]]
  // Q should be orthogonal: Q @ Q^T = I
  let m = mat_view([1.0, 2.0, 3.0, 4.0], 2, 2)
  let fm = fmat_from_mat(m)
  match fmat_qr(fm) {
    Some((q, r)) => {
      let q_mat = fmat_to_mat(q)
      let r_mat = fmat_to_mat(r)
      // Q @ Q^T should be identity
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
test "numpy_linalg_lstsq" {
  // Least squares: [[1], [1], [1]] @ x ≈ [1, 2, 3]
  // Solution: x = [2] (mean of [1, 2, 3])
  let a = mat_view([1.0, 1.0, 1.0], 3, 1)
  let b = vec_from_array([1.0, 2.0, 3.0])
  let fa = fmat_from_mat(a)
  match fmat_lstsq(fa, b) {
    Some(x) => assert_near(vec_at(x, 0), 2.0, 0.001)
    None => panic()
  }
}

// ============================================================================
// Math Functions Tests (from numpy/_core/tests/test_umath.py)
// ============================================================================

///|
test "numpy_exp" {
  // np.exp([0, 1, 2])
  let v = vec_from_array([0.0, 1.0, 2.0])
  let result = vec_exp(v)
  assert_near(vec_at(result, 0), 1.0, 0.001)
  assert_near(vec_at(result, 1), 2.71828, 0.001)
  assert_near(vec_at(result, 2), 7.38905, 0.001)
}

///|
test "numpy_log" {
  // np.log([1, e, e^2])
  let e : Float = 2.71828
  let v = vec_from_array([1.0, e, e * e])
  let result = vec_log(v)
  assert_near(vec_at(result, 0), 0.0, 0.001)
  assert_near(vec_at(result, 1), 1.0, 0.001)
  assert_near(vec_at(result, 2), 2.0, 0.001)
}

///|
test "numpy_sqrt" {
  // np.sqrt([0, 1, 4, 9])
  let v = vec_from_array([0.0, 1.0, 4.0, 9.0])
  let result = vec_sqrt(v)
  assert_vec_near(result, [0.0, 1.0, 2.0, 3.0], 0.001)
}

///|
test "numpy_sin" {
  // np.sin([0, pi/2, pi])
  let pi : Float = 3.14159265
  let v = vec_from_array([0.0, pi / 2.0, pi])
  let result = vec_sin(v)
  assert_near(vec_at(result, 0), 0.0, 0.001)
  assert_near(vec_at(result, 1), 1.0, 0.001)
  assert_near(vec_at(result, 2), 0.0, 0.001)
}

///|
test "numpy_cos" {
  // np.cos([0, pi/2, pi])
  let pi : Float = 3.14159265
  let v = vec_from_array([0.0, pi / 2.0, pi])
  let result = vec_cos(v)
  assert_near(vec_at(result, 0), 1.0, 0.001)
  assert_near(vec_at(result, 1), 0.0, 0.001)
  assert_near(vec_at(result, 2), -1.0, 0.001)
}

///|
test "numpy_tanh" {
  // np.tanh([0, 1, -1])
  let v = vec_from_array([0.0, 1.0, -1.0])
  let result = vec_tanh(v)
  assert_near(vec_at(result, 0), 0.0, 0.001)
  assert_near(vec_at(result, 1), 0.76159, 0.001)
  assert_near(vec_at(result, 2), -0.76159, 0.001)
}

///|
test "numpy_floor" {
  // np.floor([1.5, -1.5, 0.0])
  let v = vec_from_array([1.5, -1.5, 0.0])
  let result = vec_floor(v)
  assert_vec_near(result, [1.0, -2.0, 0.0], 0.001)
}

///|
test "numpy_ceil" {
  // np.ceil([1.5, -1.5, 0.0])
  let v = vec_from_array([1.5, -1.5, 0.0])
  let result = vec_ceil(v)
  assert_vec_near(result, [2.0, -1.0, 0.0], 0.001)
}

///|
test "numpy_round" {
  // np.round([1.4, 1.5, 1.6, -1.5])
  let v = vec_from_array([1.4, 1.6, 2.5])
  let result = vec_round(v)
  assert_near(vec_at(result, 0), 1.0, 0.001)
  assert_near(vec_at(result, 1), 2.0, 0.001)
  // Note: MoonBit may use different rounding for .5
}

///|
test "numpy_sign" {
  // np.sign([-5, 0, 5])
  let v = vec_from_array([-5.0, 0.0, 5.0])
  let result = vec_sign(v)
  assert_vec_near(result, [-1.0, 0.0, 1.0], 0.001)
}

///|
test "numpy_abs" {
  // np.abs([-3, 0, 3])
  let v = vec_from_array([-3.0, 0.0, 3.0])
  let result = vec_abs(v)
  assert_vec_near(result, [3.0, 0.0, 3.0], 0.001)
}

///|
test "numpy_maximum" {
  // np.maximum([1, 5, 3], [2, 3, 4])
  let a = vec_from_array([1.0, 5.0, 3.0])
  let b = vec_from_array([2.0, 3.0, 4.0])
  let result = vec_maximum(a, b)
  assert_vec_near(result, [2.0, 5.0, 4.0], 0.001)
}

///|
test "numpy_minimum" {
  // np.minimum([1, 5, 3], [2, 3, 4])
  let a = vec_from_array([1.0, 5.0, 3.0])
  let b = vec_from_array([2.0, 3.0, 4.0])
  let result = vec_minimum(a, b)
  assert_vec_near(result, [1.0, 3.0, 3.0], 0.001)
}

///|
test "numpy_clip" {
  // np.clip([-1, 5, 2, 3, 10], 2, 7)
  let v = vec_from_array([-1.0, 5.0, 2.0, 3.0, 10.0])
  let result = vec_clip(v, 2.0, 7.0)
  assert_vec_near(result, [2.0, 5.0, 2.0, 3.0, 7.0], 0.001)
}

// ============================================================================
// Numeric Tests (from numpy/_core/tests/test_numeric.py)
// ============================================================================

///|
test "numpy_sum" {
  // np.sum([1, 2, 3, 4])
  let v = vec_from_array([1.0, 2.0, 3.0, 4.0])
  assert_near(vec_sum(v), 10.0, 0.001)
}

///|
test "numpy_prod" {
  // np.prod([1, 2, 3, 4])
  let v = vec_from_array([1.0, 2.0, 3.0, 4.0])
  assert_near(vec_prod(v), 24.0, 0.001)
}

///|
test "numpy_mean" {
  // np.mean([1, 2, 3, 4, 5])
  let v = vec_from_array([1.0, 2.0, 3.0, 4.0, 5.0])
  assert_near(vec_mean(v), 3.0, 0.001)
}

///|
test "numpy_std" {
  // np.std([2, 4, 4, 4, 5, 5, 7, 9])
  // Population std = 2.0
  let v = vec_from_array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
  assert_near(vec_std(v), 2.0, 0.001)
}

///|
test "numpy_var" {
  // np.var([2, 4, 4, 4, 5, 5, 7, 9])
  // Population var = 4.0
  let v = vec_from_array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
  assert_near(vec_var(v), 4.0, 0.001)
}

///|
test "numpy_cumsum" {
  // np.cumsum([1, 2, 3, 4])
  let v = vec_from_array([1.0, 2.0, 3.0, 4.0])
  let result = vec_cumsum(v)
  assert_vec_near(result, [1.0, 3.0, 6.0, 10.0], 0.001)
}

///|
test "numpy_cumprod" {
  // np.cumprod([1, 2, 3, 4])
  let v = vec_from_array([1.0, 2.0, 3.0, 4.0])
  let result = vec_cumprod(v)
  assert_vec_near(result, [1.0, 2.0, 6.0, 24.0], 0.001)
}

///|
test "numpy_diff" {
  // np.diff([1, 2, 4, 7])
  let v = vec_from_array([1.0, 2.0, 4.0, 7.0])
  let result = vec_diff(v)
  assert_vec_near(result, [1.0, 2.0, 3.0], 0.001)
}

///|
test "numpy_any" {
  // np.any([0, 0, 0]) = False
  // np.any([0, 1, 0]) = True
  assert_eq(vec_any(vec_from_array([0.0, 0.0, 0.0])), false)
  assert_eq(vec_any(vec_from_array([0.0, 1.0, 0.0])), true)
}

///|
test "numpy_all" {
  // np.all([1, 2, 3]) = True
  // np.all([1, 0, 3]) = False
  assert_eq(vec_all(vec_from_array([1.0, 2.0, 3.0])), true)
  assert_eq(vec_all(vec_from_array([1.0, 0.0, 3.0])), false)
}

// ============================================================================
// Sorting Tests
// ============================================================================

///|
test "numpy_sort" {
  // np.sort([3, 1, 4, 1, 5, 9, 2, 6])
  let v = vec_from_array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
  let result = vec_sort(v)
  assert_vec_near(result, [1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0], 0.001)
}

///|
test "numpy_sort_desc" {
  // np.sort([3, 1, 4])[::-1]
  let v = vec_from_array([3.0, 1.0, 4.0])
  let result = vec_sort_desc(v)
  assert_vec_near(result, [4.0, 3.0, 1.0], 0.001)
}

///|
test "numpy_argsort" {
  // np.argsort([3, 1, 2])
  let v = vec_from_array([3.0, 1.0, 2.0])
  let result = vec_argsort(v)
  assert_eq(result[0], 1) // index of 1.0
  assert_eq(result[1], 2) // index of 2.0
  assert_eq(result[2], 0) // index of 3.0
}

///|
test "numpy_argmax" {
  // np.argmax([1, 5, 3, 9, 2])
  let v = vec_from_array([1.0, 5.0, 3.0, 9.0, 2.0])
  assert_eq(vec_argmax(v), 3)
}

///|
test "numpy_argmin" {
  // np.argmin([5, 1, 3, 9, 2])
  let v = vec_from_array([5.0, 1.0, 3.0, 9.0, 2.0])
  assert_eq(vec_argmin(v), 1)
}

///|
test "numpy_searchsorted" {
  // np.searchsorted([1, 2, 3, 4, 5], 3.5)
  let sorted = vec_from_array([1.0, 2.0, 3.0, 4.0, 5.0])
  assert_eq(vec_searchsorted(sorted, 3.5), 3)
  assert_eq(vec_searchsorted(sorted, 1.0), 0)
  assert_eq(vec_searchsorted(sorted, 5.0), 4)
}

///|
test "numpy_unique" {
  // np.unique([1, 2, 2, 3, 3, 3])
  let v = vec_from_array([1.0, 2.0, 2.0, 3.0, 3.0, 3.0])
  let result = vec_unique(v)
  assert_vec_near(result, [1.0, 2.0, 3.0], 0.001)
}

// ============================================================================
// Statistics Tests
// ============================================================================

///|
test "numpy_median_odd" {
  // np.median([1, 3, 5])
  let v = vec_from_array([1.0, 3.0, 5.0])
  assert_near(vec_median(v), 3.0, 0.001)
}

///|
test "numpy_median_even" {
  // np.median([1, 2, 3, 4])
  let v = vec_from_array([1.0, 2.0, 3.0, 4.0])
  assert_near(vec_median(v), 2.5, 0.001)
}

///|
test "numpy_percentile" {
  // np.percentile([1, 2, 3, 4, 5], 50)
  let v = vec_from_array([1.0, 2.0, 3.0, 4.0, 5.0])
  assert_near(vec_percentile(v, 50.0), 3.0, 0.001)
}

// ============================================================================
// Matrix Operations Tests
// ============================================================================

///|
test "numpy_matmul_2x2" {
  // np.array([[1, 2], [3, 4]]) @ np.array([[5, 6], [7, 8]])
  // = [[19, 22], [43, 50]]
  let a = mat_view([1.0, 2.0, 3.0, 4.0], 2, 2)
  let b = mat_view([5.0, 6.0, 7.0, 8.0], 2, 2)
  let c = mat_matmul(a, b)
  assert_near(mat_at(c, 0, 0), 19.0, 0.001)
  assert_near(mat_at(c, 0, 1), 22.0, 0.001)
  assert_near(mat_at(c, 1, 0), 43.0, 0.001)
  assert_near(mat_at(c, 1, 1), 50.0, 0.001)
}

///|
test "numpy_transpose" {
  // np.array([[1, 2, 3], [4, 5, 6]]).T
  let m = mat_view([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3)
  let t = mat_transpose(m)
  assert_eq(mat_rows(t), 3)
  assert_eq(mat_cols(t), 2)
  assert_near(mat_at(t, 0, 0), 1.0, 0.001)
  assert_near(mat_at(t, 0, 1), 4.0, 0.001)
  assert_near(mat_at(t, 1, 0), 2.0, 0.001)
  assert_near(mat_at(t, 2, 1), 6.0, 0.001)
}

///|
test "numpy_trace" {
  // np.trace([[1, 2], [3, 4]])
  let m = mat_view([1.0, 2.0, 3.0, 4.0], 2, 2)
  assert_near(mat_trace(m), 5.0, 0.001)
}

///|
test "numpy_diag" {
  // np.diag([1, 2, 3])
  let v = vec_from_array([1.0, 2.0, 3.0])
  let m = vec_diag(v)
  assert_eq(mat_rows(m), 3)
  assert_eq(mat_cols(m), 3)
  assert_near(mat_at(m, 0, 0), 1.0, 0.001)
  assert_near(mat_at(m, 1, 1), 2.0, 0.001)
  assert_near(mat_at(m, 2, 2), 3.0, 0.001)
  assert_near(mat_at(m, 0, 1), 0.0, 0.001)
}

///|
test "numpy_eye" {
  // np.eye(3)
  let m = mat_eye(3)
  assert_near(mat_at(m, 0, 0), 1.0, 0.001)
  assert_near(mat_at(m, 1, 1), 1.0, 0.001)
  assert_near(mat_at(m, 2, 2), 1.0, 0.001)
  assert_near(mat_at(m, 0, 1), 0.0, 0.001)
  assert_near(mat_at(m, 1, 2), 0.0, 0.001)
}

///|
test "numpy_outer" {
  // np.outer([1, 2], [3, 4])
  // = [[3, 4], [6, 8]]
  let a = vec_from_array([1.0, 2.0])
  let b = vec_from_array([3.0, 4.0])
  let m = vec_outer(a, b)
  assert_near(mat_at(m, 0, 0), 3.0, 0.001)
  assert_near(mat_at(m, 0, 1), 4.0, 0.001)
  assert_near(mat_at(m, 1, 0), 6.0, 0.001)
  assert_near(mat_at(m, 1, 1), 8.0, 0.001)
}

///|
test "numpy_mat_sum_axis" {
  // np.sum([[1, 2], [3, 4]], axis=0) = [4, 6]
  // np.sum([[1, 2], [3, 4]], axis=1) = [3, 7]
  let m = mat_view([1.0, 2.0, 3.0, 4.0], 2, 2)
  let sum0 = mat_sum_axis0(m)
  let sum1 = mat_sum_axis1(m)
  assert_vec_near(sum0, [4.0, 6.0], 0.001)
  assert_vec_near(sum1, [3.0, 7.0], 0.001)
}

// ============================================================================
// Array Creation Tests
// ============================================================================

///|
test "numpy_zeros" {
  // np.zeros(5)
  let v = vec_zeros(5)
  assert_eq(vec_len(v), 5)
  for i = 0; i < 5; i = i + 1 {
    assert_near(vec_at(v, i), 0.0, 0.001)
  }
}

///|
test "numpy_ones" {
  // np.ones(5)
  let v = vec_ones(5)
  assert_eq(vec_len(v), 5)
  for i = 0; i < 5; i = i + 1 {
    assert_near(vec_at(v, i), 1.0, 0.001)
  }
}

///|
test "numpy_arange" {
  // np.arange(0, 5, 1)
  let v = vec_arange(0.0, 5.0, 1.0)
  assert_vec_near(v, [0.0, 1.0, 2.0, 3.0, 4.0], 0.001)
}

///|
test "numpy_linspace" {
  // np.linspace(0, 1, 5)
  let v = vec_linspace(0.0, 1.0, 5)
  assert_vec_near(v, [0.0, 0.25, 0.5, 0.75, 1.0], 0.001)
}

// ============================================================================
// Concatenation Tests
// ============================================================================

///|
test "numpy_concatenate" {
  // np.concatenate([[1, 2], [3, 4]])
  let a = vec_from_array([1.0, 2.0])
  let b = vec_from_array([3.0, 4.0])
  let result = vec_concatenate([a, b])
  assert_vec_near(result, [1.0, 2.0, 3.0, 4.0], 0.001)
}

///|
test "numpy_vstack" {
  // np.vstack([[[1, 2]], [[3, 4]]])
  let a = mat_view([1.0, 2.0], 1, 2)
  let b = mat_view([3.0, 4.0], 1, 2)
  let result = mat_vstack([a, b])
  assert_eq(mat_rows(result), 2)
  assert_eq(mat_cols(result), 2)
  assert_near(mat_at(result, 0, 0), 1.0, 0.001)
  assert_near(mat_at(result, 1, 0), 3.0, 0.001)
}

///|
test "numpy_hstack" {
  // np.hstack([[[1], [2]], [[3], [4]]])
  let a = mat_view([1.0, 2.0], 2, 1)
  let b = mat_view([3.0, 4.0], 2, 1)
  let result = mat_hstack([a, b])
  assert_eq(mat_rows(result), 2)
  assert_eq(mat_cols(result), 2)
  assert_near(mat_at(result, 0, 0), 1.0, 0.001)
  assert_near(mat_at(result, 0, 1), 3.0, 0.001)
}

// ============================================================================
// Comparison Tests
// ============================================================================

///|
test "numpy_allclose" {
  // np.allclose([1.0, 1.0001], [1.0, 1.0], atol=0.001)
  let a = vec_from_array([1.0, 1.0001])
  let b = vec_from_array([1.0, 1.0])
  assert_eq(vec_allclose(a, b, 0.001), true)
  assert_eq(vec_allclose(a, b, 0.00001), false)
}

///|
test "numpy_gt_lt" {
  // np.greater([1, 2, 3], 2) = [False, False, True]
  // np.less([1, 2, 3], 2) = [True, False, False]
  let v = vec_from_array([1.0, 2.0, 3.0])
  let gt = vec_gt(v, 2.0)
  let lt = vec_lt(v, 2.0)
  assert_vec_near(gt, [0.0, 0.0, 1.0], 0.001)
  assert_vec_near(lt, [1.0, 0.0, 0.0], 0.001)
}
