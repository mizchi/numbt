///|
test "fmat_sub" {
  let a = mat_view([10.0, 20.0, 30.0, 40.0], 2, 2)
  let b = mat_view([1.0, 2.0, 3.0, 4.0], 2, 2)
  let fa = fmat_from_mat(a)
  let fb = fmat_from_mat(b)
  let fc = fmat_sub(fa, fb)
  let c = fmat_to_mat(fc)
  assert_eq(mat_to_array(c), [9.0, 18.0, 27.0, 36.0])
}

///|
test "fmat_mul (element-wise)" {
  let a = mat_view([1.0, 2.0, 3.0, 4.0], 2, 2)
  let b = mat_view([5.0, 6.0, 7.0, 8.0], 2, 2)
  let fa = fmat_from_mat(a)
  let fb = fmat_from_mat(b)
  let fc = fmat_mul(fa, fb)
  let c = fmat_to_mat(fc)
  assert_eq(mat_to_array(c), [5.0, 12.0, 21.0, 32.0])
}

///|
test "fmat_div" {
  let a = mat_view([10.0, 20.0, 30.0, 40.0], 2, 2)
  let b = mat_view([2.0, 4.0, 5.0, 8.0], 2, 2)
  let fa = fmat_from_mat(a)
  let fb = fmat_from_mat(b)
  let fc = fmat_div(fa, fb)
  let c = fmat_to_mat(fc)
  assert_eq(mat_to_array(c), [5.0, 5.0, 6.0, 5.0])
}

///|
test "fmat_add_scalar" {
  let a = mat_view([1.0, 2.0, 3.0, 4.0], 2, 2)
  let fa = fmat_from_mat(a)
  let fc = fmat_add_scalar(fa, 10.0)
  let c = fmat_to_mat(fc)
  assert_eq(mat_to_array(c), [11.0, 12.0, 13.0, 14.0])
}

///|
test "fmat_mul_scalar" {
  let a = mat_view([1.0, 2.0, 3.0, 4.0], 2, 2)
  let fa = fmat_from_mat(a)
  let fc = fmat_mul_scalar(fa, 3.0)
  let c = fmat_to_mat(fc)
  assert_eq(mat_to_array(c), [3.0, 6.0, 9.0, 12.0])
}

///|
test "fmat_sum / mean / max / min" {
  let a = mat_view([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3)
  let fa = fmat_from_mat(a)
  assert_near(fmat_sum(fa), 21.0, 1.0e-5)
  assert_near(fmat_mean(fa), 3.5, 1.0e-5)
  assert_near(fmat_max(fa), 6.0, 1.0e-6)
  assert_near(fmat_min(fa), 1.0, 1.0e-6)
}

///|
test "fmat_add_into in-place" {
  let a = mat_view([1.0, 2.0, 3.0, 4.0], 2, 2)
  let b = mat_view([10.0, 20.0, 30.0, 40.0], 2, 2)
  let fa = fmat_from_mat(a)
  let fb = fmat_from_mat(b)
  let out = fmat_zeros(2, 2)
  fmat_add_into(fa, fb, out~)
  let c = fmat_to_mat(out)
  assert_eq(mat_to_array(c), [11.0, 22.0, 33.0, 44.0])
}

///|
test "fmat_add round-trip parity with mat_add" {
  let a = mat_view([1.5, 2.5, 3.5, 4.5, 5.5, 6.5], 2, 3)
  let b = mat_view([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], 2, 3)
  let scalar_result = mat_add(a, b)
  let accel_result = fmat_to_mat(fmat_add(fmat_from_mat(a), fmat_from_mat(b)))
  for i = 0; i < 6; i = i + 1 {
    assert_near(accel_result.data[i], scalar_result.data[i], 1.0e-5)
  }
}
