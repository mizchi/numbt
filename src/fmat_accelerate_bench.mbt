///|
/// fmat (LapackMat, FixedArray[Byte] backed) element-wise ops:
/// scalar manual byte decode (the old fmat_add implementation) vs
/// vDSP zero-copy (the new implementation).
///
/// This is the apples-to-apples comparison: same storage, different op.
/// The vec_accelerate_bench Array[Float] case loses to scalar because of
/// the FFI byte-encoding hop; here both paths skip that hop.

///|
fn make_fmat(n : Int, seed : Int) -> LapackMat {
  let arr : Array[Float] = Array::make(n * n, Float::from_int(0))
  let mut x = seed
  for i = 0; i < n * n; i = i + 1 {
    x = x * 1103515245 + 12345
    arr[i] = Float::from_int((x >> 16) & 0x7fff) / 32768.0
  }
  fmat_from_mat(mat_view(arr, n, n))
}

///|
/// The pre-Accelerate fmat_add: inline byte decode + manual loop on bytes.
/// Kept for reference / parity check.
fn fmat_add_scalar_baseline(a : LapackMat, b : LapackMat) -> LapackMat {
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
fn fmat_sum_scalar_baseline(a : LapackMat) -> Float {
  let mut sum = Float::from_int(0)
  let n = a.rows * a.cols
  for i = 0; i < n; i = i + 1 {
    let idx = i * 4
    let bits = a.data[idx].to_int() |
      (a.data[idx + 1].to_int() << 8) |
      (a.data[idx + 2].to_int() << 16) |
      (a.data[idx + 3].to_int() << 24)
    sum = sum + Float::reinterpret_from_int(bits)
  }
  sum
}

///|
test "bench: fmat_add scalar (byte-decode) vs vDSP, 32x32 / 128x128 / 512x512" (
  t : @bench.T,
) {
  let sizes = [32, 128, 512]
  for s = 0; s < sizes.length(); s = s + 1 {
    let dim = sizes[s]
    let fa = make_fmat(dim, 1)
    let fb = make_fmat(dim, 2)
    t.bench(name="fmat_add scalar   \{dim}x\{dim}", fn() {
      let _ = fmat_add_scalar_baseline(fa, fb)
    })
    t.bench(name="fmat_add vDSP     \{dim}x\{dim}", fn() {
      let _ = fmat_add(fa, fb)
    })
  }
}

///|
test "bench: fmat_sum scalar (byte-decode) vs vDSP, 32x32 / 128x128 / 512x512" (
  t : @bench.T,
) {
  let sizes = [32, 128, 512]
  for s = 0; s < sizes.length(); s = s + 1 {
    let dim = sizes[s]
    let fa = make_fmat(dim, 1)
    t.bench(name="fmat_sum scalar   \{dim}x\{dim}", fn() {
      let _ = fmat_sum_scalar_baseline(fa)
    })
    t.bench(name="fmat_sum vDSP     \{dim}x\{dim}", fn() {
      let _ = fmat_sum(fa)
    })
  }
}

///|
/// In-place variants: same as above but reuse a pre-allocated output buffer.
/// This isolates pure compute time (no fmat_zeros allocation per call).
test "bench: fmat_add_into in-place vDSP, 32x32 / 128x128 / 512x512" (
  t : @bench.T,
) {
  let sizes = [32, 128, 512]
  for s = 0; s < sizes.length(); s = s + 1 {
    let dim = sizes[s]
    let fa = make_fmat(dim, 1)
    let fb = make_fmat(dim, 2)
    let out = fmat_zeros(dim, dim)
    t.bench(name="fmat_add_into vDSP \{dim}x\{dim}", fn() {
      fmat_add_into(fa, fb, out~)
    })
  }
}
