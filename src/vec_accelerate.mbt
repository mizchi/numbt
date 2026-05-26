// Accelerate (vDSP) C FFI declarations.
//
// Used by the `fmat_*` family in vec.mbt (LapackMat / FixedArray[Byte] backed)
// for zero-copy SIMD element-wise ops and reductions.
//
// IMPORTANT: these are NOT exposed via `Array[Float]`-backed `Vec` / `Mat`.
// MoonBit's C FFI requires the buffer to be `FixedArray[Byte]`, so calling
// these from `Vec`/`Mat` would need an Array[Float] -> bytes -> Array[Float]
// round-trip in pure MoonBit. Empirically (see fmat_accelerate_bench.mbt and
// vec_accelerate_bench.mbt) that hop *erases* the SIMD win up to N=1M+.
// Stay on LapackMat for hot loops.

///|
#borrow(a, b, out)
extern "C" fn numbt_vadd(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit = "numbt_vadd"

///|
#borrow(a, b, out)
extern "C" fn numbt_vsub(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit = "numbt_vsub"

///|
#borrow(a, b, out)
extern "C" fn numbt_vmul(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit = "numbt_vmul"

///|
#borrow(a, b, out)
extern "C" fn numbt_vdiv(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  n : Int,
) -> Unit = "numbt_vdiv"

///|
#borrow(x, out)
extern "C" fn numbt_vsadd(
  x : FixedArray[Byte],
  scalar : Float,
  out : FixedArray[Byte],
  n : Int,
) -> Unit = "numbt_vsadd"

///|
#borrow(x, out)
extern "C" fn numbt_vsmul(
  x : FixedArray[Byte],
  scalar : Float,
  out : FixedArray[Byte],
  n : Int,
) -> Unit = "numbt_vsmul"

///|
#borrow(x, out_ptr)
extern "C" fn numbt_vsve(
  x : FixedArray[Byte],
  out_ptr : FixedArray[Byte],
  n : Int,
) -> Unit = "numbt_vsve"

///|
#borrow(x, out_ptr)
extern "C" fn numbt_vmeanv(
  x : FixedArray[Byte],
  out_ptr : FixedArray[Byte],
  n : Int,
) -> Unit = "numbt_vmeanv"

///|
#borrow(x, out_ptr)
extern "C" fn numbt_vmaxv(
  x : FixedArray[Byte],
  out_ptr : FixedArray[Byte],
  n : Int,
) -> Unit = "numbt_vmaxv"

///|
#borrow(x, out_ptr)
extern "C" fn numbt_vminv(
  x : FixedArray[Byte],
  out_ptr : FixedArray[Byte],
  n : Int,
) -> Unit = "numbt_vminv"
