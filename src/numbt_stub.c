// numbt native acceleration using Apple Accelerate framework
// Provides optimized sort, exp, outer product, LAPACK, and random operations

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#ifdef __APPLE__
  #define ACCELERATE_NEW_LAPACK
  #include <Accelerate/Accelerate.h>
#endif

typedef uint8_t *moonbit_bytes_t;

static int random_initialized = 0;

static void ensure_random_init(void) {
  if (!random_initialized) {
    srand((unsigned int)time(NULL));
    random_initialized = 1;
  }
}

// ============================================================================
// Optimized sort using vDSP
// ============================================================================

void numbt_vsort_asc(moonbit_bytes_t data, int n) {
#ifdef __APPLE__
  vDSP_vsort((float*)data, (vDSP_Length)n, 1);  // 1 = ascending
#endif
}

void numbt_vsort_desc(moonbit_bytes_t data, int n) {
#ifdef __APPLE__
  vDSP_vsort((float*)data, (vDSP_Length)n, -1);  // -1 = descending
#endif
}

// ============================================================================
// Optimized exp using vForce
// ============================================================================

void numbt_vexp(moonbit_bytes_t src, moonbit_bytes_t dst, int n) {
#ifdef __APPLE__
  vvexpf((float*)dst, (const float*)src, &n);
#endif
}

// ============================================================================
// Outer product using BLAS sger
// ============================================================================

void numbt_outer(moonbit_bytes_t a, moonbit_bytes_t b, moonbit_bytes_t out, int m, int n) {
#ifdef __APPLE__
  // out = a * b^T (rank-1 update)
  // Initialize out to zero first
  memset(out, 0, m * n * sizeof(float));
  // sger: A := alpha * x * y^T + A
  cblas_sger(CblasRowMajor, m, n, 1.0f, (float*)a, 1, (float*)b, 1, (float*)out, n);
#endif
}

// ============================================================================
// FMat BLAS operations (zero-copy)
// ============================================================================

void numbt_sgemm(moonbit_bytes_t a, moonbit_bytes_t b, moonbit_bytes_t c, int m, int n, int k) {
#ifdef __APPLE__
  // C = A @ B, where A is m x k, B is k x n, C is m x n (row-major)
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    m, n, k, 1.0f, (float*)a, k, (float*)b, n, 0.0f, (float*)c, n);
#endif
}

// ============================================================================
// Random number generation
// ============================================================================

void numbt_seed(int seed) {
  srand((unsigned int)seed);
  random_initialized = 1;
}

// Fill with uniform random [0, 1)
void numbt_rand(moonbit_bytes_t out, int n) {
  ensure_random_init();
  float* fout = (float*)out;
  for (int i = 0; i < n; i++) {
    fout[i] = (float)rand() / ((float)RAND_MAX + 1.0f);
  }
}

// Fill with standard normal (Box-Muller transform)
void numbt_randn(moonbit_bytes_t out, int n) {
  ensure_random_init();
  float* fout = (float*)out;
  for (int i = 0; i < n; i += 2) {
    float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
    float u2 = (float)rand() / ((float)RAND_MAX + 1.0f);
    float r = sqrtf(-2.0f * logf(u1));
    float theta = 2.0f * 3.14159265358979f * u2;
    fout[i] = r * cosf(theta);
    if (i + 1 < n) {
      fout[i + 1] = r * sinf(theta);
    }
  }
}

// Fisher-Yates shuffle for indices
void numbt_shuffle_indices(int* indices, int n) {
  ensure_random_init();
  for (int i = n - 1; i > 0; i--) {
    int j = rand() % (i + 1);
    int tmp = indices[i];
    indices[i] = indices[j];
    indices[j] = tmp;
  }
}

// ============================================================================
// LAPACK: Matrix inverse and linear solve
// ============================================================================

// Matrix inverse: A^-1
// Returns 0 on success, non-zero on failure (singular matrix)
int numbt_inv(moonbit_bytes_t a, int n) {
#ifdef __APPLE__
  float* A = (float*)a;
  int* ipiv = (int*)malloc(n * sizeof(int));
  if (!ipiv) return -1;

  int info;
  int N = n;
  int lda = n;

  // LU factorization
  sgetrf_(&N, &N, A, &lda, ipiv, &info);
  if (info != 0) {
    free(ipiv);
    return info;
  }

  // Compute inverse from LU factorization
  int lwork = n * n;
  float* work = (float*)malloc(lwork * sizeof(float));
  if (!work) {
    free(ipiv);
    return -1;
  }

  sgetri_(&N, A, &lda, ipiv, work, &lwork, &info);

  free(work);
  free(ipiv);
  return info;
#else
  return -1;
#endif
}

// Solve linear system: A @ x = b
// On input: a contains A (n x n), b contains b (n)
// On output: b contains x
// Returns 0 on success
int numbt_solve(moonbit_bytes_t a, moonbit_bytes_t b, int n) {
#ifdef __APPLE__
  float* A = (float*)a;
  float* B = (float*)b;
  int* ipiv = (int*)malloc(n * sizeof(int));
  if (!ipiv) return -1;

  int info;
  int N = n;
  int nrhs = 1;
  int lda = n;
  int ldb = n;

  // LAPACK uses column-major, so transpose A in place
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      float tmp = A[i * n + j];
      A[i * n + j] = A[j * n + i];
      A[j * n + i] = tmp;
    }
  }

  sgesv_(&N, &nrhs, A, &lda, ipiv, B, &ldb, &info);

  free(ipiv);
  return info;
#else
  return -1;
#endif
}

// ============================================================================
// SVD: A = U @ S @ V^T
// ============================================================================

// SVD decomposition
// a: m x n matrix (input, destroyed)
// u: m x m matrix (output)
// s: min(m,n) vector (output, singular values)
// vt: n x n matrix (output, V transposed)
// Returns 0 on success
int numbt_svd(moonbit_bytes_t a, moonbit_bytes_t u, moonbit_bytes_t s, moonbit_bytes_t vt, int m, int n) {
#ifdef __APPLE__
  float* A = (float*)a;
  float* U = (float*)u;
  float* S = (float*)s;
  float* VT = (float*)vt;

  int info;
  int M = m;
  int N = n;
  int lda = n;  // row-major: leading dim is n
  int ldu = m;
  int ldvt = n;
  int minmn = m < n ? m : n;

  // Transpose A for column-major LAPACK
  float* A_col = (float*)malloc(m * n * sizeof(float));
  if (!A_col) return -1;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      A_col[j * m + i] = A[i * n + j];
    }
  }

  // Query optimal workspace size
  int lwork = -1;
  float work_query;
  sgesvd_("A", "A", &M, &N, A_col, &M, S, U, &M, VT, &N, &work_query, &lwork, &info);

  lwork = (int)work_query;
  float* work = (float*)malloc(lwork * sizeof(float));
  if (!work) {
    free(A_col);
    return -1;
  }

  // Compute SVD
  sgesvd_("A", "A", &M, &N, A_col, &M, S, U, &M, VT, &N, work, &lwork, &info);

  // Transpose U back to row-major
  float* U_tmp = (float*)malloc(m * m * sizeof(float));
  if (U_tmp) {
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < m; j++) {
        U_tmp[i * m + j] = U[j * m + i];
      }
    }
    memcpy(U, U_tmp, m * m * sizeof(float));
    free(U_tmp);
  }

  // Transpose VT back to row-major (VT is already transposed, so we get V^T in row-major)
  float* VT_tmp = (float*)malloc(n * n * sizeof(float));
  if (VT_tmp) {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        VT_tmp[i * n + j] = VT[j * n + i];
      }
    }
    memcpy(VT, VT_tmp, n * n * sizeof(float));
    free(VT_tmp);
  }

  free(work);
  free(A_col);
  return info;
#else
  return -1;
#endif
}

// ============================================================================
// Eigenvalues (symmetric matrix): A @ v = lambda * v
// ============================================================================

// Eigenvalue decomposition for symmetric matrix
// a: n x n symmetric matrix (input, eigenvectors on output)
// w: n eigenvalues (output)
// Returns 0 on success
int numbt_eig_symmetric(moonbit_bytes_t a, moonbit_bytes_t w, int n) {
#ifdef __APPLE__
  float* A = (float*)a;
  float* W = (float*)w;

  int info;
  int N = n;
  int lda = n;

  // Transpose A for column-major LAPACK
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      float tmp = A[i * n + j];
      A[i * n + j] = A[j * n + i];
      A[j * n + i] = tmp;
    }
  }

  // Query optimal workspace size
  int lwork = -1;
  float work_query;
  ssyev_("V", "U", &N, A, &lda, W, &work_query, &lwork, &info);

  lwork = (int)work_query;
  float* work = (float*)malloc(lwork * sizeof(float));
  if (!work) return -1;

  // Compute eigenvalues and eigenvectors
  ssyev_("V", "U", &N, A, &lda, W, work, &lwork, &info);

  // Transpose A back to row-major (eigenvectors are in columns)
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      float tmp = A[i * n + j];
      A[i * n + j] = A[j * n + i];
      A[j * n + i] = tmp;
    }
  }

  free(work);
  return info;
#else
  return -1;
#endif
}

// ============================================================================
// Cholesky decomposition: A = L @ L^T
// ============================================================================

// Cholesky decomposition (lower triangular)
// a: n x n positive definite matrix (input, L on output in lower triangle)
// Returns 0 on success, >0 if not positive definite
int numbt_cholesky(moonbit_bytes_t a, int n) {
#ifdef __APPLE__
  float* A = (float*)a;

  int info;
  int N = n;
  int lda = n;

  // Transpose A for column-major LAPACK
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      float tmp = A[i * n + j];
      A[i * n + j] = A[j * n + i];
      A[j * n + i] = tmp;
    }
  }

  // Compute Cholesky decomposition (lower triangular)
  spotrf_("L", &N, A, &lda, &info);

  // Transpose back to row-major
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      float tmp = A[i * n + j];
      A[i * n + j] = A[j * n + i];
      A[j * n + i] = tmp;
    }
  }

  // Zero out upper triangle (Cholesky returns lower)
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      A[i * n + j] = 0.0f;
    }
  }

  return info;
#else
  return -1;
#endif
}

// ============================================================================
// QR decomposition: A = Q @ R
// ============================================================================

// QR decomposition
// a: m x n matrix (input)
// q: m x m orthogonal matrix (output)
// r: m x n upper triangular matrix (output)
// Returns 0 on success
int numbt_qr(moonbit_bytes_t a, moonbit_bytes_t q, moonbit_bytes_t r, int m, int n) {
#ifdef __APPLE__
  float* A = (float*)a;
  float* Q = (float*)q;
  float* R = (float*)r;

  int info;
  int M = m;
  int N = n;
  int lda = m;
  int minmn = m < n ? m : n;

  // Transpose A for column-major LAPACK
  float* A_col = (float*)malloc(m * n * sizeof(float));
  if (!A_col) return -1;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      A_col[j * m + i] = A[i * n + j];
    }
  }

  float* tau = (float*)malloc(minmn * sizeof(float));
  if (!tau) {
    free(A_col);
    return -1;
  }

  // Query optimal workspace size
  int lwork = -1;
  float work_query;
  sgeqrf_(&M, &N, A_col, &lda, tau, &work_query, &lwork, &info);

  lwork = (int)work_query;
  float* work = (float*)malloc(lwork * sizeof(float));
  if (!work) {
    free(tau);
    free(A_col);
    return -1;
  }

  // Compute QR factorization
  sgeqrf_(&M, &N, A_col, &lda, tau, work, &lwork, &info);
  if (info != 0) {
    free(work);
    free(tau);
    free(A_col);
    return info;
  }

  // Extract R (upper triangular, stored in A_col)
  memset(R, 0, m * n * sizeof(float));
  for (int i = 0; i < minmn; i++) {
    for (int j = i; j < n; j++) {
      R[i * n + j] = A_col[j * m + i];  // Transpose back
    }
  }

  // Generate Q
  int K = minmn;
  sorgqr_(&M, &M, &K, A_col, &lda, tau, work, &lwork, &info);

  // Transpose Q back to row-major
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < m; j++) {
      Q[i * m + j] = A_col[j * m + i];
    }
  }

  free(work);
  free(tau);
  free(A_col);
  return info;
#else
  return -1;
#endif
}

// ============================================================================
// Determinant (via LU decomposition)
// ============================================================================

// Compute determinant via LU decomposition
// a: n x n matrix (input, destroyed)
// det: determinant (output)
// Returns 0 on success
int numbt_det(moonbit_bytes_t a, float* det, int n) {
#ifdef __APPLE__
  float* A = (float*)a;
  int* ipiv = (int*)malloc(n * sizeof(int));
  if (!ipiv) return -1;

  int info;
  int N = n;
  int lda = n;

  // Transpose A for column-major LAPACK
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      float tmp = A[i * n + j];
      A[i * n + j] = A[j * n + i];
      A[j * n + i] = tmp;
    }
  }

  // LU factorization
  sgetrf_(&N, &N, A, &lda, ipiv, &info);
  if (info != 0) {
    free(ipiv);
    *det = 0.0f;
    return info;
  }

  // Compute determinant from diagonal of U
  float d = 1.0f;
  int sign = 1;
  for (int i = 0; i < n; i++) {
    d *= A[i * n + i];  // diagonal element
    if (ipiv[i] != i + 1) sign = -sign;  // permutation sign
  }
  *det = d * sign;

  free(ipiv);
  return 0;
#else
  return -1;
#endif
}

// ============================================================================
// Least squares: minimize ||A @ x - b||
// ============================================================================

// Least squares solution
// a: m x n matrix (input, destroyed)
// b: m vector (input, solution x on output for first n elements)
// Returns 0 on success
int numbt_lstsq(moonbit_bytes_t a, moonbit_bytes_t b, int m, int n) {
#ifdef __APPLE__
  float* A = (float*)a;
  float* B = (float*)b;

  int info;
  int M = m;
  int N = n;
  int nrhs = 1;
  int lda = m;
  int ldb = m;

  // Transpose A for column-major LAPACK
  float* A_col = (float*)malloc(m * n * sizeof(float));
  if (!A_col) return -1;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      A_col[j * m + i] = A[i * n + j];
    }
  }

  // Query optimal workspace size
  int lwork = -1;
  float work_query;
  sgels_("N", &M, &N, &nrhs, A_col, &lda, B, &ldb, &work_query, &lwork, &info);

  lwork = (int)work_query;
  float* work = (float*)malloc(lwork * sizeof(float));
  if (!work) {
    free(A_col);
    return -1;
  }

  // Solve least squares
  sgels_("N", &M, &N, &nrhs, A_col, &lda, B, &ldb, work, &lwork, &info);

  free(work);
  free(A_col);
  return info;
#else
  return -1;
#endif
}
