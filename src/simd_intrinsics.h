#pragma once

// NOTE: scalar 32-bit float type. always available
struct R32x1
{
  using Scalar = R32x1;
  static usz constexpr count = 1; // number of elements in "vector"
  static usz constexpr size = 4; // size of "vector" in bytes

  r32 v;

  R32x1() {}
  R32x1(r32 x) : v(x) {}

  static R32x1 load(r32 const *src) { return(R32x1(*src)); }
  void store(r32 *dest) { *dest = v; }

  R32x1 operator+(R32x1 b) { return(R32x1(v + b.v)); }
  R32x1 operator-(R32x1 b) { return(R32x1(v - b.v)); }
  R32x1 operator*(R32x1 b) { return(R32x1(v * b.v)); }

  static FORCE_INLINE void
  load_deinterleave(R32x1 &re, R32x1 &im, r32 const *src)
  {
    re.v = src[0];
    im.v = src[1];
  }

  static FORCE_INLINE void
  store_interleaved(r32 *dest, R32x1 re, R32x1 im)
  {
    dest[0] = re.v;
    dest[1] = im.v;
  }

  static FORCE_INLINE void
  reverse(R32x1 &a) { UNUSED(a); }
};

#if ARCH_X86 || ARCH_X64

#include <immintrin.h>

// TODO: preprocessor checks to only compile if extensions are available on builder's machine
// NOTE: 4x 32-bit float type. requires sse extensions (TODO: which versions?)
struct R32x4
{
  using Scalar = R32x1;
  static usz constexpr count = 4; // number of elements in vector
  static usz constexpr size = 16; // size of vector in bytes

  __m128 v;

  R32x4() {}
  explicit R32x4(__m128 x) : v(x) {}
  explicit R32x4(r32 x) : v(_mm_set1_ps(x)) {}

  static R32x4 zero(void) { return(R32x4(_mm_setzero_ps())); }
  static R32x4 neg1(void) { return(R32x4(-1.f)); }

  static R32x4 load(r32 const *src) { return(R32x4(_mm_loadu_ps(src))); }
  void store(r32 *dest) { _mm_storeu_ps(dest, v); }

  R32x4 operator+(R32x4 b) { return(R32x4(_mm_add_ps(v, b.v))); }
  R32x4 operator-(R32x4 b) { return(R32x4(_mm_sub_ps(v, b.v))); }
  R32x4 operator*(R32x4 b) { return(R32x4(_mm_mul_ps(v, b.v))); }

  static FORCE_INLINE void
  load_deinterleave(R32x4 &re, R32x4 &im, r32 const *src)
  {
    __m128 const t0 = _mm_loadu_ps(src);
    __m128 const t1 = _mm_loadu_ps(src + count);

    re.v = _mm_shuffle_ps(t0, t1, _MM_SHUFFLE(2, 0, 2, 0));
    im.v = _mm_shuffle_ps(t0, t1, _MM_SHUFFLE(3, 1, 3, 1));
  }

  static FORCE_INLINE void
  store_interleaved(r32 *dest, R32x4 re, R32x4 im)
  {
    __m128 const t0 = _mm_unpacklo_ps(re.v, im.v);
    __m128 const t1 = _mm_unpackhi_ps(re.v, im.v);

    _mm_storeu_ps(dest, t0);
    _mm_storeu_ps(dest + count, t1);
  }

  static FORCE_INLINE void
  transpose4x4(R32x4 &a, R32x4 &b, R32x4 &c, R32x4 &d)
  {
    __m128 t0 = _mm_unpacklo_ps(a.v, c.v);
    __m128 t1 = _mm_unpacklo_ps(b.v, d.v);
    __m128 t2 = _mm_unpackhi_ps(a.v, c.v);
    __m128 t3 = _mm_unpackhi_ps(b.v, d.v);

    a.v = _mm_unpacklo_ps(t0, t1);
    b.v = _mm_unpackhi_ps(t0, t1);
    c.v = _mm_unpacklo_ps(t2, t3);
    d.v = _mm_unpackhi_ps(t2, t3);
  }

  static FORCE_INLINE void
  reverse(R32x4 &a)
  {
    a.v = _mm_shuffle_ps(a.v, a.v, _MM_SHUFFLE(0, 1, 2, 3));
  }
};

// TODO: untested
// NOTE: 8x 32-bit float type. requires avx extensions (TODO: which versions?)
struct R32x8
{
  using Scalar = R32x1;
  static usz constexpr count = 8; // number of elements in vector
  static usz constexpr size = 32; // size of vector in bytes

  __m256 v;

  R32x8() {}
  explicit R32x8(__m256 x) : v(x) {}
  explicit R32x8(r32 x) : v(_mm256_set1_ps(x)) {}

  static R32x8 zero(void) { return(R32x8(_mm256_setzero_ps())); }
  static R32x8 neg1(void) { return(R32x8(-1.f)); }

  static R32x8 load(r32 const *src) { return(R32x8(_mm256_loadu_ps(src))); }
  void store(r32 *dest) { _mm256_storeu_ps(dest, v); }

  R32x8 operator+(R32x8 b) { return(R32x8(_mm256_add_ps(v, b.v))); }
  R32x8 operator-(R32x8 b) { return(R32x8(_mm256_sub_ps(v, b.v))); }
  R32x8 operator*(R32x8 b) { return(R32x8(_mm256_mul_ps(v, b.v))); }

  static FORCE_INLINE void
  load_deinterleave(R32x8 &re, R32x8 &im, r32 const *src)
  {
    __m256 const t0 = _mm256_loadu_ps(src);
    __m256 const t1 = _mm256_loadu_ps(src + count);

    __m256 const s0 = _mm256_permute2f128_ps(t0, t1, 0x20);
    __m256 const s1 = _mm256_permute2f128_ps(t0, t1, 0x31);

    re.v = _mm256_shuffle_ps(s0, s1, _MM_SHUFFLE(2, 0, 2, 0));
    im.v = _mm256_shuffle_ps(s0, s1, _MM_SHUFFLE(3, 1, 3, 1));
  }

  static FORCE_INLINE void
  store_interleaved(r32 *dest, R32x8 re, R32x8 im)
  {
    __m256 const t0 = _mm256_unpacklo_ps(re.v, im.v);
    __m256 const t1 = _mm256_unpackhi_ps(re.v, im.v);

    // s0[127:0]   = t0[127:0]
    // s0[255:128] = t1[127:0]
    // s1[127:0]   = t0[255:128]
    // s2[255:128] = t1[255:128];
    __m256 const s0 = _mm256_permute2f128_ps(t0, t1, 0x20);
    __m256 const s1 = _mm256_permute2f128_ps(t0, t1, 0x31);

    _mm256_storeu_ps(dest, s0);
    _mm256_storeu_ps(dest + count, s1);
  }

  static FORCE_INLINE void
  transpose8x8(R32x8 &a, R32x8 &b, R32x8 &c, R32x8 &d,
	       R32x8 &e, R32x8 &f, R32x8 &g, R32x8 &h)
  {
    __m256 t0 = _mm256_unpacklo_ps(a.v, b.v);
    __m256 t1 = _mm256_unpackhi_ps(a.v, b.v);
    __m256 t2 = _mm256_unpacklo_ps(c.v, d.v);
    __m256 t3 = _mm256_unpackhi_ps(c.v, d.v);
    __m256 t4 = _mm256_unpacklo_ps(e.v, f.v);
    __m256 t5 = _mm256_unpackhi_ps(e.v, f.v);
    __m256 t6 = _mm256_unpacklo_ps(g.v, h.v);
    __m256 t7 = _mm256_unpackhi_ps(g.v, h.v);

    __m256 s0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 s1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 s2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 s3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 s4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 s5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 s6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 s7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));

    a.v = _mm256_permute2f128_ps(s0, s4, 0x20);
    b.v = _mm256_permute2f128_ps(s1, s5, 0x20);
    c.v = _mm256_permute2f128_ps(s2, s6, 0x20);
    d.v = _mm256_permute2f128_ps(s3, s7, 0x20);
    e.v = _mm256_permute2f128_ps(s0, s4, 0x31);
    f.v = _mm256_permute2f128_ps(s1, s5, 0x31);
    g.v = _mm256_permute2f128_ps(s2, s6, 0x31);
    h.v = _mm256_permute2f128_ps(s3, s7, 0x31);
  }

  static FORCE_INLINE void
  reverse(R32x8 &a)
  {
    __m256 t = _mm256_shuffle_ps(a.v, a.v, _MM_SHUFFLE(0, 1, 2, 3));
    a.v = _mm256_permute2f128_ps(t, t, 0x31);
  }
};

#elif ARCH_ARM || ARCH_ARM64

// TODO: neon intrinsics

#elif ARCH_WASM

// TODO: wasm intrinsics

#else
#  warning vector instructions not supported for this architecture
#endif

union WideFloat;
union WideInt;

static WideFloat wideLoadFloats(r32 *src);
static WideFloat wideSetConstantFloats(r32 src);
static WideFloat wideSetFloats(r32 a, r32 b, r32 c, r32 d);
static void      wideSetLaneFloats(WideFloat *w, r32 val, u32 lane);
static void      wideStoreFloats(r32 *dest, WideFloat src);
static WideFloat wideAddFloats(WideFloat a, WideFloat b);
static WideFloat wideSubFloats(WideFloat a, WideFloat b);
static WideFloat wideMulFloats(WideFloat a, WideFloat b);
static WideFloat wideMaskFloats(WideFloat a, WideFloat b, WideInt mask);

static void wideInterleave(r32 *dest, r32 *srcL, r32 *srcR, u32 frameCount);
static void wideDeinterleave(r32 *destL, r32 *destR, r32 *src, u32 frameCount);

static WideInt   wideLoadInts(u32 *src);
static WideInt   wideSetConstantInts(u32 src);
static WideInt   wideSetInts(u32 a, u32 b, u32 c, u32 d);
static void      wideSetLaneInts(WideInt *w, u32 val, u32 lane);
static void      wideStoreInts(u32 *dest, WideInt src);
static WideInt   wideAddInts(WideInt a, WideInt b);
static WideInt   wideSubInts(WideInt a, WideInt b);
static WideInt   wideMulInts(WideInt a, WideInt b);
static WideInt   wideAndInts(WideInt a, WideInt b);

#if ARCH_X86 || ARCH_X64

#include <immintrin.h>

union WideFloat
{
  __m128 val;
  r32 floats[4];
};

union WideInt
{
  __m128i val;
  u32 ints[4];
};

static WideFloat
wideLoadFloats(r32 *src)
{
  WideFloat result = {};
  result.val = _mm_loadu_ps(src);

  return(result);
}

static WideInt
wideLoadInts(u32 *src)
{
  WideInt result = {};
  result.val = _mm_loadu_si128((__m128i *)src);

  return(result);
}

static WideFloat
wideSetConstantFloats(r32 src)
{
  WideFloat result = {};
  result.val = _mm_set1_ps(src);

  return(result);
}

static WideFloat
wideSetFloats(r32 a, r32 b, r32 c, r32 d)
{
  WideFloat result = {};
  // r32 *val = (r32 *)&result.val;
  // val[0] = a;
  // val[1] = b;
  // val[2] = c;
  // val[3] = d;
  result.floats[0] = a;
  result.floats[1] = b;
  result.floats[2] = c;
  result.floats[3] = d;

  return(result);
}

static void
wideSetLaneFloats(WideFloat *w, r32 val, u32 lane)
{
  // r32 *vals = (r32 *)&w->val;
  // vals[lane] = val;
  w->floats[lane] = val;
}

static WideInt
wideSetConstantInts(u32 src)
{
  WideInt result = {};
  result.val = _mm_set1_epi32(src);

  return(result);
}

static WideInt
wideSetInts(u32 a, u32 b, u32 c, u32 d)
{
  WideInt result = {};
  // u32 *val = (u32 *)&result.val; // TODO: check this is legal
  // val[0] = a;
  // val[1] = b;
  // val[2] = c;
  // val[3] = d;
  result.ints[0] = a;
  result.ints[1] = b;
  result.ints[2] = c;
  result.ints[3] = d;

  return(result);
}

static void
wideSetLaneInts(WideInt *w, u32 val, u32 lane)
{
  // u32 *vals = (u32 *)&w->val;
  // vals[lane]= val;
  w->ints[lane] = val;
}

static void
wideStoreFloats(r32 *dest, WideFloat src)
{
  _mm_storeu_ps(dest, src.val);
}

static void
wideStoreInts(u32 *dest, WideInt src)
{
  _mm_storeu_si128((__m128i *)dest, src.val);
}

static WideFloat
wideAddFloats(WideFloat a, WideFloat b)
{
  WideFloat result = {};
  result.val = _mm_add_ps(a.val, b.val);

  return(result);
}

static WideInt
wideAddInts(WideInt a, WideInt b)
{
  WideInt result = {};
  result.val = _mm_add_epi32(a.val, b.val);

  return(result);
}

static WideFloat
wideSubFloats(WideFloat a, WideFloat b)
{
  WideFloat result = {};
  result.val = _mm_sub_ps(a.val, b.val);

  return(result);
}

static WideInt
wideSubInts(WideInt a, WideInt b)
{
  WideInt result = {};
  result.val = _mm_sub_epi32(a.val, b.val);

  return(result);
}

static WideFloat
wideMulFloats(WideFloat a, WideFloat b)
{
  WideFloat result = {};
  result.val = _mm_mul_ps(a.val, b.val);

  return(result);
}

static WideFloat
wideMaskFloats(WideFloat a, WideFloat b, WideInt mask)
{
  WideFloat result = {};
  __m128 maskF = _mm_castsi128_ps(mask.val);
  result.val = _mm_or_ps(_mm_and_ps(maskF, a.val),
                         _mm_andnot_ps(maskF, b.val));

  return(result);
}

static WideInt
wideMulInts(WideInt a, WideInt b)
{
  WideInt result = {};
  result.val = _mm_mul_epi32(a.val, b.val);

  return(result);
}

static WideInt
wideAndInts(WideInt a, WideInt b)
{
  WideInt result = {};
  result.val = _mm_and_si128(a.val, b.val);

  return(result);
}

static void
wideInterleave(SamplePair *dest, r32 *srcL, r32 *srcR, u32 frameCount)
{
  ASSERT((frameCount & 3) == 0); // NOTE: check the number of frames is divisible by the SIMD width
  for(u32 i = 0; i < frameCount; i += 4)
  {
    __m128 left  = _mm_loadu_ps(srcL);
    __m128 right = _mm_loadu_ps(srcR);
    srcL += 4;
    srcR += 4;

    __m128 interleavedLo = _mm_unpacklo_ps(left, right);
    __m128 interleavedHi = _mm_unpackhi_ps(left, right);
    _mm_storeu_ps((r32*)dest, interleavedLo);
    dest += 2;
    _mm_storeu_ps((r32*)dest, interleavedHi);
    dest += 2;
  }
}

static void
wideDeinterleave(r32 *destL, r32 *destR, SamplePair *src, u32 frameCount)
{
  ASSERT((frameCount & 3) == 0); // NOTE: check the number of frames is divisible by the SIMD width
  for(u32 i = 0; i < frameCount; i += 4)
  {
    __m128 interleaved0 = _mm_loadu_ps((r32*)src);
    src += 2;
    __m128 interleaved1 = _mm_loadu_ps((r32*)src);
    src += 2;

    __m128 left  = _mm_shuffle_ps(interleaved0, interleaved1, _MM_SHUFFLE(2, 0, 2, 0));
    __m128 right = _mm_shuffle_ps(interleaved0, interleaved1, _MM_SHUFFLE(3, 1, 3, 1));
    _mm_storeu_ps(destL, left);
    _mm_storeu_ps(destR, right);
    destL += 4;
    destR += 4;
  }
}

#elif ARCH_ARM || ARCH_ARM64

#include <arm_neon.h>

union WideFloat
{
  float32x4_t val;
  r32 floats[4];
};

union WideInt
{
  uint32x4_t val;
  u32 ints[4];
};

static WideFloat
wideLoadFloats(r32 *src)
{
  WideFloat result = {};
  result.val = vld1q_f32(src);

  return(result);
}

static WideInt
wideLoadInts(u32 *src)
{
  WideInt result = {};
  result.val = vld1q_u32(src);

  return(result);
}

static WideFloat
wideSetConstantFloats(r32 src)
{
  WideFloat result = {};
  result.val = vdupq_n_f32(src);

  return(result);
}

static WideInt
wideSetConstantInts(u32 src)
{
  WideInt result = {};
  result.val = vdupq_n_u32(src);

  return(result);
}

static WideFloat
wideSetFloats(r32 a, r32 b, r32 c, r32 d)
{
  WideFloat result = {};
  result.val = {a, b, c, d};

  return(result);
}

static WideInt
wideSetInts(u32 a, u32 b, u32 c, u32 d)
{
  WideInt result = {};
  result.val = {a, b, c, d};

  return(result);
}

static void
wideSetLaneFloats(WideFloat *w, r32 val, u32 lane)
{
  float32x4_t temp = vdupq_n_f32(val);

  u32 mask[4] = {};
  mask[lane] = U32_MAX;
  uint32x4_t maskVec = vld1q_u32(mask);

  w->val = vbslq_f32(maskVec, temp, w->val);
}

static void
wideSetLaneInts(WideInt *w, u32 val, u32 lane)
{
  uint32x4_t temp = vdupq_n_u32(val);

  u32 mask[4] = {};
  mask[lane] = U32_MAX;
  uint32x4_t maskVec = vld1q_u32(mask);

  w->val = vbslq_u32(maskVec, temp, w->val);
}

static void
wideStoreFloats(r32 *dest, WideFloat src)
{
  vst1q_f32(dest, src.val);
}

static void
wideStoreInts(u32 *dest, WideInt src)
{
  vst1q_u32(dest, src.val);
}

static WideFloat
wideAddFloats(WideFloat a, WideFloat b)
{
  WideFloat result = {};
  result.val = vaddq_f32(a.val, b.val);

  return(result);
}

static WideInt
wideAddInts(WideInt a, WideInt b)
{
  WideInt result = {};
  result.val = vaddq_u32(a.val, b.val);

  return(result);
}

static WideFloat
wideSubFloats(WideFloat a, WideFloat b)
{
  WideFloat result = {};
  result.val = vsubq_f32(a.val, b.val);

  return(result);
}

static WideInt
wideSubInts(WideInt a, WideInt b)
{
  WideInt result = {};
  result.val = vsubq_u32(a.val, b.val);

  return(result);
}

static WideFloat
wideMulFloats(WideFloat a, WideFloat b)
{
  WideFloat result = {};
  result.val = vmulq_f32(a.val, b.val);

  return(result);
}

static WideInt
wideMulInts(WideInt a, WideInt b)
{
  WideInt result = {};
  result.val = vmulq_u32(a.val, b.val);

  return(result);
}

static WideFloat
wideMaskFloats(WideFloat a, WideFloat b, WideInt mask)
{
  WideFloat result = {};
  result.val = vbslq_f32(mask.val, a.val, b.val);

  return(result);
}

static WideInt
wideAndInts(WideInt a, WideInt b)
{
  WideInt result = {};
  result.val = vandq_u32(a.val, b.val);
  return(result);
}

#elif ARCH_WASM32 || ARCH_WASM64

#include <wasm_simd128.h>

union WideFloat
{
  v128_t val;
  r32 floats[4];
};

union WideInt
{
  v128_t val;
  u32 ints[4];
};

static WideFloat
wideLoadFloats(r32 *src)
{
  WideFloat result = {};
  result.val = wasm_v128_load(src);
  return(result);
}

static WideFloat
wideSetConstantFloats(r32 src)
{
  WideFloat result = {};
  result.val = wasm_f32x4_make(src, src, src, src);
  return(result);
}

static WideFloat
wideSetFloats(r32 a, r32 b, r32 c, r32 d)
{
  WideFloat result = {};
  result.val = wasm_f32x4_make(a, b, c, d);
  return(result);
}

static void
wideSetLaneFloats(WideFloat *w, r32 val, u32 lane)
{
  // NOTE: this sucks
  // switch(lane)
  //   {
  //   case 0: { w->val = wasm_f32x4_replace_lane(w->val, 0, val); } break;
  //   case 1: { w->val = wasm_f32x4_replace_lane(w->val, 1, val); } break;
  //   case 2: { w->val = wasm_f32x4_replace_lane(w->val, 2, val); } break;
  //   case 3: { w->val = wasm_f32x4_replace_lane(w->val, 3, val); } break;
  //   default: { ASSERT(!"lane index out of bounds"); } break;
  //   }
  w->floats[lane] = val;
}

static void
wideStoreFloats(r32 *dest, WideFloat src)
{
  wasm_v128_store(dest, src.val);
}

static WideFloat
wideAddFloats(WideFloat a, WideFloat b)
{
  WideFloat result = {};
  result.val = wasm_f32x4_add(a.val, b.val);
  return(result);
}

static WideFloat
wideSubFloats(WideFloat a, WideFloat b)
{
  WideFloat result = {};
  result.val = wasm_f32x4_sub(a.val, b.val);
  return(result);
}

static WideFloat
wideMulFloats(WideFloat a, WideFloat b)
{
  WideFloat result = {};
  result.val = wasm_f32x4_mul(a.val, b.val);
  return(result);
}

static WideFloat
wideMaskFloats(WideFloat a, WideFloat b, WideInt mask)
{
  WideFloat result = {};
  result.val = wasm_v128_or(wasm_v128_and(a.val, mask.val),
                            wasm_v128_andnot(b.val, mask.val));
  return(result);
}

static WideInt
wideLoadInts(u32 *src)
{
  WideInt result = {};
  result.val = wasm_v128_load(src);
  return(result);
}

static WideInt
wideSetConstantInts(u32 src)
{
  WideInt result = {};
  result.val = wasm_u32x4_make(src, src, src, src);
  return(result);
}

static WideInt
wideSetInts(u32 a, u32 b, u32 c, u32 d)
{
  WideInt result = {};
  result.val = wasm_u32x4_make(a, b, c, d);
  return(result);
}

static void
wideSetLaneInts(WideInt *w, u32 val, u32 lane)
{
  // NOTE: this still sucks
  // switch(lane)
  //   {
  //   case 0: { w->val = wasm_u32x4_replace_lane(w->val, 0, val); } break;
  //   case 1: { w->val = wasm_u32x4_replace_lane(w->val, 1, val); } break;
  //   case 2: { w->val = wasm_u32x4_replace_lane(w->val, 2, val); } break;
  //   case 3: { w->val = wasm_u32x4_replace_lane(w->val, 3, val); } break;
  //   default: { ASSERT(!"lane index out of bounds"); } break;
  //   }
  w->ints[lane] = val;
}

static void
wideStoreInts(u32 *dest, WideInt src)
{
  wasm_v128_store(dest, src.val);
}

static WideInt
wideAddInts(WideInt a, WideInt b)
{
  WideInt result = {};
  result.val = wasm_i32x4_add(a.val, b.val);
  return(result);
}

static WideInt
wideSubInts(WideInt a, WideInt b)
{
  WideInt result = {};
  result.val = wasm_i32x4_sub(a.val, b.val);
  return(result);
}

static WideInt
wideMulInts(WideInt a, WideInt b)
{
  WideInt result = {};
  result.val = wasm_i32x4_mul(a.val, b.val);
  return(result);
}

static WideInt
wideAndInts(WideInt a, WideInt b)
{
  WideInt result = {};
  result.val = wasm_v128_and(a.val, b.val);
  return(result);
}

#else
// NOTE: default to scalar

struct WideFloat
{
  r32 val;
};

struct WideInt
{
  u32 val;
};

static WideFloat
wideLoadFloats(r32 *src)
{
  WideFloat result = { *src };
  return(result);
}

static WideFloat
wideSetConstantFloats(r32 src)
{
  WideFloat result = { src };
  return(result);
}

// TODO: make sure this is ok
static WideFloat
wideSetFloats(r32 a, r32 b, r32 c, r32 d)
{
  WideFloat result = { a };
  UNUSED(b);
  UNUSED(c);
  UNUSED(d);
  return(result);
}

static void
wideSetLaneFloats(WideFloat *w, r32 val, u32 lane)
{
  w->val = val;
  UNUSED(lane);
}

static void
wideStoreFloats(r32 *dest, WideFloat src)
{
  *dest = src.val;
}

static WideFloat
wideAddFloats(WideFloat a, WideFloat b)
{
  WideFloat result = { a.val + b.val };
  return(result);
}

static WideFloat
wideSubFloats(WideFloat a, WideFloat b)
{
  WideFloat result = { a.val - b.val };
  return(result);
}

static WideFloat
wideMulFloats(WideFloat a, WideFloat b)
{
  WideFloat result = { a.val * b.val };
  return(result);
}

static WideFloat
wideMaskFloats(WideFloat a, WideFloat b, WideInt mask)
{
  //  TODO: implement
  WideFloat result = {};
  return(result);
}

static WideInt
wideLoadInts(u32 *src)
{
  WideInt result = { *src };
  return(result);
}

static WideInt
wideSetConstantInts(u32 src)
{
  WideInt result = { src };
  return(result);
}

static WideInt
wideSetInts(u32 a, u32 b, u32 c, u32 d)
{
  // TODO: make sure this is ok
  WideInt result = { a };
  UNUSED(b);
  UNUSED(c);
  UNUSED(d);
  return(result);
}

static void
wideSetLaneInts(WideInt *w, u32 val, u32 lane)
{
  w->val = val;
  UNUSED(lane);
}

static void
wideStoreInts(u32 *dest, WideInt src)
{
  *dest = src.val;
}

static WideInt
wideAddInts(WideInt a, WideInt b)
{
  WideInt result = { a.val + b.val };
  return(result);
}

static WideInt
wideSubInts(WideInt a, WideInt b)
{
  WideInt result = { a.val - b.val };
  return(result);
}

static WideInt
wideMulInts(WideInt a, WideInt b)
{
  WideInt result = { a.val * b.val };
  return(result);
}

static WideInt
wideAndInts(WideInt a, WideInt b)
{
  WideInt result = { a.val & b.val };
  return(result);
}

#endif

#if LANG_CPP
static inline WideFloat operator+(WideFloat a, WideFloat b) { return(wideAddFloats(a, b)); }
static inline WideFloat operator-(WideFloat a, WideFloat b) { return(wideSubFloats(a, b)); }
static inline WideFloat operator*(WideFloat a, WideFloat b) { return(wideMulFloats(a, b)); }
static inline WideFloat& operator+=(WideFloat& a, WideFloat b) { a = a + b; return(a); }
static inline WideFloat& operator-=(WideFloat& a, WideFloat b) { a = a - b; return(a); }
static inline WideFloat& operator*=(WideFloat& a, WideFloat b) { a = a + b; return(a); }

static inline WideInt operator+(WideInt a, WideInt b) { return(wideAddInts(a, b)); }
static inline WideInt operator-(WideInt a, WideInt b) { return(wideSubInts(a, b)); }
static inline WideInt operator*(WideInt a, WideInt b) { return(wideMulInts(a, b)); }
static inline WideInt operator&(WideInt a, WideInt b) { return(wideAndInts(a, b)); }
static inline WideInt& operator+=(WideInt& a, WideInt b) { a = a + b; return(a); }
static inline WideInt& operator-=(WideInt& a, WideInt b) { a = a - b; return(a); }
static inline WideInt& operator*=(WideInt& a, WideInt b) { a = a * b; return(a); }
static inline WideInt& operator&=(WideInt& a, WideInt b) { a = a & b; return(a); }
#endif
