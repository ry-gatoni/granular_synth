// TODO: investigate different fft radices

#ifndef PROFILE_FUNCTION
#  define PROFILE_FUNCTION()
#endif

#ifndef PROFILE_BLOCK
#  define PROFILE_BLOCK(name)
#endif

#define FFT_FUNCTION(name) ComplexBuffer (name)(Arena *arena, FloatBuffer input)
#define IFFT_FUNCTION(name) FloatBuffer (name)(Arena *arena, ComplexBuffer input)
// #define FFT_FUNCTION(name) void (name)(ComplexBuffer *output, FloatBuffer input)
// #define IFFT_FUNCTION(name) void (name)(FloatBuffer *output, ComplexBuffer input)
typedef FFT_FUNCTION(FFT_Function);
typedef IFFT_FUNCTION(IFFT_Function);

static FFT_Function *fft = 0;
static IFFT_Function *ifft = 0;

// TODO: put in simd layer
#include <immintrin.h>
struct R32x4
{
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

  template<int dest_idx, int src_idx>
  static FORCE_INLINE void
  overwrite(R32x4 &a, R32x4 b)
  {
    a.v = _mm_insert_ps(a.v, b.v, _MM_MK_INSERTPS_NDX(src_idx, dest_idx, 0));
  }
};

#define MAX_FFT_COUNT 4096
// NOTE:
// twiddle_table__radix_2[2N:4N] stores the N twiddle factors for each of the forward and reverse dft
// twiddle_table__radix_2[2N + k] = sin(2pi*k/2N)
// twiddle_table__radix_2[2N + k + N/2] = sin(2pi*(k + N/2)/2N)
//                                      = sin(2pi*k/2N + pi/2)
//                                      = cos(2pi*k/2N)
// twiddle_table__radix_2[2N + k + N] = sin(2pi*(k + N)/2N)
//                                    = sin(2pi*k/2N + pi)
//                                    = -sin(2pi*k/2N)
static r32 twiddle_table__radix_2[4*MAX_FFT_COUNT] = {};

static void
fftInitTwiddleTables(void)
{
  for(usz n = 1; n <= MAX_FFT_COUNT; n *= 2)
  {
    r64 step = GS_TAU / (r64)(2*n);
    r32 *twiddles = twiddle_table__radix_2 + 2*n;
    for(usz k = 0; k < 2*n; ++k)
    {
      r64 phase = step * (r64)k;
      r32 sine = gsSin((r32)phase);
      //twiddles[k] = C64Polar(1, phase);
      twiddles[k] = sine;
    }
  }
}

enum FftDirection
{
  FftDirection_forward, /* fft */
  FftDirection_inverse, /* ifft */
};

// NOTE: template parameter should be one of the simd types in simd.h
template<typename T, FftDirection dir>
static void
fft_kernel__radix_2(c64 *io, usz level, usz count)
{
  T two(2.f);
  for(usz k = 0; k < count; k += 2*level)
  {
    r32 *twiddles_im = twiddle_table__radix_2 + 2*level;
    r32 *twiddles_re = twiddles_im + level/2;
    if(dir == FftDirection_forward) // NOTE: -sine for forward transform
    { twiddles_im += level; }

    c64 *io0 = io + 0*level + k;
    c64 *io1 = io + 1*level + k;

    for(usz j = 0; j < level; j += T::count)
    {
      T w_re = T::load(twiddles_re);
      T w_im = T::load(twiddles_im);

      T in0_re, in0_im;
      T::load_deinterleave(in0_re, in0_im, (r32*)io0);

      T in1_re, in1_im;
      T::load_deinterleave(in1_re, in1_im, (r32*)io1);

      T out0_re = in0_re + w_re*in1_re - w_im*in1_im;
      T out0_im = in0_im + w_re*in1_im + w_im*in1_re;
      T out1_re = two*in0_re - out0_re;
      T out1_im = two*in0_im - out0_im;

      T::store_interleaved((r32*)io0, out0_re, out0_im);
      T::store_interleaved((r32*)io1, out1_re, out1_im);

      twiddles_re += T::count;
      twiddles_im += T::count;

      io0 += T::count;
      io1 += T::count;
    }
  }
}

//template void fft_kernel__radix_2<R32x4>(c64 *io, usz level, usz count, FftSign sign);

template<typename T, FftDirection dir>
static usz
fft_initial__radix_2(c64 *out, c64 *in, usz count)
{
  usz count_log2 = LOG2(count);
  for(usz block_idx = 0; block_idx < count/T::count; block_idx += T::count)
  {
    // TODO: this whole 4x4 transpose thing is specific to the vector
    // width. 8-wide vectors would do 8 loads, 3 initial steps, an 8x8
    // transpose, then 8 complex stores. so we need different branches for
    // different statically-known vector widths, or maybe make this not
    // templated at all (though that might be annoying abstracting over
    // sse/neon/wasm)
    usz block0_rev = reverseBits(block_idx + 0) >> (8*sizeof(block_idx) - count_log2);
    usz block1_rev = reverseBits(block_idx + 1) >> (8*sizeof(block_idx) - count_log2);
    usz block2_rev = reverseBits(block_idx + 2) >> (8*sizeof(block_idx) - count_log2);
    usz block3_rev = reverseBits(block_idx + 3) >> (8*sizeof(block_idx) - count_log2);

    T in0_re, in0_im;
    T::load_deinterleave(in0_re, in0_im, (r32*)(in + block0_rev));
    T in1_re, in1_im;
    T::load_deinterleave(in1_re, in1_im, (r32*)(in + block1_rev));
    T in2_re, in2_im;
    T::load_deinterleave(in2_re, in2_im, (r32*)(in + block2_rev));
    T in3_re, in3_im;
    T::load_deinterleave(in3_re, in3_im, (r32*)(in + block3_rev));

    // NOTE: initial radix-2 step (stride = 1)
    T r0_re = in0_re + in1_re;
    T r0_im = in0_im + in1_im;
    T r1_re = in0_re - in1_re;
    T r1_im = in0_im - in1_im;
    T r2_re = in2_re + in3_re;
    T r2_im = in2_im + in3_im;
    T r3_re = in2_re - in3_re;
    T r3_im = in2_im - in3_im;

    // NOTE: subsequent radix-2 step (stride = 2)
    T s0_re, s0_im, s1_re, s1_im, s2_re, s2_im, s3_re, s3_im;
    if(dir == FftDirection_forward)
    {
      s0_re = r0_re + r2_re;
      s0_im = r0_im + r2_im;
      s1_re = r1_re + r3_im;
      s1_im = r1_im - r3_re;
      s2_re = r0_re - r2_re;
      s2_im = r0_im - r2_im;
      s3_re = r1_re - r3_im;
      s3_im = r1_im + r3_re;
    }
    else // dir == FftDirection_inverse
    {
      T t0_re = r0_re + r2_re;
      T t0_im = r0_im + r2_im;
      T t1_re = r1_re - r3_im;
      T t1_im = r1_im + r3_re;
      T t2_re = r0_re - r2_re;
      T t2_im = r0_im - r2_im;
      T t3_re = r1_re + r3_im;
      T t3_im = r1_im - r3_re;

      T count_inv(1.f / (r32)count);
      s0_re = count_inv * t0_re;
      s0_im = count_inv * t0_im;
      s1_re = count_inv * t1_re;
      s1_im = count_inv * t1_im;
      s2_re = count_inv * t2_re;
      s2_im = count_inv * t2_im;
      s3_re = count_inv * t3_re;
      s3_im = count_inv * t3_im;
    }

    T::transpose4x4(s0_re, s1_re, s2_re, s3_re);
    T::transpose4x4(s0_im, s1_im, s2_im, s3_im);

    T::store_interleaved((r32*)(out + block_idx + 0*count/2 + 0*count/4), s0_re, s0_im);
    T::store_interleaved((r32*)(out + block_idx + 1*count/2 + 0*count/4), s1_re, s1_im);
    T::store_interleaved((r32*)(out + block_idx + 0*count/2 + 1*count/4), s2_re, s2_im);
    T::store_interleaved((r32*)(out + block_idx + 1*count/2 + 1*count/4), s3_re, s3_im);
  }

  return(T::count);
}

template<typename T, FftDirection dir>
static void
fft_real_convert__radix_2(c64 *io, usz real_count)
{
  usz m = real_count/2;

  // NOTE: DC and Nyquist
  if(dir == FftDirection_forward)
  {
    r32 r = io[0].re;
    r32 i = io[0].im;
    io[0].re = r + i; // DC
    io[0].im = r - i; // Nyquist
  }
  else // FftDirection_inverse
  {
    r32 dc = io[0].re;
    r32 nq = io[0].im;
    io[0].re = 0.5f*(dc + nq);
    io[0].im = 0.5f*(dc - nq);
  }

  // NOTE: passes less than vector width
  {
    r32 *twiddles_im = twiddle_table__radix_2 + 2*m;
    r32 *twiddles_re = twiddles_im + m/2;
    if(dir == FftDirection_forward) // NOTE: -sine for forward transform
    { twiddles_im += m; }

    for(usz k = 1; k < T::count; ++k)
    {
      r32 w_re = twiddles_re[k];
      r32 w_im = twiddles_im[k];

      r32 l_re = io[k].re;
      r32 l_im = io[k].im;
      r32 r_re = io[m - k].re;
      r32 r_im = io[m - k].im;

      r32 out0_re, out0_im, out1_re, out1_im;
      if(dir == FftDirection_forward)
      {
	r32 in0_re = 0.5f*(l_re + r_re);
	r32 in0_im = 0.5f*(l_im - r_im);
	r32 in1_re = 0.5f*(l_im + r_im);
	r32 in1_im = 0.5f*(r_re - l_re);

	out0_re = in0_re + w_re*in1_re - w_im*in1_im;
	out0_im = in0_im + w_re*in1_im + w_im*in1_re;
	out1_re = in0_re - w_re*in1_re + w_im*in1_im;
	out1_im = w_re*in1_im + w_im*in1_re - in0_im;
      }
      else // dir == FftDirection_inverse
      {
	r32 in0_re = 0.5f*(l_re + r_re);
	r32 in0_im = 0.5f*(l_im - r_im);
	r32 in1_re = 0.5f*(l_re - r_re);
	r32 in1_im = 0.5f*(l_im + r_im);

	out0_re = in0_re - w_re*in1_im - w_im*in1_re;
	out0_im = in0_im + w_re*in1_re - w_im*in1_im;
	out1_re = in0_re + w_re*in1_im + w_im*in1_re;
	out1_im = w_re*in1_re - w_im*in1_im - in0_im;
      }

      io[k].re = out0_re;
      io[k].im = out0_im;
      io[m - k].re = out1_re;
      io[m - k].im = out1_im;
    }
  }

  // NOTE: inner values
  {
#if 1
    T half(0.5f);
    r32 *twiddles_im = twiddle_table__radix_2 + 2*m;
    r32 *twiddles_re = twiddles_im + m/2;
    if(dir == FftDirection_forward) // NOTE: -sine for forward transform
    { twiddles_im += m; }

    for(usz k = T::count; k < m/2; k += T::count) // TODO: mask?
    {
      // NOTE: extract even and odd parts of
      // io[k] = x_e[k] + i*x_o[k].
      //
      // we exploit conjugate symmetry:
      // x_e[k] = x_e[m - k]*,
      // x_o[k] = x_o[m - k]*,
      // io[m - k]* = x_e[m - k]* - i*x_o[m - k]*
      //            = x_e[k] - i*x_o[k],
      // x_e[k] = 0.5 * (io[k] + io[m - k]*),
      // x_o[k] = 0.5 * (io[k] - io[m-k]*) / i.
      T l_re, l_im;
      T::load_deinterleave(l_re, l_im, (r32*)(io + k));

      T r_re, r_im;
      T::load_deinterleave(r_re, r_im, (r32*)(io + m - k - (T::count - 1)));
      T::reverse(r_re);
      T::reverse(r_im);

      T w_re = T::load(twiddles_re + k);
      T w_im = T::load(twiddles_im + k);

      T out0_re, out0_im, out1_re, out1_im;
      if(dir == FftDirection_forward)
      {
	// x_e[k] = (io[k] + io[m - k]*) / 2
	T in0_re = half*(l_re + r_re);
	T in0_im = half*(l_im - r_im);
	// x_o[k] = (io[k] - io[m - k]*) / 2i
	T in1_re = half*(l_im + r_im);
	T in1_im = half*(r_re - l_re);

	// NOTE: radix-2 merge step:
	// x[k] = x_e[k] + w^k*x_o[k]
	out0_re = in0_re + w_re*in1_re - w_im*in1_im;
	out0_im = in0_im + w_re*in1_im + w_im*in1_re;
	// x[m - k] = x[n - (m + k)] = x[m + k]* = (x_e[k] - w^k*x_o[k])*
	out1_re = in0_re - w_re*in1_re + w_im*in1_im;
	out1_im = w_re*in1_im + w_im*in1_re - in0_im;

	// if(k == 0)
	// {
	//   T::template overwrite<0, 0>(out0_re, l_re + l_im);
	//   T::template overwrite<0, 0>(out0_im, l_re - l_im);
	// }
      }
      else // dir == FftDirection_inverse
      {
	// NOTE: inverse of the forward direction.
	//
	// x_e[k] = (io[k] + io[m - k]*)/2
	// x_o[k] = (io[k] - io[m - k]*)/2 * w^k*
	//
	// x[k] = x_e[k] + i*x_o[k]
	// x[m - k] = (x_e[k] - i*x_o[k])*

	T in0_re = half*(l_re + r_re);
	T in0_im = half*(l_im - r_im);
	T in1_re = half*(l_re - r_re);
	T in1_im = half*(l_im + r_im);

	out0_re = in0_re - w_re*in1_im - w_im*in1_re;
	out0_im = in0_im + w_re*in1_re - w_im*in1_im;
	out1_re = in0_re + w_re*in1_im + w_im*in1_re;
	out1_im = w_re*in1_re - w_im*in1_im - in0_im;

	// if(k == 0)
	// {
	//   T::template overwrite<0, 0>(out0_re, half*(l_re + l_im));
	//   T::template overwrite<0, 0>(out0_im, half*(l_re - l_im));
	// }
      }
      T::reverse(out1_re);
      T::reverse(out1_im);

      T::store_interleaved((r32*)(io + k), out0_re, out0_im);
      T::store_interleaved((r32*)(io + m - k - (T::count - 1)), out1_re, out1_im);
    }
#else
    r32 *twiddles_re = twiddle_table__radix_2 + 2*m;
    r32 *twiddles_im = twiddles_re + m/2;
    for(usz k = 1; k < m/2; ++k)
    {
      c64 l = io[k];
      c64 r = io[m - k];
      c64 in0 = 0.5f*(l + conjugateC64(r));
      c64 in1 = 0.5f*(l - conjugateC64(r))/C64(0, 1);

      c64 w = C64(twiddles_re[k], twiddles_im[k]);
      c64 out0 = in0 + w*in1;
      c64 out1 = conjugateC64(in0 - w*in1);
      io[k] = out0;
      io[m - k] = out1;
    }
#endif
  }
}

struct FftKernels
{
  usz (*fft_initial)(c64 *out, c64 *in, usz count);
  void (*fft_kernel)(c64 *io, usz level, usz count);
  void (*fft_real_convert)(c64 *io, usz real_count);

  void (*ifft_real_convert)(c64 *io, usz real_count);
  usz (*ifft_initial)(c64 *out, c64 *in, usz count);
  void (*ifft_kernel)(c64 *io, usz level, usz count);
};

static FftKernels fft_kernels = {};

static void
fftInitKernels(void)
{
  // TODO: check cpu features to determine which element type to use
  fft_kernels.fft_initial       = &fft_initial__radix_2<R32x4, FftDirection_forward>;
  fft_kernels.fft_kernel        = &fft_kernel__radix_2<R32x4, FftDirection_forward>;
  fft_kernels.fft_real_convert  = &fft_real_convert__radix_2<R32x4, FftDirection_forward>;

  fft_kernels.ifft_real_convert = &fft_real_convert__radix_2<R32x4, FftDirection_inverse>;
  fft_kernels.ifft_initial      = &fft_initial__radix_2<R32x4, FftDirection_inverse>;
  fft_kernels.ifft_kernel       = &fft_kernel__radix_2<R32x4, FftDirection_inverse>;
}

/** Computes DFT of real input.
 * Reads `count` real samples from `in` and writes `count/2` complex values to `out` (`count` floats).
 * `out[0].re` = DC, `out[0].im` = Nyquist.
 * Other half of the spectrum is the complex conjugate of the half that gets returned (excluding the endpoints).
 */
static void
fft_re(c64 *out, r32 *in, usz count, FftKernels *kernels)
{
  usz const initial_level = kernels->fft_initial(out, (c64*)in, count/2);
  for(usz level = initial_level; level < count/2; level *= 2)
  {
    kernels->fft_kernel(out, level, count/2);
  }

  // DEBUG:
  TemporaryMemory scratch = arenaGetScratch(0, 0);
  String8List debugLogList = {};
  stringListPush(scratch.arena, &debugLogList, STR8_LIT("fft out pre real-conversion:\n"));
  for(usz i = 0; i < count/2; ++i)
  {
    stringListPushFormat(scratch.arena, &debugLogList,
			 "[%3lu] = %10.4f + %10.4fi\n",
			 i, out[i].re, out[i].im);
  }
  String8 debugLog = stringListJoin(scratch.arena, &debugLogList, STR8_LIT(""));
  Buffer debugLogBuffer = {};
  debugLogBuffer.size = debugLog.size;
  debugLogBuffer.contents = debugLog.str;
  gsWriteEntireFile(DATA_PATH"test/fft_re_conversion_test.txt", debugLogBuffer);
  arenaReleaseScratch(scratch);

  kernels->fft_real_convert(out, count);
}

/** Computes inverse DFT of half-spectrum of the form above, producing real output.
 */
static void
ifft_re(r32 *out, c64 *in, usz count, FftKernels *kernels)
{
  kernels->ifft_real_convert(in, count);

  // DEBUG:
  TemporaryMemory scratch = arenaGetScratch(0, 0);
  String8List debugLogList = {};
  stringListPush(scratch.arena, &debugLogList, STR8_LIT("ifft in post real-conversion:\n"));
  for(usz i = 0; i < count/2; ++i)
  {
    stringListPushFormat(scratch.arena, &debugLogList,
			 "[%3lu] = %10.4f + %10.4fi\n",
			 i, in[i].re, in[i].im);
  }
  String8 debugLog = stringListJoin(scratch.arena, &debugLogList, STR8_LIT(""));
  Buffer debugLogBuffer = {};
  debugLogBuffer.size = debugLog.size;
  debugLogBuffer.contents = debugLog.str;
  gsWriteEntireFile(DATA_PATH"test/ifft_re_conversion_test.txt", debugLogBuffer);
  arenaReleaseScratch(scratch);

  usz const initial_level = kernels->ifft_initial((c64*)out, in, count/2);
  for(usz level = initial_level; level < count/2; level *= 2)
  {
    kernels->ifft_kernel((c64*)out, level, count/2);
  }
}

// static ComplexPair*
// fft(Arena *arena, SamplePair *samples, u64 sampleCount)
// {
//   TemporaryMemory scratch = arenaGetScratch(&arena, 1);
//   r32 *srcSamplesL = arenaPushArray(scratch.arena, 2*sampleCount, r32);
//   r32 *srcSamplesR = srcSamplesL + sampleCount;
//   // TODO: vectorize
//   for(u32 i = 0; i < sampleCount; ++i)
//   {
//     srcSamplesL[i] = samples[i].left;
//     srcSamplesR[i] = samples[i].right;
//   }
//   FloatBuffer srcL = makeFloatBuffer(srcSamplesL, sampleCount);
//   FloatBuffer srcR = makeFloatBuffer(srcSamplesR, sampleCount);

//   c64 *destBinsL = arenaPushArray(scratch.arena, 2*sampleCount, c64);
//   c64 *destBinsR = destBinsL + sampleCount;
//   ComplexBuffer destL = makeComplexBuffer(destBinsL, sampleCount);
//   ComplexBuffer destR = makeComplexBuffer(destBinsR, sampleCount);
//   fft_core(&destL, srcL);
//   fft_core(&destR, srcR);

//   ComplexPair *result = arenaPushArray(arena, sampleCount, ComplexPair);
//   // TODO: vectorize
//   for(u32 i = 0; i < sampleCount; ++i)
//   {
//     result[i].left  = destL.vals[i];
//     result[i].right = destR.vals[i];
//   }

//   arenaReleaseScratch(scratch);
//   return(result);
// }

// static SamplePair*
// ifft(Arena *arena, ComplexPair *spectrum, u64 binCount)
// {
//   SamplePair *result = arenaPushArray(arena, binCount, SamplePair);
//   c64 *srcBinsL = &spectrum->left;
//   c64 *srcBinsR = &spectrum->right;
//   ComplexBuffer srcL = makeComplexBuffer(srcBinsL, binCount, sizeof(*spectrum));
//   ComplexBuffer srcR = makeComplexBuffer(srcBinsR, binCount, sizeof(*spectrum));
//   r32 *destSamplesL = &result->left;
//   r32 *destSamplesR = &result->right;
//   FloatBuffer destL = makeFloatBuffer(destSamplesL, binCount, sizeof(*result));
//   FloatBuffer destR = makeFloatBuffer(destSamplesR, binCount, sizeof(*result));
//   ifft_core(&destL, srcL);
//   ifft_core(&destR, srcR);
//   return(result);
// }

static FFT_FUNCTION(fft_dit_radix2_scalar)
{
  PROFILE_FUNCTION();

  usz count = ROUND_UP_POW_2(input.count);
  r32 *reVals = arenaPushArray(arena, count, r32);
  r32 *imVals = arenaPushArray(arena, count, r32);
  // c64 *vals = arenaPushArray(arena, count, c64);

  // NOTE: input permutation
  {
    PROFILE_BLOCK("fft_dit_radix2_scalar:input_permutation");

    usz countLog2 = LOG2(count);
    r32 *src = input.vals;
    for(u32 i = 0; i < count; ++i)
      {
        u32 iRev = reverseBits(i) >> (sizeof(u32)*8 - countLog2);
        reVals[iRev] = src[i];
        // vals[iRev].re = src[i];
      }
  }

  // NOTE: twiddles
  {
    PROFILE_BLOCK("fft_dit_radix2_scalar:twiddles");

    for(u32 m = 2; m <= count; m <<= 1)
      {
        r32 theta = -2.f * GS_PI / (r32)m;
        r32 wmRe = gsCos(theta);
        r32 wmIm = gsSin(theta);

        for(u32 k = 0; k < count; k += m)
          {
            r32 wRe = 1.f;
            r32 wIm = 0.f;
            r32 *at0Re = reVals + k;
            r32 *at0Im = imVals + k;
            r32 *at1Re = reVals + k + m/2;
            r32 *at1Im = imVals + k + m/2;
            for(u32 j = 0; j < m/2; ++j)
              {
                PROFILE_BLOCK("fft_dit_radix2_scalar:kernel");

                r32 in0Re = *at0Re;
                r32 in0Im = *at0Im;
                r32 in1Re = *at1Re;
                r32 in1Im = *at1Im;

                r32 tRe = wRe*in1Re - wIm*in1Im;
                r32 tIm = wRe*in1Im + wIm*in1Re;

                r32 out0Re = in0Re + tRe;
                r32 out0Im = in0Im + tIm;
                r32 out1Re = in0Re - tRe;
                r32 out1Im = in0Im - tIm;

                *at0Re++ = out0Re;
                *at0Im++ = out0Im;
                *at1Re++ = out1Re;
                *at1Im++ = out1Im;

                r32 wOldRe = wRe;
                r32 wOldIm = wIm;
                wRe = wOldRe*wmRe - wOldIm*wmIm;
                wIm = wOldRe*wmIm + wOldIm*wmRe;
              }
          }
      }
  }

  ComplexBuffer result = {};
  result.count = count;
  result.reVals = reVals;
  result.imVals = imVals;
  return(result);
}

static FFT_FUNCTION(fft_dit_radix2_simd)
{
  PROFILE_FUNCTION();

  u32 simdWidth = 4; // TODO: make this a constant set in `simd_intrinsics.h`

  usz count = ROUND_UP_POW_2(input.count);
  r32 *reVals = arenaPushArray(arena, count, r32, arenaFlagsNoZeroAlign(simdWidth * sizeof(r32)));
  r32 *imVals = arenaPushArray(arena, count, r32, arenaFlagsNoZeroAlign(simdWidth * sizeof(r32)));

  // NOTE: input permutation
  // TODO: simd-ize
  {
    PROFILE_BLOCK("fft_dit_radix2_simd:input_permutation");

    usz countLog2 = LOG2(count);
    r32 *src = input.vals;
    for(u32 i = 0; i < count; ++i)
      {
        u32 iRev = reverseBits(i) >> (sizeof(u32)*8 - countLog2);
        reVals[iRev] = src[i];
      }
  }

  // NOTE: twiddles
  {
    PROFILE_BLOCK("fft_dit_radix2_simd:twiddles");

    for(u32 m = 2; m <= count; m <<= 1)
      {
        r32 theta = -2.f * GS_PI / (r32)m;
        c64 wm = C64Polar(1.f, theta);
        c64 wmSq = wm*wm;
        c64 wmCu = wmSq*wm;
        c64 wmQu = wmCu*wm;
        WideFloat wmQuRe = wideSetConstantFloats(wmQu.re);
        WideFloat wmQuIm = wideSetConstantFloats(wmQu.im);

        for(u32 k = 0; k < count; k += m)
          {
            WideFloat wRe = wideSetFloats(1.f, wm.re, wmSq.re, wmCu.re);
            WideFloat wIm = wideSetFloats(0.f, wm.im, wmSq.im, wmCu.im);

            r32 *at0Re = reVals + k;
            r32 *at0Im = imVals + k;
            r32 *at1Re = reVals + k + m/2;
            r32 *at1Im = imVals + k + m/2;
            for(u32 j = 0; j < m/2; j += simdWidth)
              {
                PROFILE_BLOCK("fft_dit_radix2_simd:kernel");

                WideInt storeMask = wideSetConstantInts(0);
                for(u32 lane = 0; lane < simdWidth; ++lane)
                  {
                    u32 val = (j + lane < m/2) ? 0xFFFFFFFF : 0;
                    wideSetLaneInts(&storeMask, val, lane);
                  }

                WideFloat in0Re = wideLoadFloats(at0Re);
                WideFloat in0Im = wideLoadFloats(at0Im);
                WideFloat in1Re = wideLoadFloats(at1Re);
                WideFloat in1Im = wideLoadFloats(at1Im);

                WideFloat tRe = wRe*in1Re - wIm*in1Im;
                WideFloat tIm = wRe*in1Im + wIm*in1Re;

                WideFloat out0Re = in0Re + tRe;
                WideFloat out0Im = in0Im + tIm;
                WideFloat out1Re = in0Re - tRe;
                WideFloat out1Im = in0Im - tIm;

                WideFloat store0Re = wideMaskFloats(out0Re, in0Re, storeMask);
                WideFloat store0Im = wideMaskFloats(out0Im, in0Im, storeMask);
                WideFloat store1Re = wideMaskFloats(out1Re, in1Re, storeMask);
                WideFloat store1Im = wideMaskFloats(out1Im, in1Im, storeMask);

                wideStoreFloats(at0Re, store0Re);
                wideStoreFloats(at0Im, store0Im);
                wideStoreFloats(at1Re, store1Re);
                wideStoreFloats(at1Im, store1Im);

                WideFloat wOldRe = wRe;
                WideFloat wOldIm = wIm;
                wRe = wOldRe*wmQuRe - wOldIm*wmQuIm;
                wIm = wOldRe*wmQuIm + wOldIm*wmQuRe;

                at0Re += simdWidth;
                at0Im += simdWidth;
                at1Re += simdWidth;
                at1Im += simdWidth;
              }
          }
      }
  }

  ComplexBuffer result = {};
  result.count = count;
  result.reVals = reVals;
  result.imVals = imVals;
  return(result);
}

static IFFT_FUNCTION(ifft_dit_radix2_scalar)
{
  PROFILE_FUNCTION();

  usz count = ROUND_UP_POW_2(input.count);
  r32 *reVals = arenaPushArray(arena, count, r32);
  r32 *imVals = arenaPushArray(arena, count, r32);

  // NOTE: input permutation
  {
    PROFILE_BLOCK("ifft_dit_radix_2_scalar:input_permutation");

    usz countLog2 = LOG2(count);
    r32 invCount = 1.f / (r32)count;
    r32 *srcRe = input.reVals;
    r32 *srcIm = input.imVals;
    for(u32 i = 0; i < count; ++i)
      {
        u32 iRev = reverseBits(i) >> (sizeof(u32)*8 - countLog2);
        reVals[iRev] = invCount * srcRe[i];
        imVals[iRev] = invCount * srcIm[i];
      }
  }

  // NOTE: twiddles
  {
    PROFILE_BLOCK("ifft_dit_radix_2_scalar:twiddles");

    for(u32 m = 2; m <= count; m <<= 1)
      {
        r32 theta = 2.f * GS_PI / (r32)m;
        r32 wmRe = gsCos(theta);
        r32 wmIm = gsSin(theta);

        for(u32 k = 0; k < count; k += m)
          {
            r32 wRe = 1.f;
            r32 wIm = 0.f;
            r32 *at0Re = reVals + k;
            r32 *at0Im = imVals + k;
            r32 *at1Re = reVals + k + m/2;
            r32 *at1Im = imVals + k + m/2;
            for(u32 j = 0; j < m/2; ++j)
              {
                PROFILE_BLOCK("ifft_dit_radix_2_scalar:kernel");

                r32 in0Re = *at0Re;
                r32 in0Im = *at0Im;
                r32 in1Re = *at1Re;
                r32 in1Im = *at1Im;

                r32 tRe = wRe*in1Re - wIm*in1Im;
                r32 tIm = wRe*in1Im + wIm*in1Re;

                r32 out0Re = in0Re + tRe;
                r32 out0Im = in0Im + tIm;
                r32 out1Re = in0Re - tRe;
                r32 out1Im = in0Im - tIm;

                *at0Re++ = out0Re;
                *at0Im++ = out0Im;
                *at1Re++ = out1Re;
                *at1Im++ = out1Im;

                r32 wOldRe = wRe;
                r32 wOldIm = wIm;
                wRe = wOldRe*wmRe - wOldIm*wmIm;
                wIm = wOldRe*wmIm + wOldIm*wmRe;
              }
          }
      }
  }

  FloatBuffer result = {};
  result.count = count;
  result.vals = reVals;
  return(result);
}

static IFFT_FUNCTION(ifft_dit_radix2_simd)
{
  PROFILE_FUNCTION();

  u32 simdWidth = 4; // TODO: make this a constant set in `simd_intrinsics.h`

  usz count = ROUND_UP_POW_2(input.count);
  r32 *reVals = arenaPushArray(arena, count, r32, arenaFlagsNoZeroAlign(simdWidth * sizeof(r32)));
  r32 *imVals = arenaPushArray(arena, count, r32, arenaFlagsNoZeroAlign(simdWidth * sizeof(r32)));

  // NOTE: input permutation
  // TODO: simd-ize
  {
    PROFILE_BLOCK("ifft_dit_radix2_simd:input_permutation");

    usz countLog2 = LOG2(count);
    r32 invCount = 1.f / (r32)count;
    r32 *srcRe = input.reVals;
    r32 *srcIm = input.imVals;
    for(u32 i = 0; i < count; ++i)
      {
        u32 iRev = reverseBits(i) >> (sizeof(u32)*8 - countLog2);
        reVals[iRev] = invCount * srcRe[i];
        imVals[iRev] = invCount * srcIm[i];
      }
  }

  // NOTE: twiddles
  {
    PROFILE_BLOCK("ifft_dit_radix2_simd:twiddles");

    for(u32 m = 2; m <= count; m <<= 1)
      {
        r32 theta = 2.f * GS_PI / (r32)m;
        c64 wm = C64Polar(1.f, theta);
        c64 wmSq = wm*wm;
        c64 wmCu = wmSq*wm;
        c64 wmQu = wmCu*wm;
        WideFloat wmQuRe = wideSetConstantFloats(wmQu.re);
        WideFloat wmQuIm = wideSetConstantFloats(wmQu.im);

        for(u32 k = 0; k < count; k += m)
          {
            WideFloat wRe = wideSetFloats(1.f, wm.re, wmSq.re, wmCu.re);
            WideFloat wIm = wideSetFloats(0.f, wm.im, wmSq.im, wmCu.im);

            r32 *at0Re = reVals + k;
            r32 *at0Im = imVals + k;
            r32 *at1Re = reVals + k + m/2;
            r32 *at1Im = imVals + k + m/2;
            for(u32 j = 0; j < m/2; j += simdWidth)
              {
                PROFILE_BLOCK("ifft_dit_radix2_simd:kernel");

                WideInt storeMask = wideSetConstantInts(0);
                for(u32 lane = 0; lane < simdWidth; ++lane)
                  {
                    u32 val = (j + lane < m/2) ? 0xFFFFFFFF : 0;
                    wideSetLaneInts(&storeMask, val, lane);
                  }

                WideFloat in0Re = wideLoadFloats(at0Re);
                WideFloat in0Im = wideLoadFloats(at0Im);
                WideFloat in1Re = wideLoadFloats(at1Re);
                WideFloat in1Im = wideLoadFloats(at1Im);

                WideFloat tRe = wRe*in1Re - wIm*in1Im;
                WideFloat tIm = wRe*in1Im + wIm*in1Re;

                WideFloat out0Re = in0Re + tRe;
                WideFloat out0Im = in0Im + tIm;
                WideFloat out1Re = in0Re - tRe;
                WideFloat out1Im = in0Im - tIm;

                WideFloat store0Re = wideMaskFloats(out0Re, in0Re, storeMask);
                WideFloat store0Im = wideMaskFloats(out0Im, in0Im, storeMask);
                WideFloat store1Re = wideMaskFloats(out1Re, in1Re, storeMask);
                WideFloat store1Im = wideMaskFloats(out1Im, in1Im, storeMask);

                wideStoreFloats(at0Re, store0Re);
                wideStoreFloats(at0Im, store0Im);
                wideStoreFloats(at1Re, store1Re);
                wideStoreFloats(at1Im, store1Im);

                WideFloat wOldRe = wRe;
                WideFloat wOldIm = wIm;
                wRe = wOldRe*wmQuRe - wOldIm*wmQuIm;
                wIm = wOldRe*wmQuIm + wOldIm*wmQuRe;

                at0Re += simdWidth;
                at0Im += simdWidth;
                at1Re += simdWidth;
                at1Im += simdWidth;
              }
          }
      }
  }

  FloatBuffer result = {};
  result.count = count;
  result.vals = reVals;
  return(result);
}

static FFT_Function *fftFunctions[] = {
  fft_dit_radix2_scalar,
  fft_dit_radix2_simd,
};

static IFFT_Function *ifftFunctions[] = {
  ifft_dit_radix2_scalar,
  ifft_dit_radix2_simd,
};

#if 0
#define REAL_FFT_FUNCTION(name) void (name)(r32 *destRe, r32 *destIm, r32 *src, u32 length)
#define REAL_IFFT_FUNCTION(name) void (name)(r32 *dest, r32 *destImTemp, r32 *srcRe, r32 *srcIm, u32 length)

static REAL_FFT_FUNCTION(fft_real_noPermute);
static REAL_IFFT_FUNCTION(ifft_real_noPermute);

static REAL_FFT_FUNCTION(fft_real_permute);
static REAL_IFFT_FUNCTION(ifft_real_permute);

static REAL_FFT_FUNCTION(fft_real_permute)
{
  ASSERT(isPowerOf2(length));
  u32 logLength = log2(length);
  u32 simdWidth = 4;

  // TODO: simd-ize this
  u32 lengthSizeInBits = 8*sizeof(length);
  for(u32 index = 0; index < length; ++index)
    {
      u32 reversedIndex = reverseBits(index);
      reversedIndex >>= lengthSizeInBits - logLength;
      ASSERT(reversedIndex < length);

      destRe[index] = src[reversedIndex];
    }

  u32 mask = 0xFFFFFFFF;
  //r32 maskF = *(r32 *)&mask;

  for(u32 m = 2; m <= length; m <<= 1)
    {
      r32 angle = -GS_TAU/(r32)m;

      // TODO: make trig ops fast
      c64 wm = C64Polar(1.f, angle);
      // TODO: maybe do these multiplies wide as well?
      c64 wmSq = wm*wm;
      c64 wmCu = wm*wmSq;
      c64 wmQu = wm*wmCu;
      WideFloat wmQuRe = wideSetConstantFloats(wmQu.re);
      WideFloat wmQuIm = wideSetConstantFloats(wmQu.im);

      for(u32 k = 0; k < length; k += m)
        {
          WideFloat wRe = wideSetFloats(1.f, wm.re, wmSq.re, wmCu.re);
          WideFloat wIm = wideSetFloats(0.f, wm.im, wmSq.im, wmCu.im);

          r32 *dest0Re = destRe + k;
          r32 *dest0Im = destIm + k;
          r32 *dest1Re = destRe + k + m/2;
          r32 *dest1Im = destIm + k + m/2;
          for(u32 j = 0; j < m/2; j += simdWidth)
            {
              WideInt storeMask = wideSetConstantInts(0);
              //WideFloat storeMaskInv = wideSetConstantFloats(0.f);
              for(u32 lane = 0; lane < simdWidth; ++lane)
                {
                  //r32 val = (j + lane < m/2) ? maskF : 0;
                  //r32 valInv = (j + lane < m/2) ? 0 : maskF;
                  u32 val = (j + lane < m/2) ? mask : 0;

                  wideSetLaneInts(&storeMask, val, lane);
                  //wideSetLaneFloats(&storeMaskInv, valInv, lane);
                }

              // NOTE: loads
              WideFloat in0Re = wideLoadFloats(dest0Re);
              WideFloat in0Im = wideLoadFloats(dest0Im);
              WideFloat in1Re = wideLoadFloats(dest1Re);
              WideFloat in1Im = wideLoadFloats(dest1Im);

              // NOTE: butterflies and twiddles
              WideFloat productRe = wideSubFloats(wideMulFloats(wRe, in1Re),
                                                  wideMulFloats(wIm, in1Im));
              WideFloat productIm = wideAddFloats(wideMulFloats(wRe, in1Im),
                                                  wideMulFloats(wIm, in1Re));
              WideFloat out0Re = wideAddFloats(in0Re, productRe);
              WideFloat out0Im = wideAddFloats(in0Im, productIm);
              WideFloat out1Re = wideSubFloats(in0Re, productRe);
              WideFloat out1Im = wideSubFloats(in0Im, productIm);

              // NOTE: mask out what not to store
              /* WideFloat store0Re = wideAddFloats(wideMaskFloats(out0Re, storeMask), */
              /*                                         wideMaskFloats(in0Re, storeMaskInv)); */
              /* WideFloat store0Im = wideAddFloats(wideMaskFloats(out0Im, storeMask), */
              /*                                         wideMaskFloats(in0Im, storeMaskInv)); */
              /* WideFloat store1Re = wideAddFloats(wideMaskFloats(out1Re, storeMask), */
              /*                                         wideMaskFloats(in1Re, storeMaskInv)); */
              /* WideFloat store1Im = wideAddFloats(wideMaskFloats(out1Im, storeMask), */
              /*                                         wideMaskFloats(in1Im, storeMaskInv)); */

              WideFloat store0Re = wideMaskFloats(out0Re, in0Re, storeMask);
              WideFloat store0Im = wideMaskFloats(out0Im, in0Im, storeMask);
              WideFloat store1Re = wideMaskFloats(out1Re, in1Re, storeMask);
              WideFloat store1Im = wideMaskFloats(out1Im, in1Im, storeMask);

              // NOTE: do the store
              wideStoreFloats(dest0Re, store0Re);
              wideStoreFloats(dest0Im, store0Im);
              wideStoreFloats(dest1Re, store1Re);
              wideStoreFloats(dest1Im, store1Im);

              WideFloat wReOld = wRe;
              WideFloat wImOld = wIm;
              wRe = wideSubFloats(wideMulFloats(wReOld, wmQuRe),
                                  wideMulFloats(wImOld, wmQuIm));
              wIm = wideAddFloats(wideMulFloats(wReOld, wmQuIm),
                                  wideMulFloats(wImOld, wmQuRe));

              dest0Re += simdWidth;
              dest0Im += simdWidth;
              dest1Re += simdWidth;
              dest1Im += simdWidth;
            }
        }
    }
}

static REAL_IFFT_FUNCTION(ifft_real_permute)
{
  ASSERT(isPowerOf2(length));
  u32 logLength = log2(length);
  r32 invLength = 1.f/(r32)length;
  u32 simdWidth = 4;

  // TODO: simd-ize this
  u32 lengthSizeInBits = 8*sizeof(length);
  for(u32 index = 0; index < length; ++index)
    {
      u32 reversedIndex = reverseBits(index);
      reversedIndex >>= lengthSizeInBits - logLength;
      ASSERT(reversedIndex < length);

      dest[index] = invLength*srcRe[reversedIndex];
      destImTemp[index] = invLength*srcIm[reversedIndex];
    }

  u32 mask = 0xFFFFFFFF;
  //  r32 maskF = *(r32 *)&mask;

  for(u32 m = 2; m <= length; m <<= 1)
    {
      r32 angle = GS_TAU/(r32)m;

      // TODO: make trig ops fast
      c64 wm = C64Polar(1.f, angle);
      // TODO: maybe do these multiplies wide as well?
      c64 wmSq = wm*wm;
      c64 wmCu = wm*wmSq;
      c64 wmQu = wm*wmCu;
      WideFloat wmQuRe = wideSetConstantFloats(wmQu.re);
      WideFloat wmQuIm = wideSetConstantFloats(wmQu.im);

      for(u32 k = 0; k < length; k += m)
        {
          WideFloat wRe = wideSetFloats(1.f, wm.re, wmSq.re, wmCu.re);
          WideFloat wIm = wideSetFloats(0.f, wm.im, wmSq.im, wmCu.im);

          r32 *dest0Re = dest + k;
          r32 *dest0Im = destImTemp + k;
          r32 *dest1Re = dest + k + m/2;
          r32 *dest1Im = destImTemp + k + m/2;
          for(u32 j = 0; j < m/2; j += simdWidth)
            {
              WideInt storeMask = wideSetConstantInts(0.f);
              //WideFloat storeMaskInv = wideSetConstantFloats(0.f);
              for(u32 lane = 0; lane < simdWidth; ++lane)
                {
                  //r32 val = (j + lane < m/2) ? maskF : 0;
                  //r32 valInv = (j + lane < m/2) ? 0 : maskF;
                  u32 val = (j + lane < m/2) ? mask : 0;

                  wideSetLaneInts(&storeMask, val, lane);
                  //wideSetLaneFloats(&storeMaskInv, valInv, lane);
                }

              // NOTE: loads
              WideFloat in0Re = wideLoadFloats(dest0Re);
              WideFloat in0Im = wideLoadFloats(dest0Im);
              WideFloat in1Re = wideLoadFloats(dest1Re);
              WideFloat in1Im = wideLoadFloats(dest1Im);

              // NOTE: butterflies and twiddles
              WideFloat productRe = wideSubFloats(wideMulFloats(wRe, in1Re),
                                                  wideMulFloats(wIm, in1Im));
              WideFloat productIm = wideAddFloats(wideMulFloats(wRe, in1Im),
                                                  wideMulFloats(wIm, in1Re));
              WideFloat out0Re = wideAddFloats(in0Re, productRe);
              WideFloat out0Im = wideAddFloats(in0Im, productIm);
              WideFloat out1Re = wideSubFloats(in0Re, productRe);
              WideFloat out1Im = wideSubFloats(in0Im, productIm);

              // NOTE: mask out what not to store
              /* WideFloat store0Re = wideAddFloats(wideMaskFloats(out0Re, storeMask), */
              /*                                         wideMaskFloats(in0Re, storeMaskInv)); */
              /* WideFloat store0Im = wideAddFloats(wideMaskFloats(out0Im, storeMask), */
              /*                                         wideMaskFloats(in0Im, storeMaskInv)); */
              /* WideFloat store1Re = wideAddFloats(wideMaskFloats(out1Re, storeMask), */
              /*                                         wideMaskFloats(in1Re, storeMaskInv)); */
              /* WideFloat store1Im = wideAddFloats(wideMaskFloats(out1Im, storeMask), */
              /*                                         wideMaskFloats(in1Im, storeMaskInv)); */

              WideFloat store0Re = wideMaskFloats(out0Re, in0Re, storeMask);
              WideFloat store0Im = wideMaskFloats(out0Im, in0Im, storeMask);
              WideFloat store1Re = wideMaskFloats(out1Re, in1Re, storeMask);
              WideFloat store1Im = wideMaskFloats(out1Im, in1Im, storeMask);

              // NOTE: do the store
              wideStoreFloats(dest0Re, store0Re);
              wideStoreFloats(dest0Im, store0Im);
              wideStoreFloats(dest1Re, store1Re);
              wideStoreFloats(dest1Im, store1Im);

              WideFloat wReOld = wRe;
              WideFloat wImOld = wIm;
              wRe = wideSubFloats(wideMulFloats(wReOld, wmQuRe),
                                  wideMulFloats(wImOld, wmQuIm));
              wIm = wideAddFloats(wideMulFloats(wReOld, wmQuIm),
                                  wideMulFloats(wImOld, wmQuRe));

              dest0Re += simdWidth;
              dest0Im += simdWidth;
              dest1Re += simdWidth;
              dest1Im += simdWidth;
            }
        }
    }
}

static REAL_FFT_FUNCTION(fft_real_noPermute)
{
  // NOTE: radix-2 dif fft
  ASSERT(isPowerOf2(length));
  //u32 logLength = log2(length);
  u32 simdWidth = 4;

  // NOTE: copy input
  r32 *read = src;
  r32 *write = destRe;
  for(u32 i = 0; i < length; i += simdWidth)
    {
      //WideFloat in = wideLoadFloats(src);
      WideFloat in = wideLoadFloats(read);
      wideStoreFloats(write, in);

      read += simdWidth;
      write += simdWidth;
    }

  //u32 mask = 0xFFFFFFFF;
  //r32 maskF = *(r32 *)&mask;
  //for(u32 m = length; m >= 2; m >>= 1)
  for(u32 m = length; m >= 2*simdWidth; m >>= 1)
    {
      r32 angle = -GS_TAU/(r32)m;

      c64 wm = C64Polar(1.f, angle);
      c64 wmSq = wm*wm;
      c64 wmCu = wm*wmSq;
      c64 wmQu = wm*wmCu;
      WideFloat wmQuRe = wideSetConstantFloats(wmQu.re);
      WideFloat wmQuIm = wideSetConstantFloats(wmQu.im);

      for(u32 k = 0; k < length; k += m)
        {
          WideFloat wRe = wideSetFloats(1.f, wm.re, wmSq.re, wmCu.re);
          WideFloat wIm = wideSetFloats(0.f, wm.im, wmSq.im, wmCu.im);

          r32 *dest0Re = destRe + k;
          r32 *dest0Im = destIm + k;
          r32 *dest1Re = destRe + k + m/2;
          r32 *dest1Im = destIm + k + m/2;
          for(u32 j = 0; j < m/2; j += simdWidth)
            {
              /* WideFloat storeMask = wideSetConstantFloats(0.f); */
              /* WideFloat storeMaskInv = wideSetConstantFloats(0.f); */
              /* for(u32 lane = 0; lane < simdWidth; ++lane) */
              /*        { */
              /*          r32 val = (j + lane < m/2) ? maskF : 0.f; */
              /*          r32 valInv = (j + lane < m/2) ? 0.f : maskF; */

              /*          wideSetLaneFloats(&storeMask, val, lane); */
              /*          wideSetLaneFloats(&storeMaskInv, valInv, lane); */
              /*        } */

              WideFloat in0Re = wideLoadFloats(dest0Re);
              WideFloat in0Im = wideLoadFloats(dest0Im);
              WideFloat in1Re = wideLoadFloats(dest1Re);
              WideFloat in1Im = wideLoadFloats(dest1Im);

              // NOTE: butterflies/twiddles
              WideFloat out0Re = wideAddFloats(in0Re, in1Re);
              WideFloat out0Im = wideAddFloats(in0Im, in1Im);
              WideFloat diffRe = wideSubFloats(in0Re, in1Re);
              WideFloat diffIm = wideSubFloats(in0Im, in1Im);
              WideFloat out1Re = wideSubFloats(wideMulFloats(diffRe, wRe),
                                               wideMulFloats(diffIm, wIm));
              WideFloat out1Im = wideAddFloats(wideMulFloats(diffRe, wIm),
                                               wideMulFloats(diffIm, wRe));

              /* WideFloat store0Re = wideAddFloats(wideMaskFloats(out0Re, storeMask), */
              /*                                         wideMaskFloats(in0Re, storeMaskInv)); */
              /* WideFloat store0Im = wideAddFloats(wideMaskFloats(out0Im, storeMask), */
              /*                                         wideMaskFloats(in0Im, storeMaskInv)); */
              /* WideFloat store1Re = wideAddFloats(wideMaskFloats(out1Re, storeMask), */
              /*                                         wideMaskFloats(in1Re, storeMaskInv)); */
              /* WideFloat store1Im = wideAddFloats(wideMaskFloats(out1Im, storeMask), */
              /*                                         wideMaskFloats(in1Im, storeMaskInv)); */

              /* wideStoreFloats(dest0Re, store0Re); */
              /* wideStoreFloats(dest0Im, store0Im); */
              /* wideStoreFloats(dest1Re, store1Re); */
              /* wideStoreFloats(dest1Im, store1Im); */
              wideStoreFloats(dest0Re, out0Re);
              wideStoreFloats(dest0Im, out0Im);
              wideStoreFloats(dest1Re, out1Re);
              wideStoreFloats(dest1Im, out1Im);

              WideFloat wReOld = wRe;
              WideFloat wImOld = wIm;
              wRe = wideSubFloats(wideMulFloats(wReOld, wmQuRe),
                                  wideMulFloats(wImOld, wmQuIm));
              wIm = wideAddFloats(wideMulFloats(wReOld, wmQuIm),
                                  wideMulFloats(wImOld, wmQuRe));

              dest0Re += simdWidth;
              dest0Im += simdWidth;
              dest1Re += simdWidth;
              dest1Im += simdWidth;
            }
        }
    }
  for(u32 m = simdWidth; m >= 2; m >>= 1)
    {
      r32 angle = -GS_TAU/(r32)m;
      c64 wm = C64Polar(1.f, angle);
      r32 wmRe = wm.re;
      r32 wmIm = wm.im;

      for(u32 k = 0; k < length; k += m)
        {
          r32 wRe = 1.f;
          r32 wIm = 0.f;

          r32 *dest0Re = destRe + k;
          r32 *dest0Im = destIm + k;
          r32 *dest1Re = destRe + k + m/2;
          r32 *dest1Im = destIm + k + m/2;
          for(u32 j = 0; j < m/2; ++j)
            {
              r32 in0Re = *dest0Re;
              r32 in0Im = *dest0Im;
              r32 in1Re = *dest1Re;
              r32 in1Im = *dest1Im;

              r32 out0Re = in0Re + in1Re;
              r32 out0Im = in0Im + in1Im;
              r32 diffRe = in0Re - in1Re;
              r32 diffIm = in0Im - in1Im;
              r32 out1Re = diffRe*wRe - diffIm*wIm;
              r32 out1Im = diffRe*wIm + diffIm*wRe;

              *dest0Re++ = out0Re;
              *dest0Im++ = out0Im;
              *dest1Re++ = out1Re;
              *dest1Im++ = out1Im;

              r32 wReOld = wRe;
              wRe = wReOld*wmRe - wIm*wmIm;
              wIm = wReOld*wmIm + wIm*wmRe;
            }
        }
    }
}

static REAL_IFFT_FUNCTION(ifft_real_noPermute)
{
  // NOTE: radix-2 dit ifft, without bit-reverse input permutation
  ASSERT(isPowerOf2(length));
  WideFloat invLength = wideSetConstantFloats(1.f/(r32)length);
  u32 simdWidth = 4;

  // NOTE: copy input
  r32 *readRe = srcRe;
  r32 *readIm = srcIm;
  r32 *writeRe = dest;
  r32 *writeIm = destImTemp;
  for(u32 i = 0; i < length; i += simdWidth)
    {
      WideFloat inRe = wideLoadFloats(readRe);
      WideFloat inIm = wideLoadFloats(readIm);

      wideStoreFloats(writeRe, wideMulFloats(invLength, inRe));
      wideStoreFloats(writeIm, wideMulFloats(invLength, inIm));

      readRe += simdWidth;
      readIm += simdWidth;
      writeRe += simdWidth;
      writeIm += simdWidth;
    }

  //u32 mask = 0xFFFFFFFF;
  //r32 maskF = *(r32 *)&mask;
  for(u32 m = 2; m <= simdWidth; m <<= 1)
    {
      r32 angle = GS_TAU/(r32)m;
      c64 wm = C64Polar(1.f, angle);
      r32 wmRe = wm.re;
      r32 wmIm = wm.im;

      for(u32 k = 0; k < length; k += m)
        {
          r32 wRe = 1.f;
          r32 wIm = 0.f;

          r32 *dest0Re = dest + k;
          r32 *dest0Im = destImTemp + k;
          r32 *dest1Re = dest + k + m/2;
          r32 *dest1Im = destImTemp + k + m/2;
          for(u32 j = 0; j < m/2; ++j)
            {
              r32 in0Re = *dest0Re;
              r32 in0Im = *dest0Im;
              r32 in1Re = *dest1Re;
              r32 in1Im = *dest1Im;

              r32 productRe = in1Re*wRe - in1Im*wIm;
              r32 productIm = in1Re*wIm + in1Im*wRe;
              r32 out0Re = in0Re + productRe;
              r32 out0Im = in0Im + productIm;
              r32 out1Re = in0Re - productRe;
              r32 out1Im = in0Im - productIm;

              *dest0Re++ = out0Re;
              *dest0Im++ = out0Im;
              *dest1Re++ = out1Re;
              *dest1Im++ = out1Im;

              r32 wReOld = wRe;
              wRe = wReOld*wmRe - wIm*wmIm;
              wIm = wReOld*wmIm + wIm*wmRe;
            }
        }
    }
  for(u32 m = 2*simdWidth; m <= length; m <<= 1)
    {
      r32 angle = GS_TAU/(r32)m;

      // TODO: make trig ops fast
      c64 wm = C64Polar(1.f, angle);
      // TODO: maybe do these multiplies wide as well?
      c64 wmSq = wm*wm;
      c64 wmCu = wm*wmSq;
      c64 wmQu = wm*wmCu;
      WideFloat wmQuRe = wideSetConstantFloats(wmQu.re);
      WideFloat wmQuIm = wideSetConstantFloats(wmQu.im);

      for(u32 k = 0; k < length; k += m)
        {
          WideFloat wRe = wideSetFloats(1.f, wm.re, wmSq.re, wmCu.re);
          WideFloat wIm = wideSetFloats(0.f, wm.im, wmSq.im, wmCu.im);

          r32 *dest0Re = dest + k;
          r32 *dest0Im = destImTemp + k;
          r32 *dest1Re = dest + k + m/2;
          r32 *dest1Im = destImTemp + k + m/2;
          for(u32 j = 0; j < m/2; j += simdWidth)
            {
              /* WideFloat storeMask = wideSetConstantFloats(0.f); */
              /* WideFloat storeMaskInv = wideSetConstantFloats(0.f); */
              /* for(u32 lane = 0; lane < simdWidth; ++lane) */
              /*        { */
              /*          r32 val = (j + lane < m/2) ? maskF : 0; */
              /*          r32 valInv = (j + lane < m/2) ? 0 : maskF; */

              /*          wideSetLaneFloats(&storeMask, val, lane); */
              /*          wideSetLaneFloats(&storeMaskInv, valInv, lane); */
              /*        } */

              // NOTE: loads
              WideFloat in0Re = wideLoadFloats(dest0Re);
              WideFloat in0Im = wideLoadFloats(dest0Im);
              WideFloat in1Re = wideLoadFloats(dest1Re);
              WideFloat in1Im = wideLoadFloats(dest1Im);

              // NOTE: butterflies and twiddles
              WideFloat productRe = wideSubFloats(wideMulFloats(wRe, in1Re),
                                                  wideMulFloats(wIm, in1Im));
              WideFloat productIm = wideAddFloats(wideMulFloats(wRe, in1Im),
                                                  wideMulFloats(wIm, in1Re));
              WideFloat out0Re = wideAddFloats(in0Re, productRe);
              WideFloat out0Im = wideAddFloats(in0Im, productIm);
              WideFloat out1Re = wideSubFloats(in0Re, productRe);
              WideFloat out1Im = wideSubFloats(in0Im, productIm);

              // NOTE: mask out what not to store
              /* WideFloat store0Re = wideAddFloats(wideMaskFloats(out0Re, storeMask), */
              /*                                         wideMaskFloats(in0Re, storeMaskInv)); */
              /* WideFloat store0Im = wideAddFloats(wideMaskFloats(out0Im, storeMask), */
              /*                                         wideMaskFloats(in0Im, storeMaskInv)); */
              /* WideFloat store1Re = wideAddFloats(wideMaskFloats(out1Re, storeMask), */
              /*                                         wideMaskFloats(in1Re, storeMaskInv)); */
              /* WideFloat store1Im = wideAddFloats(wideMaskFloats(out1Im, storeMask), */
              /*                                         wideMaskFloats(in1Im, storeMaskInv)); */

              // NOTE: do the store
              /* wideStoreFloats(dest0Re, store0Re); */
              /* wideStoreFloats(dest0Im, store0Im); */
              /* wideStoreFloats(dest1Re, store1Re); */
              /* wideStoreFloats(dest1Im, store1Im); */
              wideStoreFloats(dest0Re, out0Re);
              wideStoreFloats(dest0Im, out0Im);
              wideStoreFloats(dest1Re, out1Re);
              wideStoreFloats(dest1Im, out1Im);


              WideFloat wReOld = wRe;
              WideFloat wImOld = wIm;
              wRe = wideSubFloats(wideMulFloats(wReOld, wmQuRe),
                                  wideMulFloats(wImOld, wmQuIm));
              wIm = wideAddFloats(wideMulFloats(wReOld, wmQuIm),
                                  wideMulFloats(wImOld, wmQuRe));

              dest0Re += simdWidth;
              dest0Im += simdWidth;
              dest1Re += simdWidth;
              dest1Im += simdWidth;
            }
        }
    }
}

static void
convolve(r32 *dest, r32 *src0, r32 *src1, u32 length, Arena *tempAllocator)
{
  // NOTE: fast convolution with ffts
  ASSERT(isPowerOf2(length));
  u32 simdWidth = 4;

  r32 *temp0Re = arenaPushArray(tempAllocator, length, r32, arenaFlagsZeroAlign(simdWidth));
  r32 *temp0Im = arenaPushArray(tempAllocator, length, r32, arenaFlagsZeroAlign(simdWidth));
  r32 *temp1Re = arenaPushArray(tempAllocator, length, r32, arenaFlagsZeroAlign(simdWidth));
  r32 *temp1Im = arenaPushArray(tempAllocator, length, r32, arenaFlagsZeroAlign(simdWidth));

  fft_real_noPermute(temp0Re, temp0Im, src0, length);
  fft_real_noPermute(temp1Re, temp1Im, src1, length);

  r32 *tempDestRe = temp0Re;
  r32 *tempDestIm = temp0Im;
  r32 *tempSrc0Re = temp0Re;
  r32 *tempSrc0Im = temp0Im;
  r32 *tempSrc1Re = temp1Re;
  r32 *tempSrc1Im = temp1Im;
  ASSERT((length & 0x3) == 0);
  for(u32 i = 0; i < length; i += simdWidth)
    {
      WideFloat in0Re = wideLoadFloats(tempSrc0Re);
      WideFloat in0Im = wideLoadFloats(tempSrc0Im);
      WideFloat in1Re = wideLoadFloats(tempSrc1Re);
      WideFloat in1Im = wideLoadFloats(tempSrc1Im);

      WideFloat outRe = wideSubFloats(wideMulFloats(in0Re, in1Re), wideMulFloats(in0Im, in1Re));
      WideFloat outIm = wideAddFloats(wideMulFloats(in0Re, in1Im), wideMulFloats(in0Im, in1Re));

      wideStoreFloats(tempDestRe, outRe);
      wideStoreFloats(tempDestIm, outIm);

      tempDestRe += simdWidth;
      tempDestIm += simdWidth;

      tempSrc0Re += simdWidth;
      tempSrc0Im += simdWidth;
      tempSrc1Re += simdWidth;
      tempSrc1Im += simdWidth;
    }

  ifft_real_noPermute(dest, temp1Im, temp0Re, temp0Im, length);
}

inline void
fft(c64 *destBuffer, r32 *sourceBuffer, u32 lengthInit, u32 stride = 1)
{
#if 0
  u32 length = lengthInit;
  if(length == 1)
    {
      (*destBuffer).re = *sourceBuffer;
    }
  else
    {
      fft(destBuffer, sourceBuffer, length/2, 2*stride);
      fft(destBuffer + length/2, sourceBuffer + stride, length/2, 2*stride);

      for(u32 k = 0; k < length/2; ++k)
        {
          c64 w = C64Polar(1, -GS_TAU*k/(r32)length);
          c64 in0 = destBuffer[k];
          c64 in1 = destBuffer[k + length/2];

          destBuffer[k] = in0 + w*in1;
          destBuffer[k + length/2] = in0 - w*in1;
        }
    }
#else
  // NOTE: iterative radix-2 DIT fft
  // TODO: radix-4
  u32 length = ROUND_UP_TO_POWER_OF_2(lengthInit);
  u32 lengthSizeInBits = 8*sizeof(length);
  u32 logLength = log2(length);

  // NOTE: bit-reverse copy
  // TODO: optimize
  for(u32 index = 0; index < length; ++index)
    {
      u32 reversedIndex = reverseBits(index);
      reversedIndex >>= lengthSizeInBits - logLength;
      ASSERT(reversedIndex < length);

      destBuffer[index].re = sourceBuffer[reversedIndex];
    }

  // NOTE: twiddles
  for(u32 level = 1; level <= logLength; ++level)
    {
      u32 m = (1 << level);
      r32 angle = -GS_TAU/(r32)m;
      c64 wm = C64Polar(1, angle);

      for(u32 k = 0; k < length; k += m)
        {
          c64 w = C64(1, 0);

          c64 *dest0 = destBuffer + k;
          c64 *dest1 = destBuffer + k + m/2;
          for(u32 j = 0; j < m/2; ++j)
            {
              c64 in0 = *dest0;
              c64 in1 = *dest1;

              // TODO: optimize this math:
              //       use SIMD intrinsics, and take advantage of the fact that the input data is real
              *dest0++ = in0 + w*in1;
              *dest1++ = in0 - w*in1;

              w *= wm;
            }
        }
    }
#endif
}

inline void
ifft(r32 *destBuffer, r32 *destImScratch, c64 *sourceBuffer, u32 lengthInit, u32 stride = 1)
{
#if 0
  if(length == 1)
    {
      *destBuffer = (*sourceBuffer).re;
    }
  else
    {
      ifft(destBuffer, sourceBuffer, length/2, 2*stride);
      ifft(destBuffer + length/2, sourceBuffer + stride, length/2, 2*stride);

      for(u32 k = 0; k < length/2; ++k)
        {
          c64 w = C64Polar(1, GS_TAU*k/(r32)length);
          c64 in0 = destBuffer[k];
          c64 in1 = destBuffer[k + length/2];

          destBuffer[k] = in0 + w*in1;
          destBuffer[k + length/2] = in0 - w*in1;
        }
    }
#else
  // TODO: optimize
  //ASSERT(!(length & 1));
  u32 length = ROUND_UP_TO_POWER_OF_2(lengthInit);
  u32 logLength = log2(length);
  // NOTE: assuming power-of-two size
  // TODO: support non-power-of-two sizes, don't require zero-padding input
  ASSERT(length == (u32)(1 << logLength));
  r32 invLength = 1.f/(r32)length;

  u32 lengthSizeInBits = 8*sizeof(length);
  for(u32 index = 0; index < length; ++index)
    {
      u32 reversedIndex = reverseBits(index);
      reversedIndex >>= lengthSizeInBits - logLength;

      c64 source = invLength*sourceBuffer[reversedIndex];
      destBuffer[index] = source.re;
      destImScratch[index] = source.im;
    }

  for(u32 level = 1; level <= logLength; ++level)
    {
      u32 m = 1 << level;
      r32 angle = GS_TAU/(r32)m;
      c64 wm = C64Polar(1, angle);

      for(u32 k = 0; k < length; k += m)
        {
          c64 w = C64(1, 0);

          r32 *dest0 = destBuffer + k;
          r32 *dest0Im = destImScratch + k;
          r32 *dest1 = destBuffer + k + m/2;
          r32 *dest1Im = destImScratch + k + m/2;
          for(u32 j = 0; j < m/2; ++j)
            {
              c64 in0 = C64(*dest0, *dest0Im);
              c64 in1 = C64(*dest1, *dest1Im);

              c64 result0 = in0 + w*in1;
              c64 result1 = in0 - w*in1;

              *dest0++ = result0.re;
              *dest1++ = result1.re;
              *dest0Im++ = result0.im;
              *dest1Im++ = result1.im;

              w *= wm;
            }
        }
    }
#endif
}

inline void
fft(c64 *output, c64 *input, u32 lengthInit)
{
  u32 length = ROUND_UP_TO_POWER_OF_2(lengthInit);
  u32 lengthSizeInBits = 8*sizeof(lengthInit);
  u32 logLength = log2(length);

  for(u32 index = 0; index < length; ++index)
    {
      u32 reverseIndex = reverseBits(index);
      reverseIndex >>= lengthSizeInBits - logLength;
      output[index] = input[reverseIndex];
    }

  for(u32 s = 1; s <= logLength; ++s)
    {
      u32 m = 1 << s;
      r32 angle = -GS_TAU/(r32)m;
      c64 wm = C64Polar(1, angle);

      for(u32 k = 0; k < length; k += m)
        {
          c64 w = C64(1, 0);

          c64 *dest0 = output + k;
          c64 *dest1 = output + k + m/2;
          for(u32 j = 0; j < m/2; ++j)
            {
              c64 in0 = *dest0;
              c64 in1 = *dest1;

              // TODO: SIMD
              *dest0++ = in0 + w*in1;
              *dest1++ = in0 - w*in1;

              w *= wm;
            }
        }
    }
}

inline void
ifft(c64 *output, c64 *input, u32 lengthInit)
{
  u32 length = ROUND_UP_TO_POWER_OF_2(lengthInit);
  u32 lengthSizeInBits = 8*sizeof(lengthInit);
  u32 logLength = log2(length);
  r32 invLength = 1.f/(r32)length;

  for(u32 index = 0; index < length; ++index)
    {
      u32 reverseIndex = reverseBits(index);
      reverseIndex >>= lengthSizeInBits - logLength;
      output[index] = invLength*input[reverseIndex];
    }

  for(u32 s = 1; s <= logLength; ++s)
    {
      u32 m = 1 << s;
      r32 angle = GS_TAU/(r32)m;
      c64 wm = C64Polar(1, angle);

      for(u32 k = 0; k < length; k += m)
        {
          c64 w = C64(1, 0);

          c64 *dest0 = output + k;
          c64 *dest1 = output + k + m/2;
          for(u32 j = 0; j < m/2; ++j)
            {
              c64 in0 = *dest0;
              c64 in1 = *dest1;

              // TODO: SIMD
              *dest0++ = in0 + w*in1;
              *dest1++ = in0 - w*in1;

              w *= wm;
            }
        }
    }
}

inline void
czt(c64 *output, r32 *input, u32 length, Arena *scratchAllocator)
{
  // NOTE: chirp z transform.
  //       for taking ffts of arbitrary length signals

  u32 chirpLength = 2*length - 1;
  u32 fftLength = ROUND_UP_TO_POWER_OF_2(chirpLength);

  r32 angle = -GS_PI/(r32)length;
  c64 wBase = C64Polar(1, angle);
  c64 *chirp = arenaPushArray(scratchAllocator, chirpLength, c64);//, arenaFlagsZeroNoAlign());
  c64 *reciprocalChirp = arenaPushArray(scratchAllocator, fftLength, c64);//, arenaFlagsZeroNoAlign());
  c64 *scaledInput = arenaPushArray(scratchAllocator, fftLength, c64);//, arenaFlagsZeroNoAlign());
  c64 *wTemp = arenaPushArray(scratchAllocator, fftLength, c64);//, arenaFlagsZeroNoAlign());
  chirp[length - 1] = C64(1, 0);
  reciprocalChirp[length - 1] = C64(1, 0);
  wTemp[0] = C64(1, 0);
  scaledInput[0] = input[0]*chirp[length];
  for(u32 k = 1; k < length; ++k)
    {
      // NOTE: generate chirp, multiply the input
      c64 chirpVal = chirp[length - 1 + k - 1] * wTemp[k - 1] * wBase;
      c64 conjChirpVal = conjugateC64(chirpVal);
      chirp[length - 1 + k] = chirpVal;
      chirp[length - 1 - k] = chirpVal;
      reciprocalChirp[length - 1 + k] = conjChirpVal;
      reciprocalChirp[length - 1 - k] = conjChirpVal;
      scaledInput[k] = input[k] * chirp[length + k - 1 ];

      // TODO: don't compute more of these values than necessary. may be a source of significant numerical error
      //if(k*k >= 2*length)
      wTemp[k] = wTemp[k - 1] * wBase * wBase;
    }

  // NOTE: convolve the scaled input with the reciprocal chirp
  // TODO: these fft calls are not optimized for doing convolutions:
  //       we do an (order 2) input permutation at the beginning of each fft and ifft so that our output coefficients
  //       are in order of increasing frequency. That is not necessary to do here, and a huge burden on performance
  c64 *fftScaledInput = wTemp;
  fft(fftScaledInput, scaledInput, fftLength);

  c64 *fftReciprocalChirp = scaledInput;
  fft(fftReciprocalChirp, reciprocalChirp, fftLength);

  for(u32 index = 0; index < fftLength; ++index)
    {
      fftScaledInput[index] *= fftReciprocalChirp[index];
    }

  c64 *ifftOutput = reciprocalChirp;
  ifft(ifftOutput, fftScaledInput, fftLength);

  // NOTE: write out, scaling by the chirp
  for(u32 index = 0; index < length; ++index)
    {
      output[index] = ifftOutput[length + index - 1] * chirp[length + index - 1];
    }
}

inline void
iczt(r32 *outputReal, r32 *outputImag, c64 *input, u32 length, Arena *scratchAllocator)
{
  r32 invLength = 1.f/(r32)length;
  u32 chirpLength = 2*length - 1;
  u32 fftLength = ROUND_UP_TO_POWER_OF_2(chirpLength);

  r32 angle = GS_PI/(r32)length;
  c64 wBase = C64Polar(1, angle);
  c64 *chirp = arenaPushArray(scratchAllocator, chirpLength, c64);//, arenaFlagsZeroNoAlign());
  c64 *reciprocalChirp = arenaPushArray(scratchAllocator, fftLength, c64);//, arenaFlagsZeroNoAlign());
  c64 *scaledInput = arenaPushArray(scratchAllocator, fftLength, c64);//, arenaFlagsZeroNoAlign());
  c64 *wTemp = arenaPushArray(scratchAllocator, fftLength, c64);//, arenaFlagsZeroNoAlign());
  chirp[length - 1] = C64(1, 0);
  reciprocalChirp[length - 1] = C64(1, 0);
  wTemp[0] = C64(1, 0);
  scaledInput[0] = input[0]*chirp[length];
  for(u32 k = 1; k < length; ++k)
    {
      c64 chirpVal = chirp[length - 1 + k - 1] * wTemp[k - 1] * wBase;
      c64 conjChirpVal = conjugateC64(chirpVal);
      chirp[length - 1 + k] = chirpVal;
      chirp[length - 1 - k] = chirpVal;
      reciprocalChirp[length - 1 + k] = conjChirpVal;
      reciprocalChirp[length - 1 - k] = conjChirpVal;
      scaledInput[k] = input[k] * chirp[length + k - 1 ];

      wTemp[k] = wTemp[k - 1] * wBase * wBase;
    }

  c64 *fftScaledInput = wTemp;
  fft(fftScaledInput, scaledInput, fftLength);

  c64 *fftReciprocalChirp = scaledInput;
  fft(fftReciprocalChirp, reciprocalChirp, fftLength);

  for(u32 index = 0; index < fftLength; ++index)
    {
      fftScaledInput[index] *= fftReciprocalChirp[index];
    }

  c64 *ifftOutput = reciprocalChirp;
  ifft(ifftOutput, fftScaledInput, fftLength);

  for(u32 index = 0; index < length; ++index)
    {
      c64 outVal = invLength * ifftOutput[length + index - 1] * chirp[length + index - 1];
      outputReal[index] = outVal.re;
      outputImag[index] = outVal.im;
    }
}
#endif
