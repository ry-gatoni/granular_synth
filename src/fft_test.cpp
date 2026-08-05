#include "fft_test_signals.h"

struct FftForwardTestArgs
{
  r32 *input;
  usz count;
  FftKernels *kernels;
  c64 *target;
};

static TestResult
testForwardFFT(Arena *arena, r32 *input, usz count, FftKernels *kernels, c64 *target)
{
  c64 *output = arenaPushArray(arena, count/2, c64,
			       arenaFlagsZeroAlign(64));
  fft_re(output, input, count, kernels);

  b32 success = 1;
  String8List log = {};
  {
    r32 const tol = 1e-3;
    r32 const eps = 1e-6;
    // NOTE: test first value (DC + i*Nyquist)
    {
      r32 resultDc = output[0].re;
      r32 resultNq = output[0].im;
      r32 targetDc = target[0].re;
      r32 targetNq = target[count/2].re;

      r32 resultMagSq = resultDc*resultDc + resultNq*resultNq;
      r32 targetMagSq = targetDc*targetDc + targetNq*targetNq;
      r32 percentErr = gsAbs(resultMagSq - targetMagSq)/(targetMagSq + eps);
      b32 sampleSuccess = percentErr < tol;
      if(!sampleSuccess)
      {
	success = 0;
	stringListPushFormat(arena, &log,
			     "fft discrepancy at sample 0:\n"
			     "  result = %.4f + %.4fi\n"
			     "  target = %.4f + %.4fi\n"
			     "     err = %.4f%%\n",
			     resultDc, resultNq,
			     targetDc, targetNq,
			     percentErr);
      }
    }

    // NOTE: test rest of values
    for(usz i = 1; i < count/2; ++i)
    {
      r32 resultRe = output[i].re;
      r32 resultIm = output[i].im;
      r32 targetRe = target[i].re;
      r32 targetIm = target[i].im;

      r32 resultMagSq = resultRe*resultRe + resultIm*resultIm;
      r32 targetMagSq = targetRe*targetRe + targetIm*targetIm;
      r32 percentErr = gsAbs(resultMagSq - targetMagSq)/(targetMagSq + eps);
      b32 sampleSuccess = percentErr < tol;
      if(!sampleSuccess)
      {
	success = 0;
	stringListPushFormat(arena, &log,
			     "fft discrepancy at sample %lu:\n"
			     "  result = %.4f + %.4fi\n"
			     "  target = %.4f + %.4fi\n"
			     "     err = %.4f%%\n",
			     i,
			     resultRe, resultIm,
			     targetRe, targetIm,
			     percentErr);
      }
    }
  }

  TestResult result = {};
  result.success = success;
  result.log = log;
  return(result);
}

struct FftReverseTestArgs
{
  c64 *input;
  usz count;
  FftKernels *kernels;
  r32 *target;
};

static TestResult
testReverseFFT(Arena *arena, c64 *input, usz count, FftKernels *kernels, r32 *target)
{
  c64 *in = arenaPushArray(arena, count/2, c64,
			   arenaFlagsZeroAlign(64));
  COPY_ARRAY(in, input, count/2, c64);
  in[0].im = input[count/2].re;
  r32 *output = arenaPushArray(arena, count, r32,
			       arenaFlagsZeroAlign(64));

  ifft_re(output, in, count, kernels);

  b32 success = 1;
  String8List log = {};
  {
    r32 const tol = 2e-2;
    for(usz i = 0; i < count; ++i)
    {
      r32 resultVal = output[i];
      r32 targetVal = target[i];
      // r32 resultSq = resultVal*resultVal;
      // r32 targetSq = targetVal*targetVal;
      //r32 percentErr = gsAbs(resultVal - targetVal)/(targetVal + eps);
      r32 err = gsAbs(resultVal - targetVal);
      b32 sampleSuccess = err < tol;
      if(!sampleSuccess)
      {
	success = 0;
	stringListPushFormat(arena, &log,
			     "ifft discrepancy at sample %lu\n"
			     "  result = %.4f\n"
			     "  target = %.4f\n"
			     "     err = %.4f\n",
			     i,
			     resultVal,
			     targetVal,
			     err);
      }
    }
  }

  TestResult result = {};
  result.success = success;
  result.log = log;
  return(result);
}

// TODO: pass the args
FftForwardTestArgs fft_test_args = {
  (r32*)fft_test_signal,
  ARRAY_COUNT(fft_test_signal)/sizeof(r32),
  &fft_kernels,
  (c64*)fft_test_result,
};
TEST_FN_DEF(fft_test, &fft_test_args)
{
  auto *fftArgs = (FftForwardTestArgs*)args;
  r32 *input = fftArgs->input;
  usz count = fftArgs->count;
  FftKernels *kernels = fftArgs->kernels;
  c64 *target = fftArgs->target;

  TestResult result = testForwardFFT(arena, input, count, kernels, target);
  return(result);
}

FftReverseTestArgs ifft_test_args = {
  (c64*)fft_test_result,
  ARRAY_COUNT(fft_test_signal)/sizeof(r32),
  &fft_kernels,
  (r32*)fft_test_signal
};
TEST_FN_DEF(ifft_test, &ifft_test_args)
{
  auto *ifftArgs = (FftReverseTestArgs*)args;
  c64 *input = ifftArgs->input;
  usz count = ifftArgs->count;
  FftKernels *kernels = ifftArgs->kernels;
  r32 *target = ifftArgs->target;

  TestResult result = testReverseFFT(arena, input, count, kernels, target);
  return(result);
}

struct FFT_TestResult
{
  b32 success;
  usz cycleCount;
  String8List log;
};

#if 0
static FFT_TestResult
testFFTFunction(Arena *arena, FFT_Function *fft, FloatBuffer input, ComplexBuffer target)
{
  ComplexBuffer fftResult = fft(arena, input);

  String8List log = {};

  b32 success = target.count == fftResult.count;
  if(success)
    {
      r32 tol = 1e-3;
      for(u32 i = 0; i < target.count; ++i)
	{
	  r32 resultRe = fftResult.reVals[i];
	  r32 resultIm = fftResult.imVals[i];
	  r32 targetRe = target.reVals[i];
	  r32 targetIm = target.imVals[i];

	  r32 resultMagSq = resultRe*resultRe + resultIm*resultIm;
	  r32 targetMagSq = targetRe*targetRe + targetIm*targetIm;
	  success = ((gsAbs(resultMagSq - targetMagSq) / targetMagSq) < tol);
	  if(!success)
	    {
	      stringListPushFormat(arena, &log,
				   "fft discrepancy at sample %lu: \n"
				   "  result = %.4f + %.4fi\n"
				   "  target = %.4f + %.4fi\n",
				   i,
				   resultRe, resultIm,
				   targetRe, targetIm);
	    }
	}
    }
  else
    {
      stringListPushFormat(arena, &log,
			   "ERROR: result and target lengths differ:\n"
			   "       result.count = %u\n"
			   "       target.count = %u\n",
			   fftResult.count,
			   target.count);
    }

  FFT_TestResult result = {};
  result.success = success;
  result.cycleCount = 0; // TODO: profile
  result.log = log;
  return(result);
}

static FFT_TestResult
testIFFTFunction(Arena *arena, IFFT_Function *ifft, ComplexBuffer input, FloatBuffer target)
{
  FloatBuffer ifftResult = ifft(arena, input);

  String8List log = {};

  b32 success = target.count == ifftResult.count;
  if(success)
    {
      r32 tol = 1e-3;
      for(u32 i = 0; i < target.count; ++i)
	{
	  r32 resultSample = ifftResult.vals[i];
	  r32 targetSample = target.vals[i];
	  success = ((gsAbs(resultSample - targetSample) / targetSample) < tol);
	  if(!success)
	    {
	      stringListPushFormat(arena, &log,
				   "ifft discrepancy at sample %lu: \n"
				   "  result = %.7f\n"
				   "  target = %.7f\n",
				   i,
				   resultSample,
				   targetSample);
	    }
	}
    }
  else
    {
      stringListPushFormat(arena, &log,
			   "ERROR: result and target lengths differ:\n"
			   "       result.count = %u\n"
			   "       target.count = %u\n",
			   ifftResult.count,
			   target.count);
    }

  FFT_TestResult result = {};
  result.success = success;
  result.cycleCount = 0; // TODO: profile
  result.log = log;
  return(result);
}
#endif

#if 0
static bool
fftTest(Arena *allocator)
{
  u32 testSignalLength = 16;
  r32 freq = 4;
  r32 nFreq = freq*GS_TAU/(r32)testSignalLength;
  r32 *testSignal = arenaPushArray(allocator, testSignalLength, r32, arenaFlagsZeroAlign(4*sizeof(r32)));
  for(u32 i = 0; i < testSignalLength; ++i)
    {
      testSignal[i] = Sin((r32)i*nFreq);
    }

  r32 *testDestRe = arenaPushArray(allocator, testSignalLength, r32, arenaFlagsZeroAlign(4*sizeof(r32)));
  r32 *testDestIm = arenaPushArray(allocator, testSignalLength, r32, arenaFlagsZeroAlign(4*sizeof(r32)));
  c64 *testDestComplex = arenaPushArray(allocator, testSignalLength, c64, arenaFlagsZeroAlign(4*sizeof(r32)));

  fft(testDestComplex, testSignal, testSignalLength);
  //fft_real_permute(testDestRe, testDestIm, testSignal, testSignalLength);
  fft_real_noPermute(testDestRe, testDestIm, testSignal, testSignalLength);

  r32 *testIfftResult = arenaPushArray(allocator, testSignalLength, r32, arenaFlagsZeroAlign(4*sizeof(r32)));
  r32 *testIfftTemp = arenaPushArray(allocator, testSignalLength, r32, arenaFlagsZeroAlign(4*sizeof(r32)));

  //ifft(testIfftResult, testIfftTemp, testDestComplex, testSignalLength);
  //ifft_real_permute(testIfftResult, testIfftTemp, testDestRe, testDestIm, testSignalLength);
  ifft_real_noPermute(testIfftResult, testIfftTemp, testDestRe, testDestIm, testSignalLength);

  r32 maxErr = 0.f;
  r32 errTol = 0.001f;
  for(u32 i = 0; i < testSignalLength; ++i)
    {
      r32 err = Abs(testSignal[i] - testIfftResult[i]);
      maxErr = MAX(maxErr, err);
    }

  bool result = (maxErr <= errTol);
  return(result);
}
#endif
