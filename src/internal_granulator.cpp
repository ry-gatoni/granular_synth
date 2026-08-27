static inline r32
getRandomStereoPosition(r32 spreadAmount)
{
  // Generate a random value between -1 and 1 using a normal distribution
  // Higher spreadAmount = wider stereo field

  // Simple random number between -1 and 1 (using two calls to rand() for better distribution)
  // TODO: replace rand() with custom prng for better performance, uniformity, and consistency across platforms
  // r32 randVal = ((((r32)gsRand() / (r32)RAND_MAX) * 2.0f - 1.0f) * 0.5f +
  //             (((r32)gsRand() / (r32)RAND_MAX) * 2.0f - 1.0f)) * 0.5f;
  RangeR32 range = {-0.5f, 0.5f};
  r32 randVal = gsRand(range) + gsRand(range);

  // Scale by spread parameter (0 = mono, 1 = full stereo)
  return randVal * spreadAmount;
}

static inline void
initializeWindows(GrainManager* grainManager)
{
  r32 windowLengthF = (r32)WINDOW_LENGTH;
  r32 windowLengthInv = 1.f/windowLengthF;

  r32 *hannBuffer = grainManager->windowBuffer[WindowShape_hann];
  r32 *sineBuffer = grainManager->windowBuffer[WindowShape_sine];
  r32 *triangleBuffer = grainManager->windowBuffer[WindowShape_triangle];
  r32 *rectangularBuffer = grainManager->windowBuffer[WindowShape_rectangle];
  for(u32 sample = 0; sample < WINDOW_LENGTH; ++sample)
  {
    *hannBuffer++ = 0.5f * (1.f - gsCos(GS_TAU * sample * windowLengthInv));
    *sineBuffer++ = gsSin(GS_PI * sample * windowLengthInv);
    *triangleBuffer++ = (1.f - gsAbs((sample - (windowLengthF - 1.f) / 2.f) / ((windowLengthF - 1.f) / 2.f)));
    *rectangularBuffer++ = 1.f;
  }
}

static inline void
makeNewGrain(GrainManager* grainManager, u32 grainSize, r32 windowParam, r32 spread, u32 sampleIndex)
{
  AudioRingBuffer *grainBuffer = grainManager->internalBuffer;

  Grain* result = grainManager->grainFreeList;
  if(result)
  {
    STACK_POP(grainManager->grainFreeList);
  }
  else
  {
    result = arenaPushStruct(grainManager->grainAllocator, Grain);
  }
  ASSERT(result);

  result->readIndex = grainBuffer->readIndex;

  result->samplesToPlay = grainSize;
  result->length = grainSize;
  result->lengthInv = (r32)1/grainSize;
  result->windowParam = MIN(windowParam, (WindowShape_count - 1));

  result->stereoPosition = getRandomStereoPosition(spread);

  result->startSampleIndex = sampleIndex;
  result->isFinished = 0;

  DLL_PUSH_BACK(grainManager->firstPlayingGrain, grainManager->lastPlayingGrain, result);

  grainManager->samplesProcessedSinceLastSeed = 0;
  ++grainManager->grainCount;

  logFormatString("created a grain. grain count is now %u", grainManager->grainCount);
}

static inline void
destroyGrain(GrainManager* grainManager, Grain* grain)
{
  ASSERT(grain->isFinished);
  DLL_REMOVE(grainManager->firstPlayingGrain, grainManager->lastPlayingGrain, grain);
  STACK_PUSH(grainManager->grainFreeList, grain);
  grain->prev = 0;
  --grainManager->grainCount;

  logFormatString("destroyed a grain. grain count is now %u", grainManager->grainCount);
}

inline r32
getWindowVal(GrainManager* grainManager, r32 samplesPlayedFrac, r32 windowParam)
{
  r32 tablePosition = samplesPlayedFrac * (r32)(WINDOW_LENGTH - 1);
  u32 tableIndex = (u32)tablePosition;
  r32 indexFrac = tablePosition - tableIndex;

  u32 windowIndex = (u32)windowParam;
  r32 windowFrac = windowParam - windowIndex;

  u32 tableIndex0 = tableIndex;
  u32 tableIndex1 = MIN(tableIndex + 1, WINDOW_LENGTH - 1);

  u32 windowIndex0 = windowIndex;
  u32 windowIndex1 = MIN(windowIndex + 1, ARRAY_COUNT(grainManager->windowBuffer) - 1);

  ASSERT(tableIndex0 < WINDOW_LENGTH && tableIndex1 < WINDOW_LENGTH);
  ASSERT(windowIndex0 < ARRAY_COUNT(grainManager->windowBuffer) && windowIndex1 < ARRAY_COUNT(grainManager->windowBuffer));

  r32 floor0 = grainManager->windowBuffer[windowIndex0][tableIndex0];
  r32 ceil0  = grainManager->windowBuffer[windowIndex0][tableIndex1];
  r32 floor1 = grainManager->windowBuffer[windowIndex1][tableIndex0];
  r32 ceil1  = grainManager->windowBuffer[windowIndex1][tableIndex1];

  // NOTE: bilinear blend
  r32 windowVal0 = lerp(floor0, ceil0, indexFrac);
  r32 windowVal1 = lerp(floor1, ceil1, indexFrac);
  r32 result = lerp(windowVal0, windowVal1, windowFrac);

  return(result);
}

static void
synthesize(GrainManager* grainManager, AudioRingBufferView dest)
{
  AudioRingBuffer *grainBuffer = grainManager->internalBuffer;

  u64 samplesToWrite = dest.sampleCount;

  PluginFloatParameter *densityParam = grainManager->parameters + PluginParameter_density;
  PluginFloatParameter *sizeParam = grainManager->parameters + PluginParameter_size;
  PluginFloatParameter *windowParam = grainManager->parameters + PluginParameter_window;
  PluginFloatParameter *spreadParam = grainManager->parameters + PluginParameter_spread;
  PluginFloatParameter *offsetParam = grainManager->parameters + PluginParameter_offset;

  u32 targetOffset = pluginReadFloatParameter(offsetParam);
  u32 currentOffset = grainBuffer->writeIndex - grainBuffer->readIndex;
  r32 readPositionIncrement = ((r32)currentOffset - (r32)targetOffset)/(r32)samplesToWrite;
#if 0
  logFormatString("currentOffset: %.2f", currentOffset);
  logFormatString("targetOffset: %.2f", targetOffset);
  logFormatString("readPositionIncrement: %.2f", readPositionIncrement);
#endif

  // NOTE: process grains
  {
    r32 maxDensity = R32_MIN;

    // NOTE: create new grains
    {
      u32 startReadIndex = grainBuffer->readIndex;
      for(u32 sampleIndex = 0; sampleIndex < samplesToWrite; ++sampleIndex)
      {
        r32 window     = pluginUpdateFloatParameter(windowParam);
        u32 grainSize  = (u32)pluginUpdateFloatParameter(sizeParam);
        r32 density    = pluginUpdateFloatParameter(densityParam);
        r32 spread     = pluginUpdateFloatParameter(spreadParam);
        pluginUpdateFloatParameter(offsetParam); // TODO: should we update the read position target & increment each sample?

        r32 iot = (r32)grainSize/density;
        if(grainManager->samplesProcessedSinceLastSeed >= iot)
        {
          makeNewGrain(grainManager, grainSize, window, spread, sampleIndex);
          logFormatString("creating grain at sample %u", sampleIndex);
        }

        ++grainManager->samplesProcessedSinceLastSeed;

        r32 newReadPosition = (r32)startReadIndex + readPositionIncrement*(r32)(sampleIndex + 1);
        grainBuffer->readIndex = (u32)newReadPosition;

        maxDensity = MAX(maxDensity, density);
      }
    }

    // TODO: loop over channels so we access samples contiguously
    // NOTE: process playing grains
    r32 *destL = dest.start[0];
    r32 *destR = dest.start[1];
    r32 attenFactor = 1.f/MAX(1.f, maxDensity);
    for(Grain *grain = grainManager->firstPlayingGrain; grain; grain = grain->next)
    {
      ASSERT(!grain->isFinished);
      // TODO: vectorize
      for(u32 sampleIndex = grain->startSampleIndex; sampleIndex < samplesToWrite; ++sampleIndex)
      {
        if(grain->samplesToPlay)
        {
          u64 grainReadIndexWrapped = grain->readIndex & (grainBuffer->sampleCount - 1);
	  r32 grainSampleL = grainBuffer->samples[0][grainReadIndexWrapped];
	  r32 grainSampleR = grainBuffer->samples[1][grainReadIndexWrapped];

          u32 samplesPlayed = grain->length - grain->samplesToPlay;
          r32 samplesPlayedFrac = (r32)samplesPlayed*grain->lengthInv;
          r32 windowVal = getWindowVal(grainManager, samplesPlayedFrac, grain->windowParam);

          r32 panL = 1.0f - MAX(0.0f, grain->stereoPosition);
          r32 panR = 1.0f + MIN(0.0f, grain->stereoPosition);

          r32 lVal = attenFactor * windowVal * panL * grainSampleL;
          r32 rVal = attenFactor * windowVal * panR * grainSampleR;

	  destL[sampleIndex] += lVal;
	  destR[sampleIndex] += rVal;

          ++grain->readIndex;
          --grain->samplesToPlay;
        }
        else
        {
          grain->isFinished = 1;
          logFormatString("destroying grain at sample %u", sampleIndex);
          break;
        }
      }
    }

    // NOTE: remove finished grains from playlist
    for(Grain *grain = grainManager->firstPlayingGrain; grain;)
    {
      Grain *next = grain->next;
      if(grain->isFinished)
      {
        destroyGrain(grainManager, grain);
      }
      else
      {
        grain->startSampleIndex = 0;
      }
      grain = next;
    }
  }
}

static void
grainManagerRefill(AudioBufferStream *stream)
{
  GrainManager *grainManager = (GrainManager*)stream;
  AudioRingBuffer *grainInputBuffer = grainManager->internalBuffer;
  AudioRingBuffer *grainOutputBuffer = grainManager->outputBuffer;

  // NOTE: advance read cursor
  {
    u32 samplesRead = stream->sampleCursor;
    ZERO_ARRAY(stream->startSamples[0], samplesRead, r32);
    ZERO_ARRAY(stream->startSamples[1], samplesRead, r32);
    rbEndRead(grainOutputBuffer, samplesRead);
  }

  // NOTE: refill our input buffers if needed
  AudioBufferStream *sampleSource  = grainManager->sampleSource;
  if(sampleSource->sampleCursor == sampleSource->sampleCount)
  {
    sampleSource->refill(sampleSource);
    ASSERT(sampleSource->sampleCursor == 0)
  }

  r32 *srcL = sampleSource->startSamples[0] + sampleSource->sampleCursor;
  r32 *srcR = sampleSource->startSamples[1] + sampleSource->sampleCursor;
  u64 availableReadSamples = sampleSource->sampleCount;

  u64 availableWriteSamples;
  // NOTE: fill the grain buffer
  {
    AudioRingBufferView destSamples = rbGetWriteView(grainInputBuffer);
    u64 samplesToWrite = MIN(availableReadSamples, destSamples.sampleCount);
    COPY_ARRAY(destSamples.start[0], srcL, samplesToWrite, r32);
    COPY_ARRAY(destSamples.start[1], srcR, samplesToWrite, r32);
    availableWriteSamples = samplesToWrite;

    rbEndWrite(grainInputBuffer, samplesToWrite);
    sampleSource->sampleCursor += samplesToWrite;

    enqueueGrainStateView(grainManager);
  }

  AudioRingBufferView destSamples = rbGetWriteView(grainOutputBuffer);
  destSamples.sampleCount = availableWriteSamples;
  synthesize(grainManager, destSamples);
  rbEndWrite(grainOutputBuffer, destSamples.sampleCount);

  AudioRingBufferView readSamples = rbGetReadView(grainOutputBuffer);
  stream->startSamples[0] = readSamples.start[0];
  stream->startSamples[1] = readSamples.start[1];
  stream->sampleCursor = 0;
  stream->sampleCount = readSamples.sampleCount;
}

static GrainManager
initializeGrainManager(PluginState *pluginState)
{
  GrainManager result = {};
  result.windowBuffer[0] = arenaPushArray(pluginState->permanentArena, WINDOW_LENGTH, r32,
                                          arenaFlagsNoZeroAlign(4*sizeof(r32)));
  result.windowBuffer[1] = arenaPushArray(pluginState->permanentArena, WINDOW_LENGTH, r32,
                                          arenaFlagsNoZeroAlign(4*sizeof(r32)));
  result.windowBuffer[2] = arenaPushArray(pluginState->permanentArena, WINDOW_LENGTH, r32,
                                          arenaFlagsNoZeroAlign(4*sizeof(r32)));
  result.windowBuffer[3] = arenaPushArray(pluginState->permanentArena, WINDOW_LENGTH, r32,
                                          arenaFlagsNoZeroAlign(4*sizeof(r32)));
  initializeWindows(&result);

  result.grainAllocator = pluginState->audioArena;

  result.firstPlayingGrain = 0;
  result.lastPlayingGrain = 0;
  result.grainCount = 0;
  result.grainFreeList = 0;

  result.stream.refill = grainManagerRefill;
  //result.sampleSource = &pluginState->inputStream.stream;
  result.sampleSource = &pluginState->pvStream.stream;

  result.parameters = pluginState->parameters;
  result.grainStateViewBuffer = &pluginState->grainStateViewBuffer;

  result.internalBuffer = &pluginState->grainInputBuffer;
  result.outputBuffer = &pluginState->grainOutputBuffer;

  return(result);
}
