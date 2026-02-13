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

// static AudioRingBuffer
// initializeGrainBuffer(PluginState *pluginState, u32 bufferCount)
// {
//   AudioRingBuffer grainBuffer = {};
//   grainBuffer.capacity = bufferCount;
//   grainBuffer.samples[0] = arenaPushArray(pluginState->permanentArena, bufferCount, r32,
//                                           arenaFlagsNoZeroAlign(4*sizeof(r32)));
//   grainBuffer.samples[1] = arenaPushArray(pluginState->permanentArena, bufferCount, r32,
//                                           arenaFlagsNoZeroAlign(4*sizeof(r32)));

//   r32 offset = pluginReadFloatParameter(&pluginState->parameters[PluginParameter_offset]);
//   u32 offsetSamples = (u32)offset;
//   grainBuffer.writeIndex = offsetSamples;
//   grainBuffer.readIndex = 0;

//   return(grainBuffer);
// }

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
grainMakeViews(GrainManager *grainManager, AudioRingBufferView newSamples)
{
  GrainStateView *grainStateView = grainManager->grainStateView;
  AudioRingBuffer *grainBuffer = grainManager->internalBuffer;
  AudioRingBuffer *viewBuffer = grainStateView->viewBuffer;

  u64 samplesToWrite = INT_FROM_PTR(newSamples.end - newSamples.start);

  // NOTE: queue a new view and fill out its data
  u32 viewWriteIndex = (grainStateView->viewWriteIndex + 1) % ARRAY_COUNT(grainStateView->views);
  u32 entriesQueued = gsAtomicLoad(&grainStateView->entriesQueued);
  GrainBufferViewEntry *newView = grainStateView->views + viewWriteIndex;
  newView->grainCount = 0;

  // NOTE: fill view buffer with grain buffer samples
  {
    AudioRingBufferView destSamples = rbGetWritableView(viewBuffer, samplesToWrite);
    u64 availableWriteSamples = INT_FROM_PTR(destSamples.end - destSamples.start);
    ASSERT(samplesToWrite == availableWriteSamples);
    COPY_ARRAY(destSamples.start, newSamples.start, samplesToWrite, SamplePair);
    rbCommitWrite(viewBuffer, destSamples);
  }

  newView->bufferReadIndex = grainBuffer->readIndex & (grainBuffer->sampleCount - 1);
  newView->bufferWriteIndex = grainBuffer->writeIndex & (grainBuffer->sampleCount - 1);

  for(Grain *grain = grainManager->firstPlayingGrain; grain; grain = grain->next)
  {
    GrainViewEntry *grainView = newView->grainViews + newView->grainCount++;
    grainView->startIndex = grain->readIndex & (grainBuffer->sampleCount - 1);
    grainView->endIndex = (grain->readIndex + grain->samplesToPlay) & (grainBuffer->sampleCount - 1);
  }

  // NOTE: queue view entry
  grainStateView->viewWriteIndex = viewWriteIndex;
  u32 newEntriesQueued = (entriesQueued + 1) % ARRAY_COUNT(grainStateView->views);
  while(gsAtomicCompareAndSwap(&grainStateView->entriesQueued, entriesQueued, newEntriesQueued) != entriesQueued)
  {
    entriesQueued = gsAtomicLoad(&grainStateView->entriesQueued);
    newEntriesQueued = (entriesQueued + 1) % ARRAY_COUNT(grainStateView->views);
  }
}

static void
synthesize(GrainManager* grainManager, AudioRingBufferView dest)
{
  AudioRingBuffer *grainBuffer = grainManager->internalBuffer;

  u64 samplesToWrite = INT_FROM_PTR(dest.end - dest.start);

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

    // NOTE: process playing grains
    SamplePair *destSamples = dest.start;
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
          SamplePair grainSample = grainBuffer->samples[grainReadIndexWrapped];

          u32 samplesPlayed = grain->length - grain->samplesToPlay;
          r32 samplesPlayedFrac = (r32)samplesPlayed*grain->lengthInv;
          r32 windowVal = getWindowVal(grainManager, samplesPlayedFrac, grain->windowParam);

          r32 panL = 1.0f - MAX(0.0f, grain->stereoPosition);
          r32 panR = 1.0f + MIN(0.0f, grain->stereoPosition);

          r32 lVal = attenFactor * windowVal * panL * grainSample.left;
          r32 rVal = attenFactor * windowVal * panR * grainSample.right;

          destSamples[sampleIndex].left += lVal;
          destSamples[sampleIndex].right += rVal;

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
grainManagerRefill(BufferStream *stream)
{
  ASSERT(stream->at == stream->end);

  GrainManager *grainManager = (GrainManager*)stream;
  AudioRingBuffer *grainInputBuffer = grainManager->internalBuffer;
  AudioRingBuffer *grainOutputBuffer = grainManager->outputBuffer;

  // NOTE: refill our input buffers if needed
  BufferStream *sampleSource  = grainManager->sampleSource;
  if(sampleSource->at == sampleSource->end)
  {
    sampleSource->refill(sampleSource);
    ASSERT(sampleSource->at == sampleSource->start);
  }

  SamplePair *sampleSourceEnd = (SamplePair*)sampleSource->end;
  SamplePair *sampleSourceAt = (SamplePair*)sampleSource->at;
  u64 availableReadSamples = INT_FROM_PTR(sampleSourceEnd - sampleSourceAt);

  u64 availableWriteSamples = 0;
  // NOTE: fill the grain buffer
  {
    AudioRingBufferView destSamples = rbGetWritableView(grainInputBuffer, availableReadSamples);
    availableWriteSamples = INT_FROM_PTR(destSamples.end - destSamples.start);
    u64 samplesToWrite = MIN(availableReadSamples, availableWriteSamples);
    COPY_ARRAY(destSamples.start, sampleSourceAt, samplesToWrite, SamplePair);
    rbCommitWrite(grainInputBuffer, destSamples);
    sampleSource->at += samplesToWrite*sizeof(*sampleSourceAt);

    grainMakeViews(grainManager, destSamples);
  }

  AudioRingBufferView destSamples = rbGetWritableView(grainOutputBuffer, availableWriteSamples);
  synthesize(grainManager, destSamples);
  rbCommitWrite(grainOutputBuffer, destSamples);

  AudioRingBufferView readSamples = rbGetReadableView(grainOutputBuffer);
  stream->start = (u8*)readSamples.start;
  stream->at = stream->start;
  stream->end = (u8*)readSamples.end;
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
  result.sampleSource = &pluginState->inputStream.stream;

  result.parameters = pluginState->parameters;
  result.grainStateView = &pluginState->grainStateView;

// #define GRAIN_BUFFER_SAMPLE_COUNT (1ULL << 16)
//   STATIC_ASSERT(IS_POWER_OF_2(GRAIN_BUFFER_SAMPLE_COUNT), grainBufferSampleCountCheck);

  result.internalBuffer = &pluginState->grainInputBuffer;
  result.outputBuffer = &pluginState->grainOutputBuffer;

  return(result);
}
