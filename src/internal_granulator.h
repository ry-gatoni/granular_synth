#define WINDOW_LENGTH 1024

struct Grain
{
  Grain* next;
  Grain* prev;

  // TODO: we can probably store less stuff in the grain
  u32 readIndex;
  r32 windowParam;

  s32 samplesToPlay;
  u32 length;
  r32 lengthInv;
  r32 stereoPosition;

  u32 startSampleIndex; // NOTE: the sample index in the processing loop at which we start processing this grain
  b32 isFinished;
};

struct GrainView
{
  u32 startIndex;
  u32 endIndex;
};

struct GrainStateView
{
  r32 *samples[2];
  u32 sampleCount;

  u32 readIndex;
  u32 writeIndex;

  u32 grainCount;
  GrainView grainViews[32];
};

#define PENDING_WRITE_POINTER_TAG_BIT 1ULL
struct GrainStateViewBuffer
{
  CACHE_ALIGN_FIELD GrainStateView *volatile read;
  CACHE_ALIGN_FIELD GrainStateView *volatile write;
  CACHE_ALIGN_FIELD GrainStateView *volatile shared;
};

struct GrainManager
{
  AudioBufferStream stream; // NOTE: must always be the first member (so we can do casting tricks)
  AudioBufferStream *sampleSource;

  Arena *grainAllocator;

  PluginFloatParameter *parameters;

  GrainStateViewBuffer *grainStateViewBuffer;

  u32 grainCount;
  Grain *firstPlayingGrain;
  Grain *lastPlayingGrain;

  Grain *grainFreeList;

  AudioRingBuffer *internalBuffer;
  AudioRingBuffer *outputBuffer;

  u32 samplesProcessedSinceLastSeed;

  r32 *windowBuffer[WindowShape_count];
};

// NOTE: functions

static inline void
enqueueGrainStateView(GrainManager *grainManager)
{
  GrainStateViewBuffer *viewBuffer = grainManager->grainStateViewBuffer;

  GrainStateView *write = viewBuffer->write;
  AudioRingBuffer *grainBuffer = grainManager->internalBuffer;
  COPY_ARRAY(write->samples[0], grainBuffer->samples[0], grainBuffer->sampleCount, r32);
  COPY_ARRAY(write->samples[1], grainBuffer->samples[1], grainBuffer->sampleCount, r32);
  write->readIndex = grainBuffer->readIndex & (grainBuffer->sampleCount - 1);
  write->writeIndex = grainBuffer->writeIndex & (grainBuffer->sampleCount - 1);
  write->grainCount = grainManager->grainCount;

  GrainView *grainView = &write->grainViews[0];
  for(Grain *grain = grainManager->firstPlayingGrain; grain; grain = grain->next)
  {
    grainView->startIndex = grain->readIndex & (grainBuffer->sampleCount - 1);
    grainView->endIndex = (grain->readIndex + grain->samplesToPlay) & (grainBuffer->sampleCount - 1);
    ++grainView;
  }

  usz taggedWrite = INT_FROM_PTR(write)|PENDING_WRITE_POINTER_TAG_BIT;
  void *shared = gsAtomicExchangePointers((void *volatile*)&viewBuffer->shared, PTR_FROM_INT(taggedWrite));
  usz untaggedShared = INT_FROM_PTR(shared) & ~PENDING_WRITE_POINTER_TAG_BIT;
  viewBuffer->write = (GrainStateView*)PTR_FROM_INT(untaggedShared);
}

static inline GrainStateView*
dequeueGrainStateView(GrainManager *grainManager)
{
  GrainStateViewBuffer *viewBuffer = grainManager->grainStateViewBuffer;

  if(INT_FROM_PTR(viewBuffer->shared) & PENDING_WRITE_POINTER_TAG_BIT)
  {
    void *shared = gsAtomicExchangePointers((void *volatile*)&viewBuffer->shared, viewBuffer->read);
    usz untaggedShared = INT_FROM_PTR(shared) & ~PENDING_WRITE_POINTER_TAG_BIT;
    viewBuffer->read = (GrainStateView*)PTR_FROM_INT(untaggedShared);
  }

  GrainStateView *result = viewBuffer->read;
  return result;
}

static GrainManager initializeGrainManager(PluginState *pluginState);
static void synthesize(GrainManager* grainManager, AudioRingBufferView dest);
static void grainManagerRefill(BufferStream *stream);
