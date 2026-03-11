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

struct GrainViewEntry
{
  u32 startIndex;
  u32 endIndex;
};

struct GrainBufferViewEntry
{
  u32 bufferReadIndex;
  u32 bufferWriteIndex;

  u32 grainCount;
  GrainViewEntry grainViews[32];
};

struct GrainStateView
{
  u32 viewReadIndex;
  u32 viewWriteIndex;
  volatile u32 entriesQueued;

  GrainBufferViewEntry views[32];

  AudioRingBuffer *viewBuffer;
};

struct GrainManager
{
  AudioBufferStream stream; // NOTE: must always be the first member (so we can do casting tricks)
  AudioBufferStream *sampleSource;

  Arena *grainAllocator;

  PluginFloatParameter *parameters;

  GrainStateView *grainStateView;

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

static GrainManager initializeGrainManager(PluginState *pluginState);
static void grainMakeViews(GrainManager *grainManager);
static void synthesize(GrainManager* grainManager, AudioRingBufferView dest);
static void grainManagerRefill(BufferStream *stream);
