#if 1

// NOTE: This structure is designed to provide contiguous views of samples when
// reading. This is achieved using virtual memory tricks when the host platform
// supports it, or by writing to two contiguous regions of memory otherwise. See
// https://fgiesen.wordpress.com/2012/07/21/the-magic-ring-buffer/ for details.
struct AudioRingBuffer
{
  SamplePair *samples;
  /* r32 *samples[2]; */
  u64 sampleCount;
  u64 writeIndex;
  u64 readIndex;
  b32 isMagic;
};

static b32
rbInit(AudioRingBuffer *rb, u64 sampleCount, b32 isMagic)
{
  /* void *lSampleMemory = 0; */
  /* void *rSampleMemory = 0; */
  void *sampleMemory = 0;

  sampleCount = ROUND_UP_POW_2(sampleCount);
  // usz bufferSizeInBytes = sampleCount*sizeof(r32);
  usz bufferSizeInBytes = sampleCount*sizeof(SamplePair);
  /* lSampleMemory = gsAllocateRingBufferMemory(&bufferSizeInBytes); */
  /* rSampleMemory = gsAllocateRingBufferMemory(&bufferSizeInBytes); */
  sampleMemory = gsAllocateRingBufferMemory(&bufferSizeInBytes);
  /* if(lSampleMemory == 0 || rSampleMemory == 0) */
  if(sampleMemory == 0)
  {
    logString("rbInit: ring buffer memory allocation failure\n");
    /* if(lSampleMemory != 0) gsFreeRingBufferMemory(lSampleMemory, bufferSizeInBytes); */
    /* if(rSampleMemory != 0) gsFreeRingBufferMemory(rSampleMemory, bufferSizeInBytes); */
    ZERO_STRUCT(rb);
    return(0);
  }

  /* rb->sampleCount = bufferSizeInBytes/sizeof(r32); */
  rb->sampleCount = bufferSizeInBytes/sizeof(SamplePair);
  /* rb->samples[0] = (r32*)lSampleMemory; */
  /* rb->samples[1] = (r32*)rSampleMemory; */
  rb->samples = (SamplePair*)sampleMemory;
  rb->writeIndex = 0;
  rb->readIndex = 0;
  rb->isMagic = isMagic;
  return(1);
}

#if 0
static inline u64
rbAvailableReadSampleCount(AudioRingBuffer *rb)
{
  u64 result = rb->writeIndex - rb->readIndex;
  return(result);
}

static inline void
rbReadFromBuffer(AudioRingBuffer *rb, SamplePair *dest, u64 sampleCount)
{
  u64 readIndexWrapped = (rb->readIndex & (rb->sampleCount - 1));
  u64 samplesToBufferEnd = rb->sampleCount - readIndexWrapped;
  u64 samplesToRead = MIN(sampleCount, samplesToBufferEnd);
  COPY_ARRAY(dest, rb->samples + readIndexWrapped, samplesToRead, SamplePair);
  if(samplesToRead < sampleCount)
  {
    u64 samplesRemaining = sampleCount - samplesToBufferEnd;
    COPY_ARRAY(dest + samplesToBufferEnd, rb->samples, samplesRemaining, SamplePair);
  }
  rb->readIndex += sampleCount;
}
#endif

struct AudioRingBufferView
{
  SamplePair *start;
  SamplePair *end;
};

static inline AudioRingBufferView
rbGetReadableView(AudioRingBuffer *rb)
{
  u64 readIndexWrapped = (rb->readIndex & (rb->sampleCount - 1));
  u64 sampleCount = rb->writeIndex - rb->readIndex;
  SamplePair *start = rb->samples + readIndexWrapped;
  SamplePair *end = start + sampleCount;
  rb->readIndex += sampleCount;

  AudioRingBufferView result = {};
  result.start = start;
  result.end = end;
  return(result);
}

static inline AudioRingBufferView
rbGetWritableView(AudioRingBuffer *rb, u64 sampleCount)
{
  u64 writeIndexWrapped = (rb->writeIndex & (rb->sampleCount - 1));
  u64 samplesAvailable = rb->sampleCount + rb->readIndex - rb->writeIndex;
  sampleCount = MIN(sampleCount, samplesAvailable);
  SamplePair *start = rb->samples + writeIndexWrapped;
  SamplePair *end = start + sampleCount;
  ZERO_ARRAY(start, sampleCount, SamplePair);

  AudioRingBufferView result = {};
  result.start = start;
  result.end = end;
  return(result);
}

static inline void
rbCommitWrite(AudioRingBuffer *rb, AudioRingBufferView view)
{
  u64 samplesWritten = INT_FROM_PTR(view.end - view.start);
  if(!rb->isMagic)
  {
    u64 startIndex = INT_FROM_PTR(view.start - rb->samples);
    u64 samplesToBufferEnd = rb->sampleCount - startIndex;
    u64 samplesToWrite = MIN(samplesWritten, samplesToBufferEnd);
    SamplePair *src = view.start;
    SamplePair *dest = rb->samples + rb->sampleCount + startIndex;
    COPY_ARRAY(dest, view.start, samplesToWrite, SamplePair);

    if(samplesToWrite < samplesWritten)
    {
      samplesToWrite = samplesWritten - samplesToWrite;
      src = view.start + samplesToWrite;
      dest = rb->samples;
      COPY_ARRAY(dest, src, samplesToWrite, SamplePair);
    }
  }
  rb->writeIndex += samplesWritten;
}

#if 0
static inline void
rbWriteToBuffer(AudioRingBuffer *rb, SamplePair *src, u64 sampleCount)
{
  u64 writeIndexWrapped = (rb->writeIndex & (rb->sampleCount - 1));
  u64 samplesToBufferEnd = rb->sampleCount - writeIndexWrapped;
  u64 samplesToWrite = MIN(sampleCount, samplesToBufferEnd);
  COPY_ARRAY(rb->samples + writeIndexWrapped, src, samplesToWrite, SamplePair);
  if(samplesToWrite < sampleCount)
  {
    u64 samplesRemaining = sampleCount - samplesToBufferEnd;
    COPY_ARRAY(rb->samples, src + samplesToBufferEnd, samplesRemaining, SamplePair);
  }
  rb->writeIndex += sampleCount;
}

static inline void
rbClearSamples(AudioRingBuffer *rb, u64 startIndex, u64 endIndex)
{
  ASSERT(startIndex < endIndex);
  ASSERT(endIndex <= rb->readIndex);
  u64 sampleCount = endIndex - startIndex;
  u64 startIndexWrapped = startIndex & (rb->sampleCount - 1);
  u64 samplesToBufferEnd = rb->sampleCount - startIndexWrapped;
  u64 samplesToClear = MIN(sampleCount, samplesToBufferEnd);
  ZERO_ARRAY(rb->samples + startIndexWrapped, samplesToClear, SamplePair);
  if(samplesToClear < sampleCount)
  {
    u64 samplesRemaining = sampleCount - samplesToClear;
    ZERO_ARRAY(rb->samples, samplesRemaining, SamplePair);
  }
}

#endif

#else
struct AudioRingBuffer
{
  r32 *samples[2];
  u32 capacity;

  u32 writeIndex;
  u32 readIndex;
};

inline u32
getAudioRingBufferOffset(AudioRingBuffer *rb)
{
  u32 offset = ((rb->writeIndex >= rb->readIndex) ?
                (rb->writeIndex - rb->readIndex) :
                (rb->capacity + rb->writeIndex - rb->readIndex));

  return(offset);
}

inline void
writeSamplesToAudioRingBuffer(AudioRingBuffer *rb, r32 *srcL, r32 *srcR, u32 count, bool increment = true)
{
  u32 samplesToBufferEnd = rb->capacity - rb->writeIndex;
  u32 samplesToCopy = MIN(samplesToBufferEnd, count);
  COPY_ARRAY(rb->samples[0] + rb->writeIndex, srcL, samplesToCopy, r32);
  COPY_ARRAY(rb->samples[1] + rb->writeIndex, srcR, samplesToCopy, r32);

  if(samplesToCopy < count)
    {
      u32 samplesRemaining = count - samplesToCopy;
      COPY_ARRAY(rb->samples[0], srcL + samplesToCopy, samplesRemaining, r32);
      COPY_ARRAY(rb->samples[1], srcR + samplesToCopy, samplesRemaining, r32);
    }

  if(increment)
    {
      rb->writeIndex = (rb->writeIndex + count) % rb->capacity;
    }
}

inline void
readSamplesFromAudioRingBuffer(AudioRingBuffer *rb, r32 *destL, r32 *destR, u32 count, bool increment = true)
{
  u32 samplesToBufferEnd = rb->capacity - rb->readIndex;
  u32 samplesToCopy = MIN(samplesToBufferEnd, count);
  COPY_ARRAY(destL, rb->samples[0] + rb->readIndex, samplesToCopy, r32);
  COPY_ARRAY(destR, rb->samples[1] + rb->readIndex, samplesToCopy, r32);

  if(samplesToCopy < count)
    {
      u32 samplesRemaining = count - samplesToCopy;
      COPY_ARRAY(destL + samplesToCopy, rb->samples[0], samplesRemaining, r32);
      COPY_ARRAY(destR + samplesToCopy, rb->samples[1], samplesRemaining, r32);
    }

  if(increment)
    {
      rb->readIndex = (rb->readIndex + count) % rb->capacity;
    }
}
#endif

struct SharedRingBuffer
{
  u8 *entries;
  u32 capacity;

  u32 writeIndex;
  u32 readIndex;
  volatile u32 queuedCount;
};

inline void
queueSharedRingBufferEntry_(SharedRingBuffer *rb, void *entry, usz entrySize)
{
  u8 *newEntry = rb->entries + entrySize*rb->writeIndex;
  COPY_SIZE(newEntry, entry, entrySize);

  rb->writeIndex = (rb->writeIndex + 1) % rb->capacity;
  u32 queuedCount = gsAtomicLoad(&rb->queuedCount);
  while(gsAtomicCompareAndSwap(&rb->queuedCount, queuedCount, queuedCount + 1) != queuedCount)
    {
      queuedCount = gsAtomicLoad(&rb->queuedCount);
    }
}

#define dequeueSharedRingBufferEntry(rb, type) (type *)dequeueSharedRingBufferEntry_((rb), sizeof(type))
#define dequeueAllSharedRingBufferEntries(dest, rb, type) dequeueAllSharedRingBufferEntry_(dest, (rb), sizeof(type))

inline u8 *
dequeueSharedRingBufferEntry_(SharedRingBuffer *rb, usz entrySize)
{
  u8 *entry = rb->entries + entrySize*rb->readIndex;

  rb->readIndex = (rb->readIndex + 1) % rb->capacity;
  u32 queuedCount = gsAtomicLoad(&rb->queuedCount);
  ASSERT(queuedCount);
  while(gsAtomicCompareAndSwap(&rb->queuedCount, queuedCount, queuedCount - 1) != queuedCount)
    {
      queuedCount = gsAtomicLoad(&rb->queuedCount);
    }

  return(entry);
}

inline void
dequeueAllSharedRingBufferEntries_(void *destInit, SharedRingBuffer *rb, usz entrySize)
{
  u8 *dest = (u8 *)destInit;
  u32 queuedCount = gsAtomicLoad(&rb->queuedCount);
  for(u32 entryIndex = 0; entryIndex < queuedCount; ++entryIndex)
    {
      u32 readIndex = (rb->readIndex + entryIndex) % rb->capacity;
      u8 *entry = rb->entries + readIndex*entrySize;
      COPY_SIZE(dest, entry, entrySize);

      dest += entrySize;
    }

  rb->readIndex = (rb->readIndex + queuedCount) % rb->capacity;
  u32 oldQueuedCount = queuedCount;
  while(gsAtomicCompareAndSwap(&rb->queuedCount, queuedCount, queuedCount - oldQueuedCount) != queuedCount)
    {
      queuedCount = gsAtomicLoad(&rb->queuedCount);
    }
}
