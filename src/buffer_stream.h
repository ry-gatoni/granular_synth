struct BufferStream;
#define BUFFER_STREAM_REFILL_PROC(name) void (name) (BufferStream *stream)
typedef BUFFER_STREAM_REFILL_PROC(BufferStreamRefill);

struct BufferStream
{
  u8 *start;
  u8 *end;
  u8 *at;

  BufferStreamRefill *refill;
};

struct AudioBufferStream;
#define AUDIO_BUFFER_STREAM_REFILL_PROC(name) void (name) (AudioBufferStream *stream)
typedef AUDIO_BUFFER_STREAM_REFILL_PROC(AudioBufferStreamRefill);

struct AudioBufferStream
{
  r32 *startSamples[2];
  
  u32 sampleCount;
  u32 sampleCursor;

  AudioBufferStreamRefill *refill;
};
