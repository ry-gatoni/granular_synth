#if !defined(RIFF)

#define RIFF(str) FOURCC(str)

union RiffID
{
  u32 id;
  u8 str[4];
};

#pragma pack(push, 1)
struct RiffHeader
{
  RiffID chunkID;
  u32 chunkSize;
};
#pragma pack(pop)

#endif

#pragma pack(push, 1)
struct WaveHeader
{
  RiffID waveID;
};

struct WaveFormatChunk
{
  RiffHeader header;
  u16 formatTag;
  u16 channelCount;
  u32 sampleRate;
  u32 avgBytesPerSec;
  u16 dataBlockSize;
  u16 bitsPerSample;
};

struct WaveFormatExtension
{
  u16 cbSize;
  u16 validBitsPerSample;
  u32 channelMask;
  u8 subFmt[16];
};

struct WaveFormatExtended
{
  WaveFormatChunk fmt;
  WaveFormatExtension ex;
};
#pragma pack(pop)

typedef RiffHeader WaveDataChunk;
