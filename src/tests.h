struct TestResult
{
  b32 success;
  String8List log;
};

#define TEST_FN(name) TestResult(name)(Arena *arena, void *args)
typedef TEST_FN(TestFunction);

struct Tester
{
  String8 name;
  void *args;
  TestFunction *fn;
};

#define SYMBOL_SET_DEFINE TEST
#define TEST_Sym_Type Tester
#if COMPILER_MSVC
#  pragma section(".gs." STRINGIFY(SYMBOL_SET_DEFINE)"$a", read, write)
#  pragma section(".gs." STRINGIFY(SYMBOL_SET_DEFINE)"$i", read, write)
#  pragma section(".gs." STRINGIFY(SYMBOL_SET_DEFINE)"$z", read, write)
#  define TEST_Sym_Section ".gs."STRINGIFY(SYMBOL_SET_DEFINE)"$i"
// TODO: support msvc
#elif COMPILER_CLANG || COMPILER_GCC
#  define TEST_Sym_Marker GLUE(gs, SYMBOL_SET_DEFINE)
#  define TEST_Sym_Section STRINGIFY(TEST_Sym_Marker)//"gs" STRINGIFY(SYMBOL_SET_DEFINE)
#  define TEST_Sym_First GLUE(__start_, TEST_Sym_Marker)
#  define TEST_Sym_Last GLUE(__stop_, TEST_Sym_Marker)
// TODO: test on mac
#if OS_MAC
extern TEST_Sym_Type TEST_Sym_First[] asm("section$start$__DATA$" STRINGIFY(TEST_Sym_Marker));
extern TEST_Sym_Type TEST_Sym_Last[] asm("section$end$__DATA$" STRINGIFY(TEST_Sym_Marker));
#elif OS_LINUX
extern TEST_Sym_Type TEST_Sym_First[];
extern TEST_Sym_Type TEST_Sym_Last[];
#endif
static usz testGetCount(void) { return(INT_FROM_PTR(TEST_Sym_Last - TEST_Sym_First)); }
static TEST_Sym_Type* testGetFirst(void) { return(TEST_Sym_First); }
#else
#  error unsupported compiler
#endif

#define TEST_FN_DEF(name, args)\
  TEST_FN(GLUE(name, __testfn));\
  SECTION(TEST_Sym_Section) TEST_Sym_Type name = {STR8_LIT(#name), args, &GLUE(name, __testfn)};\
  TEST_FN(GLUE(name, __testfn))
