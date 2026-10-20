// The release network compiled into the binary (3.0 on; CMake EMBED_NET):
// the file's bytes placed in the read-only data at build time by the
// assembler's .incbin, so the engine plays with its network without an
// EvalFile. Built without one, embedded_size() is 0 and EvalFile alone loads
// a network (the hand-written evaluation otherwise).
#include "nnue.h"

#if defined(FUNCOMP_EMBED_NET_PATH)
#if defined(_WIN32)
#define FUNCOMP_RODATA ".section .rdata,\"dr\"\n"
#else
#define FUNCOMP_RODATA ".section .rodata\n"
#endif
__asm__(FUNCOMP_RODATA
        ".balign 64\n"
        ".globl funcomp_embedded_net\n"
        "funcomp_embedded_net:\n"
        ".incbin \"" FUNCOMP_EMBED_NET_PATH "\"\n"
        ".globl funcomp_embedded_net_end\n"
        "funcomp_embedded_net_end:\n"
        ".byte 0\n"
        ".text\n");
extern "C" const unsigned char funcomp_embedded_net[];
extern "C" const unsigned char funcomp_embedded_net_end[];

namespace nnue {
const unsigned char* embedded_data() { return funcomp_embedded_net; }
std::size_t embedded_size() {
    return static_cast<std::size_t>(funcomp_embedded_net_end - funcomp_embedded_net);
}
}  // namespace nnue
#else
namespace nnue {
const unsigned char* embedded_data() { return nullptr; }
std::size_t embedded_size() { return 0; }
}  // namespace nnue
#endif
