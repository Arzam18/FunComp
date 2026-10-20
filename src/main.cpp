#include "bench.h"
#include "nnue.h"
#include "uci.h"
#include "position.h"
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    init_zobrist();
    // "chess bench [depth]" runs the bench and exits, for scripts and
    // testing frameworks.
    if (argc >= 2 && std::string(argv[1]) == "bench") {
        if (nnue::embedded()) nnue::load("<embedded>");  // as the UCI loop does
        Searcher searcher;
        bench(searcher, argc >= 3 ? std::stoi(argv[2]) : BENCH_DEPTH, std::cout);
        return 0;
    }
    uci_loop();
    return 0;
}
