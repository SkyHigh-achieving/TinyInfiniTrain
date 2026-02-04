#include <iostream>
#include <string>
#include <vector>
#include "example/gpt2/net.h"
#include "glog/logging.h"

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = 1;

    if (argc > 1 && std::string(argv[1]) == "--version") {
        std::cout << "TinyInfiniTrain Version 0.3.0" << std::endl;
        return 0;
    }

    std::cout << "TinyInfiniTrain GPT-2 Example" << std::endl;
    std::cout << "Usage: " << argv[0] << " [--version]" << std::endl;
    
    // Simple initialization check
    GPT2Config config;
    try {
        GPT2 model(config);
        std::cout << "Model initialized successfully with vocab_size: " << config.vocab_size << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize model: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
