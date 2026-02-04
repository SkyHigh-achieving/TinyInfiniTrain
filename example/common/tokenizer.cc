#include "example/common/tokenizer.h"

#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "glog/logging.h"

namespace infini_train {

constexpr uint32_t kGpt2Eot = 50256;
constexpr uint32_t kLLaMA3Eot = 128001;
constexpr uint64_t kRandomU32Multiplier = 0x2545F4914F6CDD1Dull;
constexpr float kF32Divisor = 16777216.0f; // 2^24
constexpr uint64_t kRngState = 1337;

using Version = Tokenizer::Version;

const std::unordered_map<uint32_t, uint32_t> kEotMap = {
    {20240328, kGpt2Eot},   // GPT-2
    {20240801, kLLaMA3Eot}, // LLaMA-3
};

const std::unordered_map<uint32_t, std::vector<uint32_t>> kPromptMap = {
    // e.g. "The meaning of life is"
    // ref: https://tiktokenizer.vercel.app/
    {20240328, std::vector<uint32_t>{464, 3616, 286, 1204, 318}}, // GPT-2
    {20240801, std::vector<uint32_t>{791, 7438, 315, 2324, 374}}, // LLaMA-3
};

std::vector<uint8_t> ReadSeveralBytesFromIfstream(size_t num_bytes, std::ifstream *ifs) {
    std::vector<uint8_t> result(num_bytes);
    ifs->read(reinterpret_cast<char *>(result.data()), num_bytes);
    return result;
}

template <typename T> T BytesToType(const std::vector<uint8_t> &bytes, size_t offset) {
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable.");
    T value;
    std::memcpy(&value, &bytes[offset], sizeof(T));
    return value;
}

unsigned int RandomU32(uint64_t &state) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return (state * kRandomU32Multiplier) >> 32;
}

float RandomF32(uint64_t &state) { // random float32 in [0,1)
    return (RandomU32(state) >> 8) / kF32Divisor;
}

int SampleMult(float *probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from RandomF32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

Tokenizer::Tokenizer(const std::string &filepath) {
    std::ifstream ifs(filepath, std::ios::binary);
    CHECK(ifs.is_open()) << "Failed to open file: " << filepath;

    auto header_bytes = ReadSeveralBytesFromIfstream(1024, &ifs);
    magic_number_ = BytesToType<uint32_t>(header_bytes, 0);
    uint32_t version = BytesToType<uint32_t>(header_bytes, 4);
    vocab_size_ = BytesToType<uint32_t>(header_bytes, 8);

    auto it = kEotMap.find(magic_number_);
    CHECK(it != kEotMap.end()) << "Invalid magic number: " << magic_number_;
    eot_token_ = it->second;

    token_table_.resize(vocab_size_);
    for (uint32_t i = 0; i < vocab_size_; ++i) {
        uint8_t len;
        ifs.read(reinterpret_cast<char *>(&len), 1);
        std::string token(len, ' ');
        ifs.read(&token[0], len);
        token_table_[i] = std::move(token);
    }
}

std::string Tokenizer::Decode(uint32_t token_id) const {
    if (token_id < token_table_.size()) {
        return token_table_[token_id];
    }
    return "";
}

void Tokenizer::GenerateText(infini_train::nn::Module &model, uint32_t batch_size, uint32_t sequence_length,
                             uint32_t text_length, Device device) const {
    std::vector<int64_t> dims;
    dims.assign({batch_size, sequence_length});
    // x_tensor (FLAGS_batch_size, FLAGS_sequence_length) eq:(4, 64)
    infini_train::Tensor x_tensor = infini_train::Tensor(dims, DataType::kINT64);
    int64_t *x_buff = static_cast<int64_t *>(x_tensor.DataPtr());
    for (int i = 0; i < batch_size * sequence_length; ++i) { x_buff[i] = eot_token_; }

    // Give some contexts: "The meaning of life is "
    auto prompt = kPromptMap.at(magic_number_);
    auto prompt_len = prompt.size();
    for (int i = 0; i < prompt_len; ++i) { x_buff[i] = prompt[i]; }
    std::cout << "The meaning of life is";

    auto x = std::make_shared<infini_train::Tensor>(x_tensor.To(device));
    uint64_t state = kRngState;
    LOG(INFO) << "start generate text:";
    for (int t = prompt_len; t < text_length; t++) {
        auto outputs = model.Forward({x});
        auto logits = outputs[0]->To(Device(DeviceType::kCPU, 0));
        
        // logits: (batch_size, sequence_length, vocab_size)
        // 获取最后一个token对应的logits进行采样
        float *logits_ptr = static_cast<float *>(logits.DataPtr()) + (0 * sequence_length + (t - 1)) * vocab_size_;
        
        // Softmax with Temperature=1.0 and Top-k (optional, here we do simple sampling)
        float max_logit = logits_ptr[0];
        for (uint32_t i = 1; i < vocab_size_; ++i) {
            if (logits_ptr[i] > max_logit) max_logit = logits_ptr[i];
        }
        
        std::vector<float> probs(vocab_size_);
        float sum_exp = 0.0f;
        for (uint32_t i = 0; i < vocab_size_; ++i) {
            probs[i] = std::exp(logits_ptr[i] - max_logit);
            sum_exp += probs[i];
        }
        for (uint32_t i = 0; i < vocab_size_; ++i) {
            probs[i] /= sum_exp;
        }
        
        float coin = RandomF32(state);
        int next_token = SampleMult(probs.data(), vocab_size_, coin);
        
        std::cout << Decode(next_token);
        std::cout.flush();
        
        // 将新生成的token放入输入序列中
        if (t < sequence_length) {
            x_buff[t] = next_token;
            x = std::make_shared<infini_train::Tensor>(x_tensor.To(device));
        } else {
            // 如果超过最大序列长度，则停止生成或采取滑动窗口（此处作业要求通常为定长）
            break;
        }
    }
    std::cout << std::endl;
}
} // namespace infini_train
