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
    /* ===================================== 作业 =====================================
    TODO：实现Tokenizer二进制文件加载

    文件格式说明：
    ----------------------------------------------------------------------------------
    | HEADER (1024 bytes)                     | VOCAB TABLE                           |
    | magic(4B) | version(4B) | vocab_size(4B) | reserved(1012B) | token词表数据       |
    ----------------------------------------------------------------------------------
    ===================================== 作业 ===================================== */
    std::ifstream ifs(filepath, std::ios::binary);
    CHECK(ifs.is_open()) << "Failed to open tokenizer file: " << filepath;

    std::vector<uint8_t> header = ReadSeveralBytesFromIfstream(1024, &ifs);
    magic_number_ = BytesToType<uint32_t>(header, 0);
    // version at offset 4
    uint32_t vocab_size = BytesToType<uint32_t>(header, 8);
    
    CHECK(kEotMap.count(magic_number_)) << "Unknown magic number: " << magic_number_;
    eot_token_ = kEotMap.at(magic_number_);

    // Read Vocab Table
    // Format assumption: len (1 byte) + string content? Or len (4 bytes)?
    // Usually simple tokenizers use 1 byte len.
    // Let's assume 1 byte length (uint8_t) followed by bytes.
    for (uint32_t i = 0; i < vocab_size; ++i) {
        std::vector<uint8_t> len_byte = ReadSeveralBytesFromIfstream(1, &ifs);
        uint8_t len = len_byte[0];
        std::vector<uint8_t> str_bytes = ReadSeveralBytesFromIfstream(len, &ifs);
        std::string token_str(str_bytes.begin(), str_bytes.end());
        token_table_[i] = token_str;
    }
}

std::string Tokenizer::Decode(uint32_t token_id) const {
    if (token_table_.count(token_id)) {
        return token_table_.at(token_id);
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
    uint64_t kRngState = kRngState;
    LOG(INFO) << "start generate text:";
    for (int t = prompt_len; t < text_length; t++) {
        /* ===================================== 作业 =====================================
        TODO：实现单步文本生成逻辑
        HINT：调用model.Forward推理获取logits，根据推理结果进行随机采样，调用Decode获取文本结果
        ===================================== 作业 ===================================== */
        // Forward pass
        auto output = model.Forward({x})[0]; // (1, seq_len, vocab_size)
        
        // Get logits for the last token
        // output shape: (1, seq_len, vocab_size)
        // We need output[0, t-1, :] ?
        // Or output[0, -1, :]?
        // x has shape (1, seq_len). Filled up to t.
        // We want to predict t-th token (index t).
        // Input was x[0...seq_len-1].
        // At step t, we have valid tokens up to t-1.
        // We want to predict x[t].
        // The model output at position t-1 predicts x[t].
        
        // We need to slice the output to get logits at t-1.
        // output dims: (batch, seq, vocab)
        // Slice dim 1 at index t-1.
        auto logits = output->Slice(1, t - 1, t); // (batch, 1, vocab)
        
        // Squeeze to (vocab)
        // Slice returns (batch, 1, vocab).
        // Flatten or Squeeze?
        // We need to access data.
        
        // Copy logits to host to sample
        // If device is CUDA, we need to copy.
        // Tensor::To(CPU).
        auto logits_cpu = logits->To(infini_train::Device(infini_train::DeviceType::kCPU, 0));
        float* logits_ptr = static_cast<float*>(logits_cpu.DataPtr());
        size_t vocab_size = logits_cpu.Dims().back();
        
        // Softmax to get probabilities (optional if SampleMult handles logits, but usually it expects probs)
        // SampleMult expects probabilities summing to 1.
        // So we need to apply Softmax.
        // We can use helper or implement simple softmax here.
        // Or use Tensor::Softmax if available?
        // Tensor doesn't have Softmax method in `tensor.h` snippet.
        // `autograd/softmax.h` exists.
        
        // Let's implement simple softmax on CPU
        std::vector<float> probs(vocab_size);
        float max_logit = -1e9;
        for (size_t i = 0; i < vocab_size; ++i) max_logit = std::max(max_logit, logits_ptr[i]);
        
        float sum_exp = 0.0f;
        for (size_t i = 0; i < vocab_size; ++i) {
            probs[i] = std::exp(logits_ptr[i] - max_logit);
            sum_exp += probs[i];
        }
        for (size_t i = 0; i < vocab_size; ++i) probs[i] /= sum_exp;
        
        // Sample
        float coin = RandomF32(kRngState);
        int next_token = SampleMult(probs.data(), vocab_size, coin);
        
        // Decode and print
        std::string token_str = Decode(next_token);
        std::cout << token_str << std::flush;
        
        // Update x for next step
        // x is (1, seq_len).
        // x[0, t] = next_token.
        if (t < sequence_length) {
            // We need to update x_tensor (CPU) then copy to x (Device)?
            // x is shared_ptr to Tensor on Device.
            // We can't update it directly if on GPU.
            // But we have `x_buff` which is pointer to `x_tensor` (CPU) data.
            // Wait, `x` was created from `x_tensor.To(device)`.
            // `x_tensor` is local.
            x_buff[t] = next_token;
            // Re-upload x?
            // `x = std::make_shared<infini_train::Tensor>(x_tensor.To(device));`
            // This is inefficient but simple.
             x = std::make_shared<infini_train::Tensor>(x_tensor.To(device));
        }
    }
    std::cout << std::endl;
}
} // namespace infini_train
