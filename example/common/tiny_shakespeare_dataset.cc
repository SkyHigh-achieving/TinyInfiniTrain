#include "example/common/tiny_shakespeare_dataset.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace {
using DataType = infini_train::DataType;
using TinyShakespeareType = TinyShakespeareDataset::TinyShakespeareType;
using TinyShakespeareFile = TinyShakespeareDataset::TinyShakespeareFile;

const std::unordered_map<int, TinyShakespeareType> kTypeMap = {
    {20240520, TinyShakespeareType::kUINT16}, // GPT-2
    {20240801, TinyShakespeareType::kUINT32}, // LLaMA 3
};

const std::unordered_map<TinyShakespeareType, size_t> kTypeToSize = {
    {TinyShakespeareType::kUINT16, 2},
    {TinyShakespeareType::kUINT32, 4},
};

const std::unordered_map<TinyShakespeareType, DataType> kTypeToDataType = {
    {TinyShakespeareType::kUINT16, DataType::kUINT16},
    {TinyShakespeareType::kUINT32, DataType::kINT32},
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

TinyShakespeareFile ReadTinyShakespeareFile(const std::string &path, size_t sequence_length) {
    /* =================================== 作业 ===================================
       TODO：实现二进制数据集文件解析
       文件格式说明：
    ----------------------------------------------------------------------------------
    | HEADER (1024 bytes)                     | DATA (tokens)                        |
    | magic(4B) | version(4B) | num_toks(4B) | reserved(1012B) | token数据           |
    ----------------------------------------------------------------------------------
       =================================== 作业 =================================== */
    std::ifstream ifs(path, std::ios::binary);
    CHECK(ifs.is_open()) << "Failed to open file: " << path;

    std::vector<uint8_t> header = ReadSeveralBytesFromIfstream(1024, &ifs);
    int32_t magic = BytesToType<int32_t>(header, 0);
    int32_t version = BytesToType<int32_t>(header, 4);
    int32_t num_toks = BytesToType<int32_t>(header, 8);

    CHECK(kTypeMap.count(magic)) << "Unknown magic number: " << magic;
    TinyShakespeareType type = kTypeMap.at(magic);
    size_t token_size = kTypeToSize.at(type);
    DataType dtype = kTypeToDataType.at(type);

    size_t data_size = num_toks * token_size;
    std::vector<uint8_t> data_bytes = ReadSeveralBytesFromIfstream(data_size, &ifs);
    
    std::vector<int64_t> dims = {num_toks};
    auto tensor = std::make_shared<infini_train::Tensor>(dims, dtype);
    std::memcpy(tensor->DataPtr(), data_bytes.data(), data_size);

    // Reshape to (num_samples, sequence_length)
    // We drop the last few tokens that don't fit
    // NOTE: operator[] assumes x and y (shifted).
    // If we treat it as sliding window or distinct chunks?
    // dims in struct is used for bounds check.
    // If we set dims to {num_samples, seq_len}, then dims[0] is num_samples.
    // operator[] checks idx < dims[0] - 1.
    // So we should set dims[0] to count of possible sequences.
    
    // Let's stick to flat tensor for now and let dataset handle logic?
    // But struct has `dims`.
    // I will return flat dims in tensor, but calculate logical dims for the struct.
    
    // Actually, looking at operator[]:
    // `std::vector<int64_t> dims = std::vector<int64_t>(text_file_.dims.begin() + 1, text_file_.dims.end());`
    // It expects text_file_.dims to be at least size 2? [num_samples, seq_len, ...]
    
    int64_t num_samples = num_toks / sequence_length; 
    std::vector<int64_t> logical_dims = {num_samples, static_cast<int64_t>(sequence_length)};
    
    TinyShakespeareFile file;
    file.tensor = *tensor;
    file.dims = logical_dims;
    file.type = type;
    return file;
}
} // namespace

TinyShakespeareDataset::TinyShakespeareDataset(const std::string &filepath, size_t sequence_length) 
    : sequence_length_(sequence_length) {
    // =================================== 作业 ===================================
    // TODO：初始化数据集实例
    // HINT: 调用ReadTinyShakespeareFile加载数据文件
    // =================================== 作业 ===================================
    text_file_ = ReadTinyShakespeareFile(filepath, sequence_length);
    const_cast<size_t&>(sequence_size_in_bytes_) = sequence_length * kTypeToSize.at(text_file_.type);
    const_cast<size_t&>(num_samples_) = text_file_.dims[0];
}

std::pair<std::shared_ptr<infini_train::Tensor>, std::shared_ptr<infini_train::Tensor>>
TinyShakespeareDataset::operator[](size_t idx) const {
    CHECK_LT(idx, text_file_.dims[0] - 1);
    std::vector<int64_t> dims = std::vector<int64_t>(text_file_.dims.begin() + 1, text_file_.dims.end());
    
    auto tensor_ptr = std::make_shared<infini_train::Tensor>(text_file_.tensor);
    size_t token_size = kTypeToSize.at(text_file_.type);
    
    // x: (seq_len), y: (seq_len) -> stack -> (bs, seq_len) (bs, seq_len)
    return {std::make_shared<infini_train::Tensor>(tensor_ptr, idx * sequence_size_in_bytes_, dims),
            std::make_shared<infini_train::Tensor>(tensor_ptr, idx * sequence_size_in_bytes_ + token_size,
                                                   dims)};
}

size_t TinyShakespeareDataset::Size() const { return num_samples_; }
