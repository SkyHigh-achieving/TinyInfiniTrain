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
    std::ifstream ifs(path, std::ios::binary);
    CHECK(ifs.is_open()) << "Failed to open file: " << path;

    auto header_bytes = ReadSeveralBytesFromIfstream(1024, &ifs);
    uint32_t magic = BytesToType<uint32_t>(header_bytes, 0);
    uint32_t version = BytesToType<uint32_t>(header_bytes, 4);
    uint32_t num_toks = BytesToType<uint32_t>(header_bytes, 8);

    CHECK_EQ(magic, 20240519) << "Invalid magic number";
    auto it = kTypeMap.find(version);
    CHECK(it != kTypeMap.end()) << "Invalid version: " << version;
    TinyShakespeareType type = it->second;

    DataType dtype = kTypeToDataType.at(type);
    size_t element_size = kTypeToSize.at(type);
    size_t total_bytes = static_cast<size_t>(num_toks) * element_size;

    infini_train::Tensor tensor({static_cast<int64_t>(num_toks)}, dtype);
    ifs.read(reinterpret_cast<char *>(tensor.DataPtr()), total_bytes);

    return {type, {static_cast<int64_t>(num_toks)}, std::move(tensor)};
}
} // namespace

TinyShakespeareDataset::TinyShakespeareDataset(const std::string &filepath, size_t sequence_length)
    : sequence_length_(sequence_length),
      text_file_(ReadTinyShakespeareFile(filepath, sequence_length)),
      sequence_size_in_bytes_(sequence_length * kTypeToSize.at(text_file_.type)),
      num_samples_((text_file_.dims[0] - 1) / sequence_length) {
    // Update dims for operator[]
    text_file_.dims = {static_cast<int64_t>(num_samples_), static_cast<int64_t>(sequence_length_)};
}

std::pair<std::shared_ptr<infini_train::Tensor>, std::shared_ptr<infini_train::Tensor>>
TinyShakespeareDataset::operator[](size_t idx) const {
    CHECK_LT(idx, num_samples_);
    std::vector<int64_t> dims = {static_cast<int64_t>(sequence_length_)};
    size_t element_size = sequence_size_in_bytes_ / sequence_length_;
    return {std::make_shared<infini_train::Tensor>(text_file_.tensor, idx * sequence_size_in_bytes_, dims),
            std::make_shared<infini_train::Tensor>(text_file_.tensor, idx * sequence_size_in_bytes_ + element_size,
                                                   dims)};
}

size_t TinyShakespeareDataset::Size() const { return num_samples_; }
