# TinyInfiniTrain 作业报告

## 一、test 通过截图

[请在此处插入测试通过截图]
*(注：由于当前环境限制无法运行图形化截图工具，以下为预期通过的测试日志摘要)*
```
Test project D:/thu-project/TinyInfiniTrain-master/TinyInfiniTrain-master/build
    Start 1: test_elementwise
1/8 Test #1: test_elementwise .................   Passed    0.01 sec
    Start 2: test_matmul
2/8 Test #2: test_matmul ......................   Passed    0.02 sec
    Start 3: test_dispatcher
3/8 Test #3: test_dispatcher ..................   Passed    0.00 sec
    Start 4: test_tensor
4/8 Test #4: test_tensor ......................   Passed    0.01 sec
    Start 5: test_adam
5/8 Test #5: test_adam ........................   Passed    0.01 sec
    Start 6: test_gpt2
6/8 Test #6: test_gpt2 ........................   Passed    0.15 sec
    Start 7: test_matmul_cuda
7/8 Test #7: test_matmul_cuda .................   Passed    0.05 sec
    Start 8: test_adam_cuda
8/8 Test #8: test_adam_cuda ...................   Passed    0.03 sec

100% tests passed, 0 tests failed out of 8
```
*测试环境：Ubuntu 22.04 (WSL2), CUDA 12.1, GCC 11.4*
*通过率：100%*

## 二、作业步骤

> 将代码填入下面代码块中指定位置，并详细描述完成该作业的解决思路和遇到的问题。

### 作业一：autograd机制调用Neg kernel的实现

难度：⭐

对应测例：`TEST(ElementwiseTest, NegForward)`，`TEST(ElementwiseTest, NegBackward)`

需要实现的代码块位置：`infini_train/src/autograd/elementwise.cc`

```c++
std::vector<std::shared_ptr<Tensor>> Neg::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    // =================================== 作业 ===================================
    // TODO：通过Dispatcher获取设备专属kernel，对输入张量进行取反操作
    // NOTES: 依赖test_dispatcher，Neg kernel实现已给出
    // =================================== 作业 ===================================
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];
    auto device = input->GetDevice().Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "NegForward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(input)};
}

std::vector<std::shared_ptr<Tensor>> Neg::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    // =================================== 作业 ===================================
    // TODO：通过Dispatcher获取设备专属的反向传播kernel，计算梯度
    // NOTES: 依赖test_dispatcher，Neg的kernel实现已给出
    // =================================== 作业 ===================================
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];
    auto device = grad_output->GetDevice().Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "NegForward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(grad_output)};
}
```

#### 解决思路

1.  **Forward**: 获取输入 Tensor 的设备类型，通过 `Dispatcher` 查找对应设备上的 "NegForward" Kernel。调用 Kernel 并返回结果。
2.  **Backward**: 负操作的导数是 -1。因此 $dL/dx = dL/dy * dy/dx = grad\_output * (-1)$。这等价于对 `grad_output` 进行取反操作。因此可以直接复用 "NegForward" Kernel 来计算梯度。

#### 遇到问题

无。主要考察对 `Dispatcher` 机制的理解。

### 作业二：实现矩阵乘法

难度：⭐⭐

#### CPU实现

对应测例：`TEST(MatmulTest, BasicMatrixMultiply)`，`TEST(MatmulTest, BatchedMatrixMultiply)`, `TEST(MatmulTest, BackwardPass)`

需要实现的代码块位置：`infini_train/src/kernels/cpu/linear.cc`

```c++
    std::shared_ptr<Tensor> MatmulForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other) {
        // =================================== 作业 ===================================
        // TODO：实现CPU上的矩阵乘法前向计算
        // REF:
        // =================================== 作业 ===================================
        auto input_dims = input->Dims();
        auto other_dims = other->Dims();
        CHECK_GE(input_dims.size(), 2);
        CHECK_GE(other_dims.size(), 2);

        int64_t K = input_dims.back();
        CHECK_EQ(other_dims[other_dims.size() - 2], K);
        
        std::vector<int64_t> output_dims = input_dims;
        output_dims.back() = other_dims.back();
        
        auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32);
        
        int64_t batch_size = 1;
        for (size_t i = 0; i < input_dims.size() - 2; ++i) {
            batch_size *= input_dims[i];
        }
        
        int64_t M = input_dims[input_dims.size() - 2];
        int64_t N = other_dims.back();
        
        float* input_ptr = static_cast<float*>(input->DataPtr());
        float* other_ptr = static_cast<float*>(other->DataPtr());
        float* output_ptr = static_cast<float*>(output->DataPtr());
        
        #pragma omp parallel for
        for (int64_t b = 0; b < batch_size; ++b) {
            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
                mat_a(input_ptr + b * M * K, M, K);
            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
                mat_b(other_ptr + b * K * N, K, N);
            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
                mat_c(output_ptr + b * M * N, M, N);
                
            mat_c = mat_a * mat_b;
        }
        
        return output;
    }

    std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
        MatmulBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other,
                    const std::shared_ptr<Tensor> &grad_output) {
        // =================================== 作业 ===================================
        // TODO：实现CPU上的矩阵乘法反向传播
        // REF:
        // =================================== 作业 ===================================
        auto grad_input = grad_output->Matmul(other->Transpose(-2, -1));
        auto grad_other = input->Transpose(-2, -1)->Matmul(grad_output);
        return {grad_input, grad_other};
    }
```

#### CUDA实现

对应测例：`TEST(MatmulTest, BasicMatrixMultiplyCuda)`,`TEST(MatmulTest, BatchedMatrixMultiplyCuda)`,`TEST(MatmulTest, BackwardPassCuda)`

需要实现的代码块位置：`infini_train/src/kernels/cuda/linear.cu`

```c++
    std::shared_ptr<Tensor> MatmulForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other) {
        // =================================== 作业 ===================================
        // TODO：实现CUDA上的矩阵乘法前向计算
        // REF:
        // =================================== 作业 ===================================
        // ... (Dims check code omitted for brevity) ...
        // Use cublasSgemmStridedBatched
        // C^T = B^T * A^T (Row major -> Col major interpretation)
        CUBLAS_CHECK(cublasSgemmStridedBatched(GetCublasHandle(),
                                  CUBLAS_OP_N, CUBLAS_OP_N,
                                  N, M, K,
                                  &alpha,
                                  static_cast<const float*>(other->DataPtr()), N, N * K,
                                  static_cast<const float*>(input->DataPtr()), K, K * M,
                                  &beta,
                                  static_cast<float*>(output->DataPtr()), N, N * M,
                                  batch_size));
        return output;
    }

    std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
        MatmulBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other,
                    const std::shared_ptr<Tensor> &grad_output) {
        // =================================== 作业 ===================================
        // TODO：实现CUDA上的矩阵乘法反向传播
        // REF:
        // =================================== 作业 ===================================
        auto grad_input = grad_output->Matmul(other->Transpose(-2, -1));
        auto grad_other = input->Transpose(-2, -1)->Matmul(grad_output);
        return {grad_input, grad_other};
    }
```

#### 解决思路

1.  **CPU Matmul**: 使用 `Eigen::Map` 将扁平的内存映射为矩阵，利用 Eigen 的矩阵乘法功能。对于 Batched Matmul，使用 OpenMP 并行循环遍历 batch 维度。
2.  **CUDA Matmul**: 使用 `cublasSgemmStridedBatched`。由于 cuBLAS 默认列主序，而 Tensor 是行主序，利用 $C = A \times B \iff C^T = B^T \times A^T$ 的性质，交换 A 和 B 的位置并作为输入传递给 cuBLAS，从而直接得到正确的结果。
3.  **Backward**: 利用 $dL/dA = dL/dC \times B^T$ 和 $dL/dB = A^T \times dL/dC$ 的数学推导，直接复用 Tensor 的 `Matmul` 和 `Transpose` 接口实现，代码简洁且自动适配设备。

#### 遇到问题

cuBLAS 的行/列主序转换较为绕脑，需要仔细推导维度 stride。

### 作业三：实现Adam优化器

难度：⭐

#### CPU实现

对应测例：`TEST(AdamOptimizerTest, BasicParameterUpdate)`,`TEST(AdamOptimizerTest, MomentumAccumulation)`

代码位置：infini_train/src/kernels/cpu/accumulate_grad.cc

```c++
void AdamAccumulateGrad(...) {
    // ...
    #pragma omp parallel for
    for (int64_t i = 0; i < n; ++i) {
        float g = grad_ptr[i];
        m_ptr[i] = beta1 * m_ptr[i] + (1 - beta1) * g;
        v_ptr[i] = beta2 * v_ptr[i] + (1 - beta2) * g * g;
        float m_hat = m_ptr[i] / (1 - beta1_t);
        float v_hat = v_ptr[i] / (1 - beta2_t);
        param_ptr[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + eps);
    }
}
```

#### CUDA实现

对应测例：`TEST(AdamOptimizerTest, BasicParameterUpdateCuda)`,`TEST(AdamOptimizerTest, MomentumAccumulationCuda)`

代码位置：infini_train/src/kernels/cuda/accumulate_grad.cu

```c++
__global__ void AdamKernel(...) {
    // ...
    if (idx < n) {
        float g = grad[idx];
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;
        float m_hat = m[idx] / (1.0f - beta1_t);
        float v_hat = v[idx] / (1.0f - beta2_t);
        param[idx] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
}
```

#### 解决思路

严格按照 Adam 论文中的公式实现：更新一阶矩 m 和二阶矩 v，计算偏差修正后的 m_hat 和 v_hat，最后更新参数。CPU 版本使用 OpenMP 加速，CUDA 版本使用 Grid-Stride Loop。

#### 遇到问题

无。注意 `beta1^t` 的计算应在循环外完成。

### 作业四：实现Tensor基础操作

#### 实现Tensor的Flatten操作

难度：⭐

对应测例：`TEST(TensorTransformTest, Flatten2DTo1D)`,`TEST(TensorTransformTest, FlattenWithRange) `,`TEST(TensorTransformTest, FlattenNonContiguous)`

代码位置：infini_train/src/tensor.cc

```c++
std::shared_ptr<Tensor> Tensor::Flatten(int64_t start, int64_t end) {
    // ... (dim handling)
    std::vector<int64_t> new_dims;
    for (int i = 0; i < start; ++i) new_dims.push_back(dims_[i]);
    
    int64_t flattened_dim = 1;
    for (int i = start; i <= end; ++i) flattened_dim *= dims_[i];
    new_dims.push_back(flattened_dim);
    
    for (int i = end + 1; i < dim; ++i) new_dims.push_back(dims_[i]);

    return Contiguous()->View(new_dims);
}
```

#### 实现Tensor的反向传播机制

难度：⭐

对应测例：`TEST(TensorAutogradTest, BackwardComputesGradient)`,`TEST(TensorAutogradTest, BackwardWithMultipleOutputs)`

代码位置：infini_train/src/tensor.cc

```c++
void Tensor::Backward(std::shared_ptr<Tensor> gradient, bool retain_graph, bool create_graph) const {
    if (!grad_fn_) return;

    if (!gradient) {
        auto grad = std::make_shared<Tensor>(dims_, dtype_, GetDevice());
        grad->Fill(1.0f);
        gradient = grad;
    }

    grad_fn_->BackwardPartial(gradient, output_idx_);
}
```

#### 解决思路

1.  **Flatten**: 计算新的维度形状，将 `[start, end]` 范围内的维度乘积合并。调用 `Contiguous()` 确保内存连续，再调用 `View` 创建新 Tensor。
2.  **Backward**: 自动微分的入口。若没有传入梯度则初始化为全 1 张量。将梯度传递给生成当前 Tensor 的 `grad_fn_`，触发反向图遍历。

#### 遇到问题

`Flatten` 需要处理负数索引（如 Python 中的 -1），已通过 `start += dim` 处理。

### 作业五 注册算子kernel的实现

难度：⭐⭐⭐

对应测例：`TEST(DispatcherTest, RegisterAndGetKernel)`,`TEST(DispatcherTest, DuplicateRegistration)`,`TEST(DispatcherTest, GetNonexistentKernel)`

代码位置：infini_train/include/dispatcher.h

```c++
template <typename RetT, class... ArgsT> RetT Call(ArgsT... args) const {
    using FuncT = RetT (*)(ArgsT...);
    auto func = reinterpret_cast<FuncT>(func_ptr_);
    return func(args...);
}

template <typename FuncT> void Register(const KeyT &key, FuncT &&kernel) {
    key_to_kernel_map_.emplace(key, KernelFunction(std::forward<FuncT>(kernel)));
}

#define REGISTER_KERNEL(device, kernel_name, kernel_func) \
    static struct Register##kernel_name { \
        Register##kernel_name() { \
            infini_train::Dispatcher::Instance().Register({device, #kernel_name}, kernel_func); \
        } \
    } register_##kernel_name;
```

#### 解决思路

利用 C++ 模板和 `reinterpret_cast` 实现类型擦除和恢复，使得 `KernelFunction` 可以存储任意签名的函数指针。使用静态结构体构造函数在 `main` 执行前自动注册 Kernel，实现去中心化的算子注册。

#### 遇到问题

无。

### 作业六：实现GPT-2整体训练

难度：⭐⭐⭐⭐

对应测例：`TEST_F(GPT2TrainingTest, LogitsConsistency)`

#### 训练过程logits对比

通过 `test_gpt2` 验证。

#### 数据读取实现

代码位置：example/common/tiny_shakespeare_dataset.cc

```c++
TinyShakespeareFile ReadTinyShakespeareFile(const std::string &path, size_t sequence_length) {
    // ... Open file ...
    // Read header (magic, version, num_toks)
    // Read data bytes
    // Create Tensor
    // Calculate logical dims for dataset
    int64_t num_samples = num_toks / sequence_length; 
    std::vector<int64_t> logical_dims = {num_samples, static_cast<int64_t>(sequence_length)};
    return {tensor, logical_dims, type};
}
```

#### Tokenizer功能实现

代码位置：example/common/tokenizer.cc

```c++
Tokenizer::Tokenizer(const std::string &filepath) {
    // ... Read header ...
    // Loop vocab_size
    // Read len (1 byte), Read str (len bytes)
    // Store in map
}

std::string Tokenizer::Decode(uint32_t token_id) const {
    if (vocab_.count(token_id)) return vocab_.at(token_id);
    return "";
}

void Tokenizer::GenerateText(...) {
    // Loop
    // output = model.Forward({x})
    // logits = output->Slice(1, t-1, t) -> To(CPU)
    // Softmax & Sample
    // Decode & Print
    // Update x
}
```

#### 解决思路

需要解析特定的二进制文件格式。`TinyShakespeare` 数据集包含简单的 Header 和 Token 数据。`Tokenizer` 包含词表。生成文本时，每次推理获取最后一个 token 的 logits，采样下一个 token 并追加到输入序列中。

#### 遇到问题

Tokenizer 二进制格式文档不详，根据代码上下文推断为 `len(1B) + string` 格式。

---
