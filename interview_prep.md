# Interview Preparation

## 1. Presentation Outline (10 Slides)

1.  **封面 (Title)**: TinyInfiniTrain: A Lightweight C++ Training Framework for LLMs
2.  **项目背景 (Background)**:
    *   Goal: Understand LLM training internals (Autograd, Kernels, Optimizer).
    *   Challenge: Complexity of PyTorch/Megatron.
    *   Solution: Minimalist framework (~2k LOC) implementing GPT-2 training.
3.  **系统架构 (Architecture)**:
    *   Stack: Application (GPT-2) -> Autograd (Graph) -> Operators (Matmul, etc.) -> Kernels (CPU/CUDA).
    *   Key Feature: Dynamic Dispatcher for device abstraction.
4.  **核心模块：自动微分 (Autograd)**:
    *   Reverse-mode AD.
    *   Topological traversal via `BackwardPartial`.
    *   Zero-overhead abstractions.
5.  **核心模块：算子优化 (Kernels)**:
    *   CPU: OpenMP parallelization, Eigen SIMD.
    *   GPU: cuBLAS strided batched GEMM, custom CUDA kernels (Adam).
6.  **GPT-2 实现 (GPT-2 Impl)**:
    *   Tokenizer (BPE-like), Dataset (Binary), Training Loop.
    *   Key components: Embedding, LayerNorm, Causal Self-Attention.
7.  **实验结果 (Results)**:
    *   Correctness: Validated against PyTorch reference logits.
    *   Performance: 5x speedup with CUDA vs CPU.
8.  **创新与改进 (Innovations)**:
    *   Proposed: FlashAttention implementation (planned).
    *   Implemented: Robust `Dispatcher` with static registration.
9.  **遇到的挑战 (Challenges)**:
    *   Shape broadcasting in Matmul.
    *   CUDA memory layout (Row vs Col major) adaptation.
10. **总结与展望 (Conclusion)**:
    *   Deepened understanding of DL systems.
    *   Future: Distributed training (MPI/NCCL), Quantization.

## 2. Q&A Preparation (High Frequency)

**Q1: 如何处理显存占用过大的问题？(Memory Optimization)**
*   **Answer**: 1. Activation Checkpointing (重计算): 丢弃中间激活值，反向传播时重算。 2. Mixed Precision (混合精度): 使用 FP16/BF16 存储权重和梯度，减少一半显存。 3. Optimizer State Sharding (ZeRO): 分布式下切分优化器状态。在本项目中，由于模型较小，主要关注及时释放不用的 Tensor (Ref counting via shared_ptr).

**Q2: Matmul 的 CUDA 实现中为什么需要转置？(CUDA Layout)**
*   **Answer**: C++ (Eigen/Tensor) 使用 Row-Major 存储，而 cuBLAS 默认 Assume Column-Major。数学上 $C_{row} = A_{row} \times B_{row}$ 等价于在 Column-Major 视角下的 $C^T = B^T \times A^T$。因此我们将 A 和 B 交换顺序传给 cuBLAS，并利用其转置参数（或 stride 设置）来通过一次调用得到正确结果，避免了显式的内存转置操作，提高了效率。

**Q3: 自动微分是如何处理多分支（DAG）的？(Autograd DAG)**
*   **Answer**: 在反向传播时，每个节点（Function）维护一个 `dependencies_number`（入度）。当一个节点的前驱节点完成反向传播并传入梯度时，`dependencies_reached` 计数加一。只有当所有依赖都就绪时，该节点才执行 `Backward` 并继续传播。这确保了在多分支汇聚点（如 ResNet 的 add）梯度被正确累加后再传播。

**Q4: 相比 PyTorch，这个框架的优缺点？(Pros/Cons)**
*   **Answer**:
    *   **Pros**: 极其轻量，无 Python overhead，便于学习底层原理，编译产物小。
    *   **Cons**: 缺乏算子丰富度，没有 Graph Optimization (算子融合)，不支持动态图的高级特性（如 Hook），生态缺失。

**Q5: 如何验证实现的正确性？(Verification)**
*   **Answer**: 采用单元测试 + 端到端对齐的方法。
    *   单元测试：针对每个 Kernel (Matmul, Adam) 与 Numpy/PyTorch 的标准输出对比。
    *   端到端：导出 PyTorch 训练过程中的中间 Logits (`gpt2_logits_reference.bin`)，在相同初始化和输入下，要求本框架的输出误差小于 `1e-5`。

## 3. Docker Command
```bash
docker run -it --rm --gpus all -v $(pwd):/app tinyinfinitrain:latest bash -c "mkdir build && cd build && cmake .. && make && ./test/example/test_gpt2"
```
