
## 迭代记录 (Iteration Record)

| 修改文件 (File) | 修改原因 (Reason) | 性能提升指标 (Performance) | Commit ID |
|-----------------|-------------------|----------------------------|-----------|
| `kernels/cuda/linear.cu` | 实现了 MatmulForward/Backward | 相比 CPU 实现加速约 5-10x (Estimated) | - |
| `kernels/cuda/accumulate_grad.cu` | 实现了 Adam 优化器 | 减少 Host-Device 传输开销 | - |
| `example/common/tokenizer.cc` | 实现了 Tokenizer 解析 | 支持端到端文本生成 | - |
| `CMakeLists.txt` | (Planned) 添加 -O3 优化 | 预计提升 CPU 计算效率 20% | TBD |
