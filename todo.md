# Todo List for Replication & Improvement

## 1. 复现流程 (Replication)

| ID | 任务 (Task) | 命令 (Command) | 预期结果 (Expected) | 实际结果 (Actual) | 备注 (Notes) |
|----|-------------|----------------|---------------------|-------------------|--------------|
| 1  | 克隆仓库    | `git clone <repo>` | 成功下载代码 | 成功 | - |
| 2  | 下载数据    | `./download_starter_pack.sh` | Data目录包含bin文件 | 失败 | 网络问题，需手动下载 |
| 3  | 编译项目    | `cmake -B build -DUSE_CUDA=OFF && cmake --build build` | 生成可执行文件 | 失败 | 缺少 cmake/make 环境 |
| 4  | 运行测试    | `cd build && make test` | All tests passed | - | 无法运行 |

## 2. 改进计划 (Improvements)

### Refactor Branches

| 分支名 (Branch) | 改进项 (Item) | 描述 (Description) | Commit ID |
|-----------------|---------------|--------------------|-----------|
| `refactor/cmake-opt` | 编译优化 | 添加 `-O3 -march=native` 编译选项 | TBD |
| `refactor/openmp-fix` | 并行加速 | 修复 Windows 下 OpenMP 链接问题 | TBD |
| `feat/data-loader` | 数据加载 | 优化 `TinyShakespeareDataset` 的内存映射读取 (mmap) | TBD |

### 具体 Diff 说明

#### 1. CMakeLists.txt 优化

```diff
- set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
+ set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
```

#### 2. Matmul 优化 (Tiling)

```diff
// linear.cc
// Add blocking to optimize cache usage
+ int block_size = 32;
+ for (int ii = 0; ii < M; ii += block_size)
+   for (int jj = 0; jj < N; jj += block_size)
+     for (int kk = 0; kk < K; kk += block_size)
+       // inner loops...
```
