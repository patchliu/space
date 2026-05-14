# Triton Ascend 分核与 Autotune 示例

本目录聚焦一个窄主题：**Triton-Ascend 上如何设计分核、逻辑核循环和 autotune 参数**。这里不展开完整 profiling、IR、反汇编流程，只关注会直接影响 kernel 写法和配置搜索的优化点。

## 关注范围

- **分核**：grid 应该贴近物理核数，而不是简单等于全部数据块数量。
- **逻辑核**：当逻辑 block 多于物理核时，在 kernel 内用跨步循环覆盖剩余 block。
- **切分**：`BLOCK_SIZE` 决定一次逻辑块规模，`BLOCK_SIZE_SUB` 决定核内子块搬运和 UB 压力。
- **autotune**：把核数、tile、sub-tile、`multibuffer`、Ascend 后端编译选项作为候选配置，而不是沿用 GPU 配置。

## 阅读顺序

1. [triton-ascend-vs-triton.md](./triton-ascend-vs-triton.md)：持续记录 Triton-Ascend 相比标准 Triton / GPU Triton 的关键差异。
2. [split-autotune-examples.md](./split-autotune-examples.md)：通过真实 Triton kernel 解释分核、逻辑核循环和 autotune。
3. 运行下面的配套脚本，用 profiling 和 dump IR 验证配置是否生效。

## 可执行用例

| 文件 | 说明 |
|------|------|
| [silu-mul-split.py](./silu-mul-split.py) | 对比 `grid = num_blocks` 和物理核 grid + 逻辑块跨步 |
| [silu-mul-autotune.py](./silu-mul-autotune.py) | 枚举 `NUM_CORE`、`BLOCK_SIZE`、`BLOCK_SIZE_SUB`、`multibuffer` 并选择 best config |
| [softmax-row-split.py](./softmax-row-split.py) | row-wise softmax 的行维分核与 `ROWS_PER_PROGRAM` |
| [scale-alias-multibuffer.py](./scale-alias-multibuffer.py) | 对比 in-place alias 和 out-of-place 对 MultiBuffer 的影响 |

示例运行：

```shell
TRITON_PRINT_AUTOTUNING=1 \
TRITON_ALWAYS_COMPILE=1 \
TRITON_DEBUG=1 \
TRITON_KERNEL_DUMP=1 \
TRITON_DUMP_DIR=./cache \
python knowledge/triton-ascend/silu-mul-autotune.py
```

如果当前后端不接受 launch 侧的 `multibuffer=True`，可先加 `--include-no-multibuffer` 或在单用例里加 `--no-multibuffer` 跑通基线，再对照 profiling 与 dump IR 调整。

## 主题对应关系

| 主题 | 本目录落地方式 |
|------------|----------------|
| 核数设置不合理 | 用固定物理核 grid + 核内跨步循环替代 `grid = ceil(n / block)` |
| 切分设置不合理 | 用 `BLOCK_SIZE` / `BLOCK_SIZE_SUB` 拆开总 tile 和 UB 子 tile |
| DoubleBuffer / MultiBuffer | 在 autotune 配置中把 `multibuffer` 作为候选项，并检查 alias 是否阻断 |
| 编译 / Autotune 选项未调优 | 把 Ascend 后端选项放进配置矩阵，按 profiling 选择 |
