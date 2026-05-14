# Triton-Ascend 与标准 Triton 差异记录

本文用于持续记录 Triton-Ascend 相比标准 GPU Triton 的关键差异。重点不是介绍 Triton-Ascend 的完整用法，而是记录从常规 Triton / GPU Triton 迁移到 Ascend NPU 时，会影响 kernel 写法、grid 设计、tiling、autotune 和性能分析的区别。

参考上游文档：[Migrating Triton Operators from GPUs](https://github.com/Ascend/triton-ascend/blob/main/docs/en/migration_guide/migrate_from_gpu.md)。

## 记录原则

- 优先记录会改变代码结构或调参策略的差异。
- 每条差异尽量包含：标准 Triton 习惯、Triton-Ascend 现象、原因判断、迁移建议。
- 结论后续可继续补充来源、芯片型号、复现实验和反例。

## 1. Python 侧设备和运行时接口

### 结论

标准 Triton 示例通常围绕 CUDA 设备编写，迁移到 Triton-Ascend 时，应先把 Python 侧设备和运行时接口替换为 NPU 版本，再保持 kernel body 尽量不变做正确性验证。

常见改动：

- 增加 `import torch_npu`。
- 将 `device="cuda"`、`.cuda()`、`.to("cuda")` 改为 `device="npu"`、`.npu()`、`.to("npu")`。
- 移除或替换 `torch.cuda.*`、CUDA stream、CUDA event、CUDA synchronize 等 GPU 专用逻辑。
- 删除只服务于 GPU device discovery 的断言，例如围绕 `triton.runtime.driver.active.get_active_torch_device()` 的设备一致性检查。

### 迁移建议

先只改设备和 runtime 接口，保证 NPU tensor 可以触发 Triton-Ascend 编译并通过正确性测试；随后再处理 grid、tiling、UB 和性能问题。不要一开始同时改 kernel 结构和设备接口，否则很难判断错误来自语义迁移还是性能改写。

## 2. Grid / 逻辑核 / 物理核

### 结论

Triton-Ascend 上不应简单沿用 GPU Triton 的 `grid = num_tiles` 写法。更推荐让 grid 或 `blockDim` 贴近实际参与计算的物理核数，并在每个逻辑核内部用跨步循环覆盖剩余 tile。

标准 GPU Triton 中，grid 表示要提交的 program instance，也就是 CUDA 语境下的 CTA / thread block 数量。CTA 数量不需要等于 SM 数量，通常可以远大于 SM 数量，由 GPU runtime / hardware 动态调度到可用 SM 上分批执行。

Triton-Ascend 中，grid / `blockDim` 更接近本次 kernel 的显式分核数量，也就是要动用多少个逻辑核去映射物理 AI Core / Vector Core。逻辑核数量超过物理核后，额外任务不是低成本的 GPU CTA 队列模型，而更容易表现为分轮下发或分轮执行，并引入核启动、参数初始化等固定开销。因此实践上要求或强烈建议逻辑核数量不要超过对应的物理核数量。

### 对代码写法的影响

GPU Triton 常见写法：

```python
pid = tl.program_id(0)
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
```

这种写法默认每个 tile 对应一个 program instance，`grid = ceil(n / BLOCK_SIZE)` 可以远大于 SM 数量。

Triton-Ascend 更推荐写成：

```python
pid = tl.program_id(0)

for tile_id in range(pid, num_tiles, NUM_CORE):
    offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
```

其中 `NUM_CORE` 对齐物理核数或当前算子适合使用的核数，`num_tiles` 表示完整数据需要处理的逻辑 tile 数量。这样可以把“超过物理核数量的并行任务”留在核内循环中完成，而不是把 grid 放大到全部 tile 数。

### 调参建议

- 把 `NUM_CORE` / grid 作为 Ascend 侧 autotune 参数，而不是直接继承 GPU 上的 grid 设计。
- Vector-only 算子围绕 Vector Core 数量组织并发任务；包含 `tl.dot` 的算子围绕 AI Core 数量组织并发任务。
- 对 CV 分离、纯 Vector、Cube 参与等不同算子，分别确认应该对齐 AI Core 还是 Vector Core 数量。
- 小数据量场景下尤其要避免过量逻辑核，因为核启动和首轮初始化开销可能主导总耗时。
- 当 profiling 看到核数过大、单核工作量过小或固定开销占比高时，优先尝试减少 grid，并增加核内循环工作量。

## 3. Grid 维度和 `coreDim` 上限

### 结论

标准 Triton 中常用 2D / 3D grid 表达 tile 空间；Triton-Ascend 更推荐优先使用 1D grid。NPU 侧 2D 适配会被合并成 1D，例如 `(20,)` 和 `(4, 5)` 在执行结果上等价。

此外，Ascend NPU 的 `coreDim` 不能超过 `UINT16_MAX`，也就是 65535。大 shape 下如果直接按 `ceil(N / BLOCK_SIZE)` 生成 grid，可能触发 `coreDim=xxxx can't be greater than UINT16_MAX`。

### 迁移建议

- 优先把并发任务展平为 1D grid。
- 大 shape 下通过增大 `BLOCK_SIZE` 或增加核内循环控制 grid 数量。
- 需要满足 `ceil(N / BLOCK_SIZE) <= 65535`。
- 如果增大 `BLOCK_SIZE` 后触发 UB overflow，需要再引入 `BLOCK_SIZE_SUB` 做核内子块处理。
- 对没有顺序依赖的逻辑 programs，可评估 `TRITON_ALL_BLOCKS_PARALLEL=1`，但仍应结合实际 profiling 验证收益。

## 4. 单 program 数据搬运和对齐

### 结论

标准 GPU Triton 对非对齐访问通常更宽容，性能退化也常由 coalescing、cache、transaction 等角度分析。Triton-Ascend 对搬运对齐更敏感：Vector 算子需要关注 32-byte 访问对齐，Cube-Vector 融合类算子需要关注 512-byte 对齐。

尾块 mask 仍然需要保留，避免越界访问。但 mask、非连续 stride、离散访问在 Ascend 上可能带来额外 UB 对齐、搬运或标量化开销。

### 迁移建议

- 检查每个 program 内部 `tl.load` / `tl.store` 的地址是否连续、是否满足对齐要求。
- 对 2D 数据优先让最低维是连续维，使用合理的 `shape` / `strides` / `block_shape` / `order`。
- 避免把连续的二维数据误建模为带大 stride 的一维访问；例如 `[1024, 32]` 更适合用 `shape=(1024, 32)`、`strides=(32, 1)`、`block_shape=(BT, 32)` 表达。
- 保留边界 mask，但要关注 masked load / store 是否导致额外初始化或搬运开销。

## 5. UB 容量、主块和子块切分

### 结论

Triton-Ascend 的 tile 设计强依赖 UB 容量。标准 GPU Triton 里一个较大的 `BLOCK_SIZE` 可能只是影响 occupancy 或寄存器压力；在 Ascend 上，过大的单 program 数据量可能直接触发 `ub overflow, requires xxxx bits while xxxx bits available`。

`coreDim` 和 UB overflow 经常是联动问题：为降低 `coreDim` 而增大 `BLOCK_SIZE`，可能让单 program 的 UB 占用超过上限。

### 迁移建议

- 用较大的 `BLOCK_SIZE` 控制 grid / `coreDim`，再用 `BLOCK_SIZE_SUB` 控制每次实际搬运和计算的数据量。
- 在 kernel 内对一个主块做子块循环，避免单次 load / compute 占满 UB。
- double buffer / multi-buffer 开启后，可用 UB 通常会减少，调 tile 时需要预留空间。
- 看到 UB overflow 时，不要只盲目减小 `BLOCK_SIZE`；先判断它是否还承担控制 `coreDim` 的作用，再决定是否拆主块和子块。

示意写法：

```python
pid = tl.program_id(0)
base = pid * BLOCK_SIZE
num_sub_blocks = tl.cdiv(BLOCK_SIZE, BLOCK_SIZE_SUB)

for sub_id in range(num_sub_blocks):
    offsets = base + sub_id * BLOCK_SIZE_SUB + tl.arange(0, BLOCK_SIZE_SUB)
    mask = offsets < N
    vals = tl.load(x + offsets, mask=mask, other=0.0)
    tl.store(y + offsets, vals, mask=mask)
```

## 6. 单 program 计算和 dtype

### 结论

NPU 和 GPU 的计算单元、支持 dtype、矩阵乘路径和执行行为不同。标准 Triton 上可接受的 index dtype、accumulator dtype、输出 dtype 组合，迁移到 Triton-Ascend 后需要重新做正确性和性能验证。

### 迁移建议

- 对整数 index、offset、length 等，确认当前 dtype 是否适合 NPU 路径。
- 对包含 `tl.dot` 的算子，重点检查 M/N/K tiling、accumulator dtype 和 output dtype。
- 对长序列、长 hidden size、大 K 循环，优先用 tiling 控制单次搬运和计算规模。
- 不要假设 GPU 上的 `BLOCK_M` / `BLOCK_N` / `BLOCK_K`、`num_warps`、`num_stages` 是 Ascend 上的合理起点之外的最终配置。

## 7. Autotune 参数空间

### 结论

标准 GPU Triton 的调参重点通常是 tile、`num_warps`、`num_stages`、cache modifier 等。Triton-Ascend 上还要把核数、主块、子块、multi-buffer 和后端编译选项纳入搜索。

### 迁移建议

- 把 `NUM_CORE`、`BLOCK_SIZE`、`BLOCK_SIZE_SUB` 一起作为候选参数。
- 单独区分 Vector-only、Cube、Cube-Vector 混合算子的搜索空间。
- GPU 配置只能作为初始参考，不能当作 Ascend 最优配置。
- autotune 结果要结合 profiling、IR dump 和 UB 报错一起解释。

## 8. MultiBuffer / DoubleBuffer 和存算重叠

### 结论

GPU Triton 性能分析常围绕 occupancy、warp 调度、memory coalescing、L2 reuse 展开。Triton-Ascend 上还要重点确认 MTE 搬运和 Vector / Cube 计算是否重叠，MultiBuffer / DoubleBuffer 是否实际生效。

### 迁移建议

- 对搬运密集算子，检查是否打开并实际启用了 multi-buffer。
- 若存在 input/output alias 或 in-place 读写，编译器可能因为依赖关系无法安全做 ping-pong。
- 结合 ttadapter、HIVM IR、profiling 和流水图判断 MTE2 / MTE3 与计算阶段是否重叠。
- 如果 load 和 compute 明显串行，优先检查 UB 空间、依赖关系、tile 结构和 multi-buffer 选项。

## 9. IR 排查和标量化风险

### 结论

Triton-Ascend 迁移不能只看 Python kernel 是否正确，还需要看 lowering 后的 IR 是否保留了期望的向量化和连续搬运。离散访问、错误的 `block_ptr` 建模、低效 stride 可能导致纯标量搬运或计算，成为严重性能瓶颈。

### 迁移建议

- 设置 `TRITON_DEBUG=1`，保存 `~/.triton/cache/*.ttadapter`。
- 用 `bishengir-compile` 查看 HIVM 阶段 IR，重点看是否存在没有映射到 SIMD 的纯 scalar transfer / scalar compute。
- 逐行对照 Triton kernel 和 IR，确认核心 load / compute / store 是否按预期向量化。
- 对离散访问问题，优先重写 `make_block_ptr` 的 `shape`、`strides`、`offsets`、`block_shape` 和 `order`，让最低维保持连续访问。

## 待补充

- 不同 Ascend 芯片型号下 AI Core / Vector Core 数量的查询方式。
- `blockDim` 超过物理核时的具体 runtime 行为和可观测 profiling 指标。
- 与 Inductor 生成 Triton kernel 的 grid heuristic 对应关系。
- masked load / store 在不同 dtype 和 mask 形态下的实际开销。
