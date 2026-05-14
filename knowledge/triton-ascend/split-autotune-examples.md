# Triton-Ascend 分核、逻辑核与 Autotune 示例

本文把 Triton-Ascend 上与分核和 autotune 相关的建议，改写成可直接套用的 Triton kernel 结构。示例重点不是数学算子本身，而是：

- grid 怎么和物理核数对齐；
- 逻辑 block 多于物理核时，kernel 内怎么跨步覆盖；
- 哪些参数应该进入 `triton.autotune`；
- 为什么 GPU 上的 block 配置只能作为 NPU 首版基线。

## 1. 先分清三个数量

在 Triton-Ascend 上调分核时，建议始终把三个数量分开：

| 名称 | 含义 | 常见写法 |
|------|------|----------|
| 物理核数 | 当前设备可并行执行的 AIV 或 AIC 数量 | `num_core` |
| 逻辑块数 | 数据按 `BLOCK_SIZE` 切出来的任务数 | `num_blocks = cdiv(n_elements, BLOCK_SIZE)` |
| kernel grid | 本次 launch 实际下发的 program 数 | `grid = (min(num_blocks, num_core),)` |

不要把 `grid = (num_blocks,)` 当成默认答案。`num_blocks` 很大时会造成多轮下发和固定头开销；`num_blocks` 很小时又可能打不满硬件。更稳的结构是：**grid 对齐物理核数，逻辑块在 kernel 内用 `while block_id < num_blocks` 跨步循环**。

下面的辅助函数只表达意图，实际字段名需要按当前 Triton-Ascend / torch_npu 版本确认。

```python
def get_num_vector_core(device=0):
    from triton.runtime import driver

    props = driver.active.utils.get_device_properties(device)
    return (
        props.get("num_vectorcore")
        or props.get("num_aiv")
        or props.get("multi_processor_count")
    )
```

## 2. 示例一：1D SiLU-Mul 融合

这是一个真实的 Vector 融合 kernel：读取 `x` 和 `y`，计算 `silu(x) * y`，写入 `out`。它适合讲清楚分核逻辑，因为计算是连续 1D 访存，主要可调项就是 grid、block、sub-block、multibuffer。

配套可执行文件：

- [silu-mul-split.py](./silu-mul-split.py)
- [silu-mul-autotune.py](./silu-mul-autotune.py)

### 2.1 容易从 GPU 迁移来的写法

GPU 写法通常直接让每个 Triton program 处理一个 block：

```python
import triton
import triton.language as tl


@triton.jit
def silu_mul_naive_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x * tl.sigmoid(x) * y
    tl.store(out_ptr + offsets, out, mask=mask)


def launch_naive(x, y, out):
    n_elements = x.numel()
    block_size = 1024
    grid = (triton.cdiv(n_elements, block_size),)
    silu_mul_naive_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=block_size)
```

这段代码本身没有语义问题，但在 Ascend 上有两个常见性能风险：

- `grid` 跟着逻辑块数增长，可能远大于物理 Vector Core 数，固定 launch / program 头开销变重。
- `BLOCK_SIZE` 同时承担“单个逻辑任务规模”和“单轮 UB 搬运规模”，一旦调大可能 UB overflow，调小又可能 MTE 搬运太碎。

### 2.2 Ascend 更推荐的分核结构

下面把 `NUM_CORE` 和 `num_blocks` 拆开。每个物理 program 先处理自己的 `pid`，再以 `NUM_CORE` 为步长继续处理后续逻辑块。

```python
import triton
import triton.language as tl


@triton.jit
def silu_mul_split_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    NUM_CORE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_SUB: tl.constexpr,
):
    pid = tl.program_id(0)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)

    block_id = pid
    while block_id < num_blocks:
        block_base = block_id * BLOCK_SIZE

        for sub_offset in tl.static_range(0, BLOCK_SIZE, BLOCK_SIZE_SUB):
            offsets = block_base + sub_offset + tl.arange(0, BLOCK_SIZE_SUB)
            mask = offsets < n_elements

            x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
            out = x * tl.sigmoid(x) * y
            tl.store(out_ptr + offsets, out, mask=mask)

        block_id += NUM_CORE
```

对应 launch：

```python
def launch_split(x, y, out, num_core):
    n_elements = x.numel()
    block_size = 4096
    block_size_sub = 1024
    num_blocks = triton.cdiv(n_elements, block_size)
    grid = (min(num_blocks, num_core),)

    silu_mul_split_kernel[grid](
        x,
        y,
        out,
        n_elements,
        NUM_CORE=num_core,
        BLOCK_SIZE=block_size,
        BLOCK_SIZE_SUB=block_size_sub,
        multibuffer=True,
    )
```

这个结构对应三个优化点：

- **核数设置**：`grid` 不盲目放大到 `num_blocks`，而是贴近物理核数。
- **逻辑核循环**：`block_id += NUM_CORE` 让每个物理核处理多个逻辑块。
- **切分设置**：`BLOCK_SIZE` 是逻辑块规模，`BLOCK_SIZE_SUB` 是核内子块规模，用来控制 UB 压力和 MTE 粒度。

### 2.3 给 1D 融合加 autotune

这个 kernel 的 autotune 不应该只调 `BLOCK_SIZE`。更有价值的是一起调：

- `BLOCK_SIZE`：单个逻辑块处理多少元素；
- `BLOCK_SIZE_SUB`：每轮 load / compute / store 的子块；
- `NUM_CORE`：使用多少物理 program；
- `multibuffer`：是否尝试 MTE 和 Vector 的 ping-pong。

```python
@triton.autotune(
    configs=[
        triton.Config(
            {
                "NUM_CORE": 56,
                "BLOCK_SIZE": 2048,
                "BLOCK_SIZE_SUB": 512,
                "multibuffer": True,
            }
        ),
        triton.Config(
            {
                "NUM_CORE": 56,
                "BLOCK_SIZE": 4096,
                "BLOCK_SIZE_SUB": 1024,
                "multibuffer": True,
            }
        ),
        triton.Config(
            {
                "NUM_CORE": 64,
                "BLOCK_SIZE": 4096,
                "BLOCK_SIZE_SUB": 1024,
                "multibuffer": True,
            }
        ),
        triton.Config(
            {
                "NUM_CORE": 64,
                "BLOCK_SIZE": 8192,
                "BLOCK_SIZE_SUB": 1024,
                "multibuffer": False,
            }
        ),
    ],
    key=["n_elements"],
)
@triton.jit
def silu_mul_autotune_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    NUM_CORE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_SUB: tl.constexpr,
):
    pid = tl.program_id(0)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)

    block_id = pid
    while block_id < num_blocks:
        block_base = block_id * BLOCK_SIZE
        for sub_offset in tl.static_range(0, BLOCK_SIZE, BLOCK_SIZE_SUB):
            offsets = block_base + sub_offset + tl.arange(0, BLOCK_SIZE_SUB)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
            out = x * tl.sigmoid(x) * y
            tl.store(out_ptr + offsets, out, mask=mask)

        block_id += NUM_CORE
```

launch 时要让 grid 依赖当前 config 的 `NUM_CORE` 和 `BLOCK_SIZE`：

```python
def launch_autotune(x, y, out):
    n_elements = x.numel()

    def grid(meta):
        num_blocks = triton.cdiv(n_elements, meta["BLOCK_SIZE"])
        return (min(num_blocks, meta["NUM_CORE"]),)

    silu_mul_autotune_kernel[grid](x, y, out, n_elements)
```

调优时建议打开：

```shell
TRITON_PRINT_AUTOTUNING=1
TRITON_ALWAYS_COMPILE=1
TRITON_DEBUG=1
TRITON_KERNEL_DUMP=1
TRITON_DUMP_DIR=./cache
```

如果当前 Triton-Ascend 版本不接受把 `multibuffer` 放进 `triton.Config` 的 kwargs，可把它改成 launch 关键字参数，或按本地 backend 支持的写法传入。核心思想不变：**把是否开启 MultiBuffer 纳入同一轮性能比较**。

## 3. 示例二：Row-wise Softmax

Row-wise softmax 常见 shape 是 `[M, N]`，每行做一次归一化。GPU 上常见写法是 `grid = (M,)`，每个 row 一个 program。Ascend 上如果 `M` 很大，也会遇到逻辑 row 数远大于物理核数的问题。

配套可执行文件：[softmax-row-split.py](./softmax-row-split.py)。

### 3.1 直接每行一个 program 的写法

```python
@triton.jit
def softmax_row_naive_kernel(
    x_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    stride_m: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    x = tl.load(x_ptr + row * stride_m + offs, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    numerator = tl.exp(x)
    denominator = tl.sum(numerator, axis=0)
    out = numerator / denominator
    tl.store(out_ptr + row * stride_m + offs, out, mask=mask)
```

### 3.2 改成物理核 grid + 逻辑 row 跨步

```python
@triton.jit
def softmax_row_split_kernel(
    x_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    stride_m: tl.constexpr,
    NUM_CORE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    row = pid
    while row < M:
        x = tl.load(x_ptr + row * stride_m + offs, mask=mask, other=-float("inf"))
        x = x - tl.max(x, axis=0)
        numerator = tl.exp(x)
        denominator = tl.sum(numerator, axis=0)
        out = numerator / denominator
        tl.store(out_ptr + row * stride_m + offs, out, mask=mask)

        row += NUM_CORE
```

launch：

```python
def next_power_of_2(x):
    return 1 << (x - 1).bit_length()


def launch_softmax(x, out, m, n, num_core):
    block_n = next_power_of_2(n)
    grid = (min(m, num_core),)
    softmax_row_split_kernel[grid](
        x,
        out,
        M=m,
        N=n,
        stride_m=n,
        NUM_CORE=num_core,
        BLOCK_N=block_n,
        multibuffer=True,
    )
```

这个例子里 `BLOCK_N` 还和向量化、尾轴对齐相关：

- fp32 下，连续尾轴一次处理 64 个元素约等于 256B 向量 tile；
- fp16 下，连续尾轴一次处理 128 个元素约等于 256B；
- 如果 `N` 很小或不是对齐长度，softmax 的固定开销和尾轴补齐开销会很明显。

### 3.3 Softmax 的 autotune 维度

Row-wise softmax 的候选项通常不只是 `BLOCK_N`，还包括一组行打包策略。下面示例用 `ROWS_PER_PROGRAM` 让一个 program 一次处理多行，适合 `N` 较小、单行工作量不足时摊薄启动开销。

```python
@triton.autotune(
    configs=[
        triton.Config({"NUM_CORE": 56, "BLOCK_N": 64, "ROWS_PER_PROGRAM": 1}),
        triton.Config({"NUM_CORE": 56, "BLOCK_N": 128, "ROWS_PER_PROGRAM": 1}),
        triton.Config({"NUM_CORE": 56, "BLOCK_N": 128, "ROWS_PER_PROGRAM": 2}),
        triton.Config({"NUM_CORE": 64, "BLOCK_N": 128, "ROWS_PER_PROGRAM": 4}),
    ],
    key=["M", "N"],
)
@triton.jit
def softmax_row_autotune_kernel(
    x_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    stride_m: tl.constexpr,
    NUM_CORE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)

    logical_groups = tl.cdiv(M, ROWS_PER_PROGRAM)
    group_id = pid
    while group_id < logical_groups:
        for r in tl.static_range(0, ROWS_PER_PROGRAM):
            row = group_id * ROWS_PER_PROGRAM + r
            mask = (row < M) & (offs_n < N)
            x = tl.load(
                x_ptr + row * stride_m + offs_n,
                mask=mask,
                other=-float("inf"),
            )
            x = x - tl.max(x, axis=0)
            numerator = tl.exp(x)
            denominator = tl.sum(numerator, axis=0)
            out = numerator / denominator
            tl.store(out_ptr + row * stride_m + offs_n, out, mask=mask)

        group_id += NUM_CORE
```

grid：

```python
def softmax_grid(meta):
    groups = triton.cdiv(M, meta["ROWS_PER_PROGRAM"])
    return (min(groups, meta["NUM_CORE"]),)
```

这个版本的调参逻辑是：

- `N` 大：优先调 `BLOCK_N`，保证尾轴连续、向量化充分，同时避免 UB overflow。
- `N` 小：优先调 `ROWS_PER_PROGRAM`，让单个 program 干更多工作，摊薄固定开销。
- `M` 大：grid 仍然压到物理核数，额外 row 交给逻辑循环。

## 4. 示例三：In-place 会阻断 MultiBuffer

如果输入输出是同一物理地址，编译器很难安全开启 ping-pong。下面用一个归一化写回的例子说明。

配套可执行文件：[scale-alias-multibuffer.py](./scale-alias-multibuffer.py)。

### 4.1 不利于 MultiBuffer 的 in-place 写法

```python
@triton.jit
def inplace_scale_kernel(
    in_out_ptr,
    scale,
    n_elements,
    NUM_CORE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)

    block_id = pid
    while block_id < num_blocks:
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_out_ptr + offsets, mask=mask, other=0.0)
        y = x * scale
        tl.store(in_out_ptr + offsets, y, mask=mask)

        block_id += NUM_CORE
```

这类 kernel 可能节省显存，但 load 和 store 对同一地址有读写依赖。profiling 如果看到 MTE2、Vector、MTE3 串行，IR 中也没有明显多 buffer 交替，就要怀疑 alias 限制了 MultiBuffer。

### 4.2 更利于 MultiBuffer 的独立输出写法

```python
@triton.jit
def out_of_place_scale_kernel(
    x_ptr,
    out_ptr,
    scale,
    n_elements,
    NUM_CORE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)

    block_id = pid
    while block_id < num_blocks:
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = x * scale
        tl.store(out_ptr + offsets, y, mask=mask)

        block_id += NUM_CORE
```

如果业务必须最终写回原地址，可以比较两种方案：

- 单 kernel in-place：显存占用低，但可能丢失存算重叠。
- 两阶段 out-of-place：先写独立输出，再拷回目标地址；显存和一次额外拷贝换取更好的主 kernel 流水。

## 5. Autotune 配置矩阵怎么收敛

建议按下面顺序缩小候选项，而不是一次把所有组合全展开：

1. **先定核数候选**：只放当前硬件可能的 `num_vectorcore` / `num_aicore`，不要用 GPU SM 数。
2. **再定总 tile**：从不溢出 UB 的最大 `BLOCK_SIZE` 往下试。
3. **再定子 tile**：用 `BLOCK_SIZE_SUB` 控制单轮 MTE 粒度和 DoubleBuffer 所需 UB。
4. **最后加编译选项**：`multibuffer`、`unit_flag`、`tile_mix_vector_loop`、`auto_blockify_size` 等按算子类型加入。

一个更接近真实调参的配置生成方式：

```python
def make_ascend_configs(num_cores):
    configs = []
    for num_core in num_cores:
        for block_size in [2048, 4096, 8192]:
            for sub_block in [512, 1024]:
                if block_size % sub_block != 0:
                    continue
                configs.append(
                    triton.Config(
                        {
                            "NUM_CORE": num_core,
                            "BLOCK_SIZE": block_size,
                            "BLOCK_SIZE_SUB": sub_block,
                            "multibuffer": True,
                        }
                    )
                )
    return configs
```

如果是 CV 融合或涉及 workspace，再按当前 backend 支持情况加入：

```python
triton.Config(
    {
        "NUM_CORE": 56,
        "BLOCK_SIZE": 4096,
        "BLOCK_SIZE_SUB": 1024,
        "multibuffer": True,
        "unit_flag": True,
        "tile_mix_vector_loop": 4,
        "auto_blockify_size": 4,
    }
)
```

这些选项不是所有 Triton-Ascend 版本都接受同一种传参位置。写文档或用例时应记录三件事：

- Triton-Ascend 版本；
- 实际 `best_config`；
- 对应的 profiling 变化，例如 `BLOCK_DIM`、`aiv_mte2_time`、`aiv_vec_time`、总耗时。

## 6. Profiling 后怎么判断调参方向

| 现象 | 优先改的参数 | 解释 |
|------|--------------|------|
| `BLOCK_DIM` 远大于物理核数 | `NUM_CORE`、grid 公式 | 减少多轮 program 下发和头开销 |
| 物理核没打满 | `NUM_CORE`、`ROWS_PER_PROGRAM`、`BLOCK_SIZE` | 单次 launch 逻辑任务太少或单核工作太轻 |
| UB overflow | `BLOCK_SIZE`、`BLOCK_SIZE_SUB`、`multibuffer` | DoubleBuffer 会增加 UB 压力，先缩子 tile |
| `aiv_mte2_time` 高 | `BLOCK_SIZE_SUB`、连续访存布局、`multibuffer` | 提高单次搬运粒度和搬运/计算重叠 |
| `aiv_vec_time` 高但 MTE 不高 | `BLOCK_SIZE`、尾轴对齐、向量化结构 | 检查是否用满 256B 向量 tile |
| 小 shape 总耗时不降 | 合并行/元素、`ROWS_PER_PROGRAM`、减少 launch | 这时主要是固定开销，单纯调 tile 收益有限 |

## 7. 最小检查清单

写完或改完一个 Triton-Ascend kernel 后，至少检查：

- launch 处的 `grid` 是否依赖物理核数，而不是无条件等于逻辑块数；
- kernel 内是否有 `while work_id < total_work` + `work_id += NUM_CORE` 这类逻辑核跨步；
- `BLOCK_SIZE` 和 `BLOCK_SIZE_SUB` 是否分离；
- autotune 的 `key` 是否包含会改变最优配置的 shape 参数；
- `TRITON_PRINT_AUTOTUNING=1` 是否能打印 best config；
- dump 出的 IR / profiling 是否能证明 `multibuffer`、向量化、核数配置真的生效。
