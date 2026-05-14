# 昇腾Inductor-Triton-NPUIR 融合算子性能分析实践

## 文章定位与读者

本文面向**昇腾（主要是Atlas 910_95，大部分思路A2/A3也可参考）** 上 **Vector融合** Triton 算子的**性能分析、瓶颈定位与调参**；硬件为 **AI Core 分离架构，Cube : Vector = 1 : 2**（每组 1 个 Cube 配 2 个 Vector，**AI Core 数以 Cube 为准**）。算子来源包括 **PyTorch Inductor 融合生成**与 **GPU 迁移或手写 Triton**。**不涉及 Ascend C 算子从零开发**。其他芯片或纯 Vector（VV）场景可参考思路，细节以各产品文档为准。

**硬件与 profiling 列、数据流**见文末 **附录 A**。

**两条编译路径如何划分、可分析哪些产物、各工具用途**见下文 **「两条编译路径」** 与 **「可分析文件与工具总览」**，此处不重复。

---

## 两条编译路径

| 路径 | 入口 | 后半程（一致） |
|------|------|----------------|
| **路径 1** | PyTorch Module（小算子组成的网络）→ Inductor 图模式编译 → 融合并生成 Triton kernel | Triton kernel → Triton-Ascend → ttadapter → bishengir-compile → npubin |
| **路径 2** | 已有基于 GPU 开发的 Triton kernel | 同上：Triton-Ascend → ttadapter → bishengir-compile → npubin |

路径 1 的 grid/切分等可能由 Inductor 自动调优得到；路径 2 多为手写或从 GPU 迁来的 kernel，**首版 grid / tiling 常直接沿用 GPU 上 autotune 或手调的值**，再到 NPU 上迭代。性能分析时需先明确路径，以便知道参数来自 **Inductor best_config**、**GPU 配置** 还是 **NPU 侧重调**。

---

## 可分析文件与工具总览

| 产物 | 来源/工具 | 用途 |
|------|-----------|------|
| Triton kernel（.py） | 路径 1：output_code / cache；路径 2：手写或仓库单文件 | 理解计算逻辑、grid、block、tile、是否 multibuffer |
| ttir.mlir | TRITON_DEBUG=1 或 cache | Triton 前端输出，平台无关的高层 IR |
| ttadapter.mlir | Triton-Ascend 转换输出 | 适配 NPU 的 memref/linalg/scf IR，bishengir 的输入 |
| 各 pass 后 IR | bishengir-compile + `--mlir-print-ir-after-all` | 看 HFusion/HIVM 优化、InjectSync、AutoVectorize 等效果 |
| npubin | bishengir-compile 最终产物 | 设备可执行二进制 |
| 反汇编 | npubin + dvcmodel 等工具 | 看 SMEM_BAR、指令序列、与 IR 对应 |
| 指令流水图 | msprof op simulator + MindStudio | 看气泡、并行度、访存/计算重叠 |
| profiling 数据 | msprof / torch.profile，op_summary.csv 等 | aic/aiv/vec/scalar/mte2/mte3 等（见 **§二-2**、**附录 A**） |

---

# 一、文件筛选与准备

本章按 **路径 1 / 路径 2** 分开写：**路径 1** 多涉及整网、Inductor、`output_code.py`；**路径 2** 通常只有独立 kernel，**没有**整网与 `output_code.py` 阶段。两路径后半程一致，**保存 compile cache（ttadapter / ttir / npubin 等）的环境变量见本章末尾「共用：运行用例并保存 compile cache」**。

---

## 路径 1（Inductor 融合）：分析对象与文件

**分析对象**：先明确 **哪张网络、哪些高优先级算子**（整网耗时占比高，或与 GPU 对比时融合后性能不及预期）。

### 已拿到单算子用例和 compile cache

优先看 cache 中**包含完整文件**的用例：`triton_xxx_fused_xxx.py` + `xxx.json` + `xxx.ttadapter` + `xxx.ttir` + `xxx.npubin`。若只有部分文件，多为编译失败后 Fallback，并非实际运行的融合算子，可先不分析。

### 仅提供整网 `output_code.py`

整网 codegen 结果体量很大，建议**只抠出目标 kernel** 单独跑，不要整文件当单算子分析入口。

#### 抽取单算子

将对应 kernel name 的 Triton 定义与最小 `main` 抽成独立脚本，例如：
```python
triton_unk_fused_xxx = async_compile.triton('triton_unk_fused_xxx', '''
...
@npu_triton_heuristic...
@triton.jit
def triton_unk_fused_xxx(...)
    ...
''', device_str='npu')

if __name__ == "__main__":
    # 只保留与该算子相关的入参构造
    arg = rand_strided(shape, stride, device, dtype)
    stream = ...
    triton_unk_fused_xxx.run(arg, dims, stream)
```

#### 配置切分（Inductor）

`TORCH_LOGS='+schedule, +inductor'` 跑用例，可看到调优过程中的 Y0BLOCK、X1BLOCK 等；最优结果多在 `xxx/*.best_config`。

#### 配置 grid（Inductor）

由 shape + 切分推算 grid，或在环境中改 `torch_npu/_inductor/npu_triton_heuristics.py` 打印运行时 grid。

#### 改为可固定复现的 Triton 调用

1. 抠出 kernel 字符串，去掉或绕开 `npu_triton_heuristic` 对 grid/参数的隐藏封装（若需完全可控）。
2. 使用 Triton 原生 grid + 显式 constexpr 参数调用，例如：

```python
# 修改前：隐式 grid / 封装调用
triton_unk_fused_xxx.run(
    buf0, arg0_1,
    32, 3, 4800, 85,
    stream=stream0)
```

```python
# 修改后：显式 grid + 切分参数 + multibuffer
triton_unk_fused_xxx[(32, 1)](
    buf0, arg0_1,
    32, 3, 4800, 85,
    1, 3, 128,  # 切分参数
    stream=stream0,
    multibuffer=True)
```

---

## 路径 2（GPU 迁移 / 手写 kernel）：准备要点

- **没有** PyTorch 整网、**没有** `output_code.py`；入口一般是**独立 `.py`** 或仓库里的 kernel + 自写 `launch`。
- **首版 grid、BLOCK、tile 等**：实践中常**直接沿用该 kernel 在 GPU 上 autotune 或手调保存的配置**，作为上 NPU 的第一版基准。
- **注意**：NPU 与 GPU 在 **物理核数、UB 容量、向量宽度、访存行为** 上均不同，GPU 上较优的配置在 NPU 上**往往只是起点**，需要在 Ascend 上 **重新 autotune** 或按本文后续章节（核数、切分、向量化、MultiBuffer 等）迭代；勿假设「GPU 最优 = NPU 最优」。
- **文件准备**：保证单算子最小复现可编译运行；同样需要 dump 出 **ttadapter / ttir / npubin**（环境变量见本章「共用」小节）。若 GPU 侧有保存的 `Config`（block、num_warps 等），一并记录，便于对比 NPU 调参前后差异。

---

## 共用：运行用例并保存 compile cache

路径 1、路径 2 均可使用：

```shell
# A5 等硬件常需
TRITON_DISABLE_FFTS=1
TRITON_ASCEND_ARCH=Ascend910_9589   # 按实际芯片修改

TRITON_ALWAYS_COMPILE=1
TRITON_DEBUG=1
TRITON_KERNEL_DUMP=1
TRITON_DUMP_DIR=./cache
```

---

# 二、文件使用与分析闭环（总体）

本节给出性能分析的总体流程，具体到 bishengir 各 pass 的阅读方式在第三章展开。

1. **理解 Python 用例逻辑**：原始计算语义与 Triton 实现是否一致；**grid、切分来自 Inductor best_config（路径 1）还是 GPU 沿用/手写（路径 2）**，在 NPU 上是否合理。
2. **Profiling**：运行 `msprof --application='python xxx.py'` 或 torch.profile，关注 **op_summary.csv**（或 kernel_details.csv）中的耗时列。**aic / aiv / MTE 、核数比、CV/VV数据流**的对应见 **附录 A**。
   - **aic**：Cube 侧
   - **aiv**：Vector 侧（同组 2 Vector 与 1 Cube 配套）
   - **vec / scalar**：向量执行 / 标量调度与控制
   - **mte2 / mte3**：多与 **GM↔UB** 搬入搬出相关；**aic** 高时还需看 **MTE1、FixPipe** 等 Cube 链路（**附录 A.2**）  
   判断主要瓶颈（如 mte2/mte3 高偏访存，vec 高偏 Vector 计算，aic 高偏 Cube 或 L0/L1 通路）。
3. **ttadapter → 各 pass IR**：以 `*.ttadapter.mlir`（或工具约定的 ttadapter 输入）调用 **bishengir-compile**，加 `--mlir-print-ir-after-all`（若构建支持），结合**第三章** pass 表对照各阶段 IR。

   **默认示例（910_95 / 950、Triton-Ascend、`linalg_to_bin_enable_npu_compile_910_95` 路径）**  

   ```bash
   bishengir-compile /path/to/kernel.ttadapter.mlir \
     --target=Ascend910_9589 \
     --enable-auto-multi-buffer=True \  # 看情况
     --enable-auto-bind-sub-block=True \
     --disable-ffts \
     --enable-hfusion-compile=true \
     --enable-triton-kernel-compile=true \
     --enable-vf-merge-level=1 \
     --enable_mixed_cv=True \  # cv融合类算子需要
     -o /path/to/out/kernel
   ```

   查看各 pass 后 IR 时，在以上参数基础上追加（若当前 bishengir 构建支持）例如 **`--mlir-print-ir-after-all`**，或调试时用 **`--bishengir-print-ir-after=hivm-inject-sync`**（对应 Triton `debug=True`）。
4. **仿真流水图**：用 npubin 或 python 用例 + msprof op simulator 得到仿真数据，在 MindStudio 中查看流水图，除**气泡、load/store 与计算重叠、并行度**外，还可按时间轴区分大致阶段（指令名以实际仿真为准）。**Vector 段与 DMA 段**同 **附录 A.3** 数据流对照。
   - **头开销**：如 **DC_PRELOAD_XN_IMM**、**LDP_XI_XJ_XN** 等与核启动、预加载相关的片段；
   - **首个 VF 启动开销**：含寄存器/参数等**值填充**阶段；
   - **核心 VF 计算**：若核内有多段 tile，可能表现为**多轮循环**形态；
   - **DMA load / store**：多对应 **MTE2（load）/ MTE3（store）** 区间。  
   小数据量下上述固定段可能主导总耗时，分析要点见 **§11**

   **TODO**：插入 1 张典型流水图截图（可打码），并在图中标注「头开销 / 首 VF / 核心计算 / DMA」区间；或附官方文档截图位置与链接。

5. **反汇编**：npubin + **dvcmodel** 等工具得到反汇编，关注 **SMEM_BAR** 是否过多、是否与 IR 中 InjectSync/InjectBlockSync 对应合理；反汇编需与流水图、IR 结合看。**TODO**：补充 dvcmodel 获取方式、对 npubin 的**示例命令行**及与 IR 对照的阅读顺序。
6. **与 AscendC 手写对比**：从算法层面看是否有优化空间，例如 gelu 等小融合用数学近似、reduce 用分治相加等。

---

# 三、bishengir-compile 关键 Pass 与 IR 阅读

ttadapter 进入 bishengir-compile 后，先经 **BiShengTTIRPipeline**（Triton 方言 lowering），再经 **HFusion Pipeline**（含 Triton 时走 register-based 分支），最后 **ConvertToHIVM** → **HIVM 优化** → 代码生成。性能相关的主要阶段如下（顺序为执行顺序）：

| 阶段 | 关键 Pass/行为 | 性能关注点 |
|------|----------------|------------|
| HFusion（Triton 路径） | FlattenOps、Normalize、**AutoVectorize / AutoVectorizeV2**、PreVectorizationFusion、VFFusion | 向量化是否充分、VF 融合与提取是否合理、tree reduce 等 |
| HFusion → HIVM | HFusionToHIVM、TritonGlobalKernelArgsToHIVMOp | 核参数、内存布局 |
| HIVM 优化 | **MarkMultiBuffer**（DoubleBuffer 等）、PlanMemory、**HIVM LowerToLoops**、**InjectSync（Intra-Core Sync）**、EnableMultiBuffer、InjectBlockSync（跨核） | 同步点是否过多、MultiBuffer 是否生效、UB 规划 |
| 后续 | Bufferize、LIR、二进制生成 | 指令选择、寄存器分配 |

查看各阶段 IR：以 ttadapter 为输入运行 bishengir-compile，加上 **`--mlir-print-ir-after-all`**（若工具支持），或使用 **`MLIR_ENABLE_DUMP=1`** 看各 pass 前后 IR。重点对照：

- **InjectSync**：自动插入的同步受 MultiBuffer（多为 DoubleBuffer）影响；MultiBuffer 能隐藏访存延迟，减少为保依赖而插入的同步。
- **AutoVectorize / AutoVectorizeV2**：向量化是否用满硬件向量宽度；VF 是否被合理融合或拆出亦需关注。**默认向量 tile 上界为 256 字节及与 dtype 对应的元素个数**，详见 **§4**。

---

# 四、常见优化点

**路径 1 / 路径 2** 的划分与参数来源见文首 **「两条编译路径」**；下文各节不再每次重复。

以下列出常见瓶颈与对应分析/修改思路。更多 Triton 在 NPU 上的开发与迁移建议可参考：[Triton 算子开发指南](https://triton-ascend.readthedocs.io/zh-cn/latest/programming_guide.html)、[昇腾与 GPU 的开发差异](https://triton-ascend.readthedocs.io/zh-cn/latest/migration_guide/architecture_difference.html)。

## 1. 核数设置不合理（过大或过小）

昇腾 NPU 的 **物理核数**为几十量级（A5 **CV分离 核数1:2** 不同型号下 **num_vectorcore为56或64，具体详查AscendV网站**），与 GPU 的 SM 量级不同。grid 超过所应对齐的物理核数时，多出的任务会通过多轮下发完成，带来核启动与头开销重；过少则无法打满硬件。

- **分析文件**：Triton 调用处 grid、profiling 的 BLOCK_DIM；路径 1 可看 inductor `best_config` 与调优日志。
- **修改方案**：分核数对齐 **硬件物理核数**，核内用 `range(pid, NUM_BLOCKS, NUM_CORE)` 等跨步覆盖全量 block；`num_aicore` / `num_vectorcore` 由 `torch_npu` + `driver.active.utils.get_device_properties(device)` 读取。路径 1 可重跑 inductor 调优或改 heuristics。

## 2. 切分设置不合理

没有核内切分，或切分（tile size）过小，导致 **UB 用不满、单核利用率低**；过大则可能 **UB overflow**（如 `ub overflow, requires xxx bits while xxx bits available`）。**A5** 片上 UB 容量以当前驱动/报错中的 bits 与规格为准；**double buffer** 时有效 UB 约折半，需在调 tile 时预留。

- **分析文件**：Triton kernel 的 `BLOCK`/tile 等 constexpr 参数、ttir/ttadapter 中每块处理的数据量；profiling 中单核负载；若 UB 报错，按**当前芯片**报错中的 bits 与 UB 上限估算缩小倍数。
- **修改方案**：在 kernel 内调整 block/tile 使单块所需 UB 不超过设备上限且尽量用满；**核内用 for 循环做子块分块**（如 `BLOCK_SIZE_SUB`，每次处理一小块），避免单次 load 过大；路径 1 可结合 inductor 的 Y0BLOCK/X1BLOCK 等调优结果重算。可用 `triton.autotune` 对 BLOCK_SIZE、BLOCK_SIZE_SUB 等做自动调优（`TRITON_PRINT_AUTOTUNING=1` 可打印最优参数）。

## 3. DoubleBuffer / 存算并行未使能或无效果

DoubleBuffer（MultiBuffer）实现 **MTE2↔Vector（及 MTE3 调度）** 上的 **ping-pong**，与 **附录 A.3** 数据流一致。若未生效，流水图中 load 与 VF 计算之间气泡明显。编译器默认 `multibuffer=True`，调用侧或编译选项可能关闭。

**输入输出地址复用（in-place / alias）**：若 Triton DSL 入参中存在 **`in_out_ptr` 等同一物理地址既读又写** 的情况，会形成**读写依赖**——后一轮迭代或另一 buffer 上的计算必须等前一轮对该地址的写可见，**即使配置了 double buffer，编译器通常也无法安全使能 ping-pong**。复用能省全局内存占用，但常牺牲存算重叠。若 profiling 显示搬运与计算串行、MultiBuffer IR 未出现，需排查是否 in-place。

- **分析文件**：Triton 调用处是否传入 `multibuffer=True`（或等价方式）；kernel 形参是否 **输入指针与输出指针相同或指向重叠区域**；ttadapter/HIVM 阶段 IR 中是否出现多 buffer 分配与交替使用；bishengir-compile 的 **MarkMultiBuffer**、**EnableMultiBuffer** 相关选项；反汇编与流水图中 load/store 与计算是否交错。
- **修改方案**：在 kernel 调用处显式开启 `multibuffer=True`（参见上文「修改用例加上完整配置」）；若为路径 1，确认 inductor 生成的调用是否带 multibuffer；若编译器默认关闭自动 MultiBuffer，可查 bishengir-compile 的 `--enable-auto-multi-buffer` 等选项并开启。Autotune 时可在 `triton.Config` 中传入 `'multibuffer': True` 等编译选项。**若需 MultiBuffer 且可接受额外显存**：将输出写到独立 buffer，算子结束后再拷贝回 in-place 目标（或拆成两阶段 kernel），以打破读写 alias。

## 4. 向量化与尾轴对齐（向量化不充分、自动补齐）

**向量化不充分**：未用满单次向量访存/计算的宽度。在 A5 上编译器默认按 **256 字节** 作为一条向量化 tile 的上界（AutoVectorize / AutoVectorizeV2）；**fp32 场景下「满向量」即尾轴方向一次处理 64 个元素**，fp16 为 128、int8/fp8 为 256。若 IR/反汇编中可见远小于上述宽度的循环或窄向量，则向量单元利用率低。

**尾轴未对齐导致自动补齐**：VV 类算子要求 Tensor 尾轴能被 **32 Bytes** 整除，CV 类要求 **512 Bytes** 整除；不足时编译器自动补齐，产生多余计算与访存，性能明显恶化（如 shape (2048,3)、(2048,1)）。尾轴不对齐会限制有效向量化宽度，与“向量化不充分”常同时出现，可一并排查。

- **分析文件**：bishengir-compile 中 **AutoVectorize / AutoVectorizeV2** 之后的 IR（HFusion 或 Vector 方言），看 linalg/vector 的 vector 维度；反汇编中向量指令的宽度；profiling 中 vec 与 scalar 占比；Triton kernel 与 ttir/ttadapter 中参与计算的 **shape、stride、尾轴长度**；是否存在短轴在尾维导致补齐、实际计算量大于理论量。
- **修改方案**：
  - **向量化**：Triton 侧在 **fp32** 等场景尽量使连续尾轴 block 为 2 的幂且 **≥ 64**（对应满 256B 向量）；fp16 可对标 **≥ 128**，int8 **≥ 256**（仍为一向量宽）。保证连续访存、stride 对齐，数据块对齐到 **32 字节**边界；若 IR 中向量化被条件或形状限制，可调整循环结构或边界处理；特殊需求下可通过 bishengir 的 **`vector-length`**（字节）显式调整向量化上界（默认 256）。
  - **尾轴对齐**：**将对齐轴转到低维**（如转置），使主轴满足对齐要求，store 时再转回；或借轴转置，详见 [Triton 算子开发指南 - 尾轴对齐](https://triton-ascend.readthedocs.io/zh-cn/latest/programming_guide.html)。

## 5. 标量计算过多

算子中本可向量化的计算以标量或窄向量形式执行，导致 **scalar** 或 **vec** 占比异常（如 scalar 耗时偏高），向量核利用率低。

- **分析文件**：profiling 中 **scalar**、**vec**、**aiv** 的耗时与占比；AutoVectorize 后 IR 中是否仍存在大量逐元素标量 op；反汇编中标量指令与向量指令比例。
- **修改方案**：在 Triton 中尽量用 `tl` 的向量化 API（如 `tl.arange`、逐元素运算在 tensor 上做），避免在循环内对单元素做标量运算；检查是否有条件分支或动态索引导致编译器无法向量化，必要时用 mask 替代分支；bishengir 侧可尝试 **enable-auto-vectorize-v2**、**enable-vf-fusion**，并查看 IR 中 VF 融合与提取是否合理。

## 6. 离散索引访存与 SIMT 模式

对基于索引的 gather/scatter 或离散地址的 load/store，若直接按索引从全局内存拉取，会导致大量小批量 L2→UB 搬运，aiv_mte2 耗时与占比偏高。NPU 上更优做法通常是**先将连续或整块数据搬运到 UB，再在 UB 上用 select/gather 取目标值**；部分场景下需结合 **SIMT 模式** 或编译器能力做进一步优化。

- **分析文件**：Triton kernel 中是否存在 `tl.load(x_ptr + idx * stride)` 等离散索引；profiling 中 **aiv_mte2_time / aiv_mte2_ratio** 是否明显偏高；ttadapter/IR 中 load 模式是否为连续块 vs 离散。
- **修改方案**：可先参考 [Triton 算子开发指南 - 先将数据搬运到UB再select](https://triton-ascend.readthedocs.io/zh-cn/latest/programming_guide.html)：例如将 `tl.load(x_ptr + idx * stride_x)` 改为先 `tl.load(x_ptr + rm * stride_x)` 得到 `x_shared`，再用 `tl.gather(x_shared, idx, 0)` 从 UB 中取数，减少 L2→UB 次数。

  **TODO**：补充 **SIMT 模式**下离散索引访存的适用条件、编译选项或与同事实践一致的调参步骤。

## 7. 编译 / Autotune 选项未调优

除 multibuffer 外，AscendNPU IR 还提供多种编译优化选项，在 autotune 的 `triton.Config` 中传入可影响代码生成与流水。

- **分析文件**：当前用例是否使用 autotune、Config 中是否传入 ascend 相关选项；bishengir-compile 与 triton-ascend backend 的默认值（如 `ascend/backend/compiler.py`）。
- **修改方案**：按下表与场景调参（详见 [昇腾与 GPU 的开发差异 - AscendNPU IR 优化](https://triton-ascend.readthedocs.io/zh-cn/latest/migration_guide/architecture_difference.html)）：

| 选项 | 能力 | 说明 |
|------|------|------|
| `multibuffer` | 存算并行 | 默认 true，autotune 可配置 |
| `unit_flag` | cube 搬出优化 | autotune 可配置 |
| `limit_auto_multi_buffer_only_for_local_buffer` | CV/局部 buffer 优化 | autotune 可配置 |
| `limit_auto_multi_buffer_of_local_buffer` | double buffer scope | 如 `["no-limit","no-l0c"]` |
| `set_workspace_multibuffer` | workspace 多 buffer | 如 `[2,4]`，需与 limit_* 配合 |
| `tile_mix_vector_loop` / `tile_mix_cube_loop` | CV 算子 vector/cube 切份数 | 如 `[2,4,8]` |
| `auto_blockify_size` | TRITON_ALL_BLOCKS_PARALLEL 时首维扩展大小 | 如 `[2,4,8]` |

## 8. 冗余 permute / transpose 与随路转置（NDDMA）

Triton DSL 中若存在 **permute、transpose 等布局变换**，且与 **load / store 紧邻**，往往可以**不写显式转置**：通过改写 **load 或 store 的索引与 stride**，在语义上等价于“随路转置”。下沉到编译器后，有机会走 **NDDMA（N-Dim Direct Memory Access）** 等路径做搬移时的维序变换，通常比先在 UB 上算子级 transpose 再访存更省。

- **分析文件**：kernel 中 `tl.trans` / reshape-permute 等与 `tl.load`、`tl.store` 的相邻关系；ttadapter / HFusion IR 中是否仍存在独立大段 transpose；流水图中转置与搬运是否可合并。
- **修改方案**：能合并到边界 load/store 的，尽量用**索引表达布局**（等价于随路转置），减少中间 tensor；具体索引设计需与连续访存、尾轴对齐（§4）一并考虑。

## 9. 搬移效率与 MTE 利用率

**MTE2** 负责向量侧数据搬入（GM→UB 等）。A5 上 **MTE2 标称带宽约 1.6 TB/s**；与 profiling 常用 **GB/s** 对比时，可按 **1.6×1024 ≈ 1638.4 GB/s** 作为理论上限（1024 进制）。

**TODO**：补全 **MTE3** 标称带宽（或写明以何版本 datasheet 为准），并与下文利用率公式对称说明是否用同一进制换算。

**利用率估算示例**（单次算子、单输入、冷启动首次调用，避免 cache 干扰）：

- 输入 shape `[6528, 4, 8, 128]`，`dtype=int64`，总搬运量  
  `6528 × 4 × 8 × 128 × 8 = 213909504` Bytes。
- 取 profiling 中该次调用的 **`aiv_mte2_time`**（如 **180 µs**），有效带宽（约）  
  `213909504 / (180 × 10⁻⁶) ≈ 1188.39 GB/s`。
- **利用率** `≈ 1188.39 / 1638.4 ≈ 72.5%`。经验上 **>70%** 可认为搬移已接近带宽上限；明显偏低时需结合 **stride、对齐、是否离散 load、是否可 NDDMA/连续块** 等做专项分析。

- **分析文件**：`aiv_mte2_time`、与真实搬运字节数（注意多输入/多轮循环需累加或按块折算）；shape、dtype、是否 in-place 写回影响有效流量。
- **修改方案**：提高连续性与对齐；减少小步长 gather；参考 §6、§8；必要时调切分使每轮 MTE 为大块连续搬运。

## 10. 循环无关指令外提

与**当前最内层循环迭代无关**的计算（如仅依赖 block id、常量、与循环变量无关的指针偏移）应尽可能放在**循环外**执行，避免每轮重复开销。Triton DSL 层若未外提，部分冗余仍可能在 **NPU IR（bishengir）Pipeline** 中由编译器外提，**不能假设一定下沉优化**。

- **分析文件**：Triton 源码中循环体内的重复计算；IR 中对应区域是否在循环外只保留一份；反汇编/流水是否显示循环内多余标量或地址计算。
- **修改方案**：DSL 侧能外提的尽量手动外提；其余对照 **print-ir-after-all** 确认是否已优化，未优化则改写法或协助编译器可见性（如 constexpr、避免动态形状分支）。

## 11. 极小数据量下固定开销占比过高

在**单次处理数据量极小**（或 per-launch 有效计算量相对启动成本过低）时，**核头开销**、**第一个 VF 的启动与值填充**、**DMA 单次启动**等固定成本在总耗时中占比可能 **超过约 90%**，而**核心 VF 纯计算**占比 **低于约 10%**。此时瓶颈不在算力或 MTE 带宽吃满，而在**启动与摊销**；Profiling 上也可能表现为单次 kernel 绝对耗时很短但总调用次数多、或整段业务被大量小 launch 拖慢。

- **分析文件**：仿真流水图中 **§二-4** 各段时长比例；小 batch / 小 tensor 下的 msprof 单次耗时与调用次数；是否与框架侧「每个小 tensor 单独 launch」有关。
- **修改方案**：方向性思路包括增大单次 launch 的合并处理量（batch、融合多算子）、减少 launch 次数、评估更高层算子或 Host 侧合并后再下 NPU 等。

  **TODO**：按 Inductor / 业务场景补可操作清单（例如：如何减少融合子图拆分 launch、与 PyTorch 侧 `torch.compile` 配置相关的注意点、内部案例链接）。

---

# 五、从瓶颈到改动的快速对应

**改动分两类**：

| 类 | 含义 |
|----|------|
| **1. DSL 修改** | **手写 Triton**，或 **Inductor / triton-ascend 的生成与调用策略**（grid、BLOCK/tile、multibuffer 开关、in-place、autotune **Config 选参**、融合子图与 launch 次数等）。 |
| **2. NPUIR 编译器内部修改** | **bishengir Pipeline内**与硬件强相关的 pass、**细粒度指令展开**、同步（InjectSync 等）、向量化/搬移/**NDDMA**、UB 规划、MultiBuffer **实现算法**等；**改编译器代码或新增编译器能力**。 |

表中 **改动面**：**1** 表示通常先动 DSL/生成策略；**2** 表示主要动 NPUIR；**1；2** 表示两侧都可能要动（常先 1 再视情况 2）。

| 现象 / 瓶颈 | 优先看的文件/数据 | 常见改动方向 | 改动面 |
|-------------|-------------------|--------------|--------|
| 核数超限或过少 | grid、BLOCK_DIM、best_config、物理核数 | 与 **附录 A.1** 对齐物理核数 + 核内跨步，或重跑 inductor 调优 | **1**（**2** 仅当块映射/调度属编译器策略且需改Pipeline时，少见） |
| UB 用不满或 overflow | BLOCK/tile、ttir 每块数据量、UB 报错信息 | 调 tile/核内 for 分块、autotune BLOCK_SIZE_SUB | **1**；**2**（UB 规划、buffer 切分算法） |
| 访存与计算间气泡大 | 流水图、multibuffer、MarkMultiBuffer、**in_out 同址** | 开 multibuffer 并确认 IR；同址读写则独立输出 buffer 或拆阶段打破 alias | **1**；**2**（MarkMultiBuffer/EnableMultiBuffer、依赖与同步） |
| MTE2 带宽利用率低 | `aiv_mte2_time`、搬运字节数、理论上限计算 | 连续大块、对齐、NDDMA/随路转置、减少离散 load | **1**；**2**（MTE 指令生成与搬移调度） |
| 冗余 transpose + load/store | kernel 中 permute 与边界访存 | 改 load/store 索引做随路转置，利用 NDDMA | **1**；**2**（NDDMA 路径与 lowering） |
| 循环内重复无关计算 | Triton 循环体、IR 是否外提 | DSL 外提；查 NPUIR 是否已外提，否则改写法 | **1**；**2**（编译器外提、指令级冗余） |
| vec 占比低、向量化/尾轴问题 | AutoVectorize 后 IR、shape/尾轴、反汇编 | 尾轴对齐并尽量用满 256B、转置/借轴、auto-vectorize-v2 / vf-fusion | **1**；**2**（AutoVectorize/VFFusion、vector-length 等） |
| 标量计算过多 | scalar/vec 占比、IR 中标量 op | 用 tl 向量化 API、避免循环内标量、VF 融合 | **1**；**2**（Bisheng、VF 融合/提取、向量化 pass） |
| 离散索引访存、mte2 偏高 | kernel 中 load 模式、aiv_mte2 占比 | 先搬 UB 再 gather/select；SIMT（**TODO**：见 §6） | **1**（为主）；**2**（SIMT/离散搬移若走编译器策略） |
| 同步过多、SMEM_BAR 密集 | InjectSync 前后 IR、反汇编 SMEM_BAR | 确保 MultiBuffer 生效、减少不必要的依赖 | **2**（为主）；**1**（multibuffer/alias 减轻依赖） |
| 编译/代码生成未调优 | Config、compiler 默认值 | autotune 中配置 unit_flag、tile_mix_*、auto_blockify_size 等 | **1**（**Config 选参**）；**2**（各选项在Pipeline内的实现、缺参时扩 pass） |
| 小数据量、头/VF 启动/DMA 主导（流水图） | §二-4 各段占比、launch 次数与单次数据量 | 合并 batch/算子、减 launch（**TODO**：见 §11 可操作清单） | **1**（Inductor 融合、batch、子图拆分策略） |

---

## TODO 一览（便于全文检索）

以下与正文中 **`TODO`** 标记对应，可按关键词搜索 `TODO` 逐项补全。

| 位置 | 待补内容 |
|------|----------|
| §二-3 | ~~bishengir-compile 完整示例命令~~ 已补充 |
| §二-4 | 流水图示例截图或文档链接 |
| §二-5 | dvcmodel 用法与示例命令 |
| §6 | SIMT 模式判断与调优 |
| §9 | MTE3 标称带宽与换算说明 |
| §11 | 小数据量场景可操作优化清单 |

---

# 附录 A：硬件与 Profiling 对照

本文默认 **Atlas A5、AI Core 分离、每组 1 Cube : 2 Vector**；更一般的定义见昇腾文档：[基本架构](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/opdevg/Ascendcopdevg/atlas_ascendc_10_0008.html)。

## A.1 核数与 profiling：aic / aiv

- **拓扑**：**1 个 Cube Core** 与 **2 个 Vector Core** 为一组；**AI Core 个数 = Cube 个数**。通常 **`num_vectorcore = 2 × num_aicore`**。
- **`aic`**：Cube 侧耗时与活动；**`aiv`**：Vector 侧。CV 融合核内 Cube 与 Vector 协同，两列需结合看。
- **grid**：**宜与 `num_aicore`（Cube）对齐**；业务上多 block 用核内循环或跨 launch 摊分，与正文 **§1** 一致。

## A.2 MTE / FixPipe 与 profiling（CV）

| 单元 | 典型通路 | 与 profiling 的常见对应 |
|------|----------|-------------------------|
| **MTE2** | GM→L1/L0、**GM→UB** | **mte2 / aiv_mte2**：load、GM→片上 |
| **MTE3** | **UB→GM** | **mte3 / aiv_mte3**：store |
| **MTE1** | L1→L0A/L0B | Cube 输入上板；**aic**、Cube 路径相关 |
| **FixPipe** | L0C→GM/L1 等 | Cube 结果写出；**aic** 偏高时需关注 |

Cube 全链路还涉及 **L1/L0** 等，排障时除 **mte2/mte3** 外，**aic 高**可对照 **MTE1、FixPipe** 与流水图 Cube 段。

## A.3 Vector 数据流、流水图与 DoubleBuffer

CV 核中 **Vector 段**典型数据流：

```text
GM → (MTE2) → UB → Vector → UB → (MTE3) → GM
```

- **流水图（正文 §二-4）**：DMA load ↔ **MTE2**；核心 VF ↔ **UB 上 Vector**；DMA store ↔ **MTE3**。
- **DoubleBuffer（正文 §3）**：在 **MTE2、Vector、MTE3** 的调度上叠 **ping-pong**，压 load/计算气泡；与 IR 中 **MarkMultiBuffer**、流水图重叠情况对照阅读。

Cube 侧数据流见架构文档 **Cube 典型数据流**（GM→L1→L0→Cube→L0C→FixPipe→…），与 **aic / MTE1 / FixPipe** 对应分析。