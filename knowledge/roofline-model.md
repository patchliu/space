---
summary: "用 roofline model 判断算子是算力受限还是带宽受限。"
tags: [roofline, performance, ai-chip, llm]
source: ["Roofline model", "Transformer decoder FLOPs formulas"]
---

# Roofline Model

Roofline model 只回答一个很朴素的问题：

一个算子跑不快，到底是因为算力不够，还是因为数据搬得太慢？

## 核心公式

### 1. 算术强度

```text
算术强度 = 计算量 / 访存量
Arithmetic Intensity = FLOPs / Bytes
```

意思是：每搬 1 byte 数据，能做多少次计算。

- 算术强度高：更可能被算力限制
- 算术强度低：更可能被带宽限制

### 2. Roofline 上限

```text
理论性能上限 = min(芯片峰值算力, 算术强度 * 芯片内存带宽)
```

如果一个芯片是：

```text
峰值算力 = 100 TFLOP/s
内存带宽 = 2 TB/s
```

那么分界点是：

```text
100 TFLOP/s / 2 TB/s = 50 FLOP/Byte
```

- 算术强度小于 50 FLOP/Byte：大概率是带宽瓶颈
- 算术强度大于 50 FLOP/Byte：大概率是算力瓶颈

### 3. 理论最短时间

```text
计算时间下限 = 计算量 / 峰值算力
访存时间下限 = 访存量 / 内存带宽
理论最短时间 = max(计算时间下限, 访存时间下限)
```

哪个时间更长，哪个就是主要瓶颈。

### 4. 实际利用率

```text
实际算力 = 实际 FLOPs / 实际运行时间
相对芯片峰值利用率 = 实际算力 / 芯片峰值算力
相对 roofline 利用率 = 实际算力 / roofline 理论性能上限
```

分析优化空间时，更应该看相对 roofline 利用率，而不是只看芯片峰值利用率。

## 常见算子公式

### 矩阵乘

矩阵形状：

```text
A: M x K
B: K x N
C: M x N
```

计算量：

```text
FLOPs = 2 * M * N * K
```

最低访存量：

```text
Bytes = (M*K + K*N + M*N) * 每个元素字节数
```

这里的 `2` 来自一次乘法和一次加法。

### 向量逐元素计算

例如 add、mul、relu。

如果有 `N` 个元素：

```text
FLOPs 约等于 N
Bytes 约等于 读输入 + 写输出
```

这类算子通常算术强度很低，容易被访存限制。

### Reduction

例如 sum、max。

如果有 `N` 个元素：

```text
FLOPs 约等于 N
Bytes 约等于 读 N 个元素 + 写少量结果
```

它通常也偏带宽瓶颈。

### Softmax

Softmax 包含 max、减法、exp、sum、除法。

粗略估算：

```text
FLOPs 约等于 5N 到 10N
Bytes 约等于 多次读写 N 个元素
```

实际瓶颈经常在访存和特殊函数吞吐上。

## 例子：7B 级 decoder-only 模型

假设模型参数：

```text
层数 L = 32
hidden size d = 4096
FFN hidden size f = 11008
vocab size V = 32000
上下文长度 S = 2048
batch size B = 1
数据类型 = FP16/BF16，每个元素 2 bytes
```

为了简单，先忽略 LayerNorm、RoPE、bias、残差、softmax 的小计算量，只算主要矩阵乘。

这些参数进入公式的方式是：

```text
L: 有多少层，同一层的计算和权重要乘 L
d: hidden size，决定 attention 和大部分权重矩阵宽度
f: FFN hidden size，决定 FFN 中间层大小
V: vocab size，决定最后 logits 的矩阵大小
S: 序列长度，prefill 时参与整段计算，decode 时决定 KV cache 长度
B: batch size，这里取 1；如果 B 变大，很多计算量和激活访存要乘 B
```

### 先看一层 decoder 做什么

一层 decoder 可以粗略看成三块：

```text
1. Attention 前的 QKV projection
2. Attention 本身
3. FFN
```

输入是一段 token 的 hidden states：

```text
X: S x d
```

这里：

- `S` 是 token 数，也就是序列长度
- `d` 是每个 token 的隐藏向量长度
- 在这个例子里，`S = 2048`，`d = 4096`

#### QKV projection

模型先把 `X` 分别乘三个权重矩阵，得到 Q、K、V：

```text
Q = X * Wq
K = X * Wk
V = X * Wv
```

形状是：

```text
X:  S x d
Wq: d x d
Wk: d x d
Wv: d x d

Q/K/V: S x d
```

一次 `S x d` 乘 `d x d` 的矩阵乘，计算量是：

```text
2 * S * d * d
```

为什么有 `2`？

因为矩阵乘里一次乘加通常按 2 FLOPs 算：

```text
一次乘法 + 一次加法 = 2 FLOPs
```

Q、K、V 一共有 3 个矩阵乘，所以：

```text
QKV FLOPs = 3 * 2 * S * d * d
```

这就是 `3 * 2 * S * d * d` 的来源。

#### Attention 本身

Attention 做两次主要矩阵乘：

```text
scores = Q * K^T
out    = scores * V
```

形状是：

```text
Q:   S x d
K^T: d x S
V:   S x d

scores: S x S
out:    S x d
```

第一步 `Q * K^T`：

```text
2 * S * S * d
```

第二步 `scores * V`：

```text
2 * S * S * d
```

合起来：

```text
Attention FLOPs = 4 * S * S * d
```

这里先不展开 head 数。只要总 hidden size 还是 `d`，普通 multi-head attention 的总量仍然可以这样粗算。GQA/MQA 会改变 K/V cache 和部分权重规模，单独分析时再修正。

#### Attention output projection

Attention 输出以后，还会再乘一个输出权重：

```text
Y = out * Wo
```

形状是：

```text
out: S x d
Wo:  d x d
Y:   S x d
```

计算量：

```text
Output projection FLOPs = 2 * S * d * d
```

#### FFN

很多 LLM 使用 SwiGLU 结构。粗略看，它有三个大矩阵：

```text
gate = X * Wgate
up   = X * Wup
down = (silu(gate) * up) * Wdown
```

形状是：

```text
X:     S x d
Wgate: d x f
Wup:   d x f
Wdown: f x d
```

其中 `f` 是 FFN hidden size。

三个矩阵乘的计算量分别是：

```text
X * Wgate: 2 * S * d * f
X * Wup:   2 * S * d * f
down:      2 * S * f * d
```

因为 `S*d*f` 和 `S*f*d` 是一样的，所以总量写成：

```text
FFN FLOPs = 3 * 2 * S * d * f
```

这就是 `3 * 2 * S * d * f` 的来源。

#### Logits

最后一层输出要映射到词表：

```text
logits = X * Wvocab
```

形状是：

```text
X:      S x d
Wvocab: d x V
logits: S x V
```

计算量：

```text
Logits FLOPs = 2 * S * d * V
```

如果只取最后一个 token 的 logits，就是：

```text
2 * d * V
```

## Prefill 阶段

Prefill 是一次性处理整段 prompt。

### 每层计算量

QKV projection：

```text
Q/K/V = X * Wq/Wk/Wv
X: S x d
W: d x d

一个 projection = 2 * S * d * d
三个 projection = 3 * 2 * S * d * d

3 * 2 * S * d * d
= 3 * 2 * 2048 * 4096 * 4096
= 206.2 GFLOPs
```

Attention output projection：

```text
out: S x d
Wo:  d x d

2 * S * d * d
= 68.7 GFLOPs
```

FFN，按 SwiGLU 三个矩阵算：

```text
Wgate: d x f
Wup:   d x f
Wdown: f x d

三个矩阵乘 = 3 * 2 * S * d * f

3 * 2 * S * d * f
= 3 * 2 * 2048 * 4096 * 11008
= 554.1 GFLOPs
```

Attention 的 QK 和 AV：

```text
Q * K^T:      2 * S * S * d
softmax(QK)*V: 2 * S * S * d

QK + AV = 4 * S * S * d
= 4 * 2048 * 2048 * 4096
= 68.7 GFLOPs
```

所以每层总计算量约为：

```text
206.2 + 68.7 + 554.1 + 68.7 = 897.7 GFLOPs
```

32 层：

```text
897.7 * 32 = 28.7 TFLOPs
```

最后 logits：

```text
2 * S * d * V
= 2 * 2048 * 4096 * 32000
= 0.54 TFLOPs
```

Prefill 总计算量：

```text
28.7 + 0.54 = 29.3 TFLOPs
```

平均到每个 token：

```text
29.3 TFLOPs / 2048 = 14.3 GFLOPs/token
```

### 最低访存量

每层主要权重数量：

```text
QKV: 3 * d * d
O:   d * d
FFN: 3 * d * f
```

代入数字：

```text
每层参数量 = 3*4096*4096 + 4096*4096 + 3*4096*11008
          = 202.4M parameters
```

32 层：

```text
202.4M * 32 = 6.48B parameters
```

再加 embedding / logits 权重：

```text
V * d = 32000 * 4096 = 131.1M parameters
```

总参数量约：

```text
6.48B + 0.13B = 6.61B parameters
```

FP16 权重最低读取量：

```text
6.61B * 2 bytes = 13.2 GB
```

KV cache 写入量：

```text
每层 K/V = 2 * S * d * 2 bytes
          = 2 * 2048 * 4096 * 2
          = 33.6 MB

32 层 = 1.07 GB
```

如果 logits 全部写出：

```text
S * V * 2 bytes
= 2048 * 32000 * 2
= 0.13 GB
```

所以 prefill 的理论最低访存量约：

```text
13.2 + 1.07 + 0.13 = 14.4 GB
```

真实实现还会有 activation、临时 buffer、多次读写，所以实际访存量会更高。

### Prefill 算术强度

```text
算术强度 = 29.3 TFLOPs / 14.4 GB
        约等于 2030 FLOP/Byte
```

这是很高的算术强度。

如果芯片是：

```text
峰值算力 = 100 TFLOP/s
内存带宽 = 2 TB/s
```

分界点是：

```text
50 FLOP/Byte
```

Prefill 的 2030 FLOP/Byte 远大于 50 FLOP/Byte，所以主要是算力瓶颈。

理论时间下限：

```text
计算时间 = 29.3 TFLOPs / 100 TFLOP/s = 0.293 s
访存时间 = 14.4 GB / 2000 GB/s = 0.007 s
```

所以 prefill 更应该优化矩阵乘效率、tile、并行度、算力利用率。

## Decode 阶段

Decode 是每次只生成 1 个新 token，但要读已有 KV cache。

这里假设当前上下文长度仍然是 `S = 2048`。

### 每个新 token 的计算量

Decode 和 prefill 用的是同一套算法。区别是：

- prefill 的输入是 `S x d`
- decode 的输入只有当前新 token，是 `1 x d`
- decode 的 attention 还要读历史 `K/V cache`

所以很多公式可以把 prefill 里的 `S` 换成 `1`。

每层 QKV：

```text
当前 token: 1 x d
Wq/Wk/Wv:  d x d

三个 projection = 3 * 2 * 1 * d * d

3 * 2 * d * d = 100.7 MFLOPs
```

每层 output projection：

```text
out: 1 x d
Wo:  d x d

2 * d * d = 33.6 MFLOPs
```

每层 FFN：

```text
三个矩阵乘 = 3 * 2 * 1 * d * f

3 * 2 * d * f = 270.5 MFLOPs
```

每层 attention 读历史 KV：

```text
当前 Q:       1 x d
历史 K cache: S x d
历史 V cache: S x d

Q * K^T:       2 * S * d
softmax(QK)*V: 2 * S * d

QK + AV = 4 * S * d
        = 4 * 2048 * 4096
        = 33.6 MFLOPs
```

每层总计算量：

```text
100.7 + 33.6 + 270.5 + 33.6 = 438.4 MFLOPs
```

32 层：

```text
438.4M * 32 = 14.0 GFLOPs
```

logits：

```text
2 * d * V
= 2 * 4096 * 32000
= 0.26 GFLOPs
```

Decode 每 token 总计算量：

```text
14.0 + 0.26 = 14.3 GFLOPs/token
```

### 每个新 token 的访存量

权重读取：

```text
约 13.2 GB
```

读取历史 KV cache：

```text
每层 = 2 * S * d * 2 bytes
     = 33.6 MB

32 层 = 1.07 GB
```

写入新 token 的 KV cache：

```text
32 层 * 2 * d * 2 bytes
= 32 * 2 * 4096 * 2
= 1.0 MB
```

所以 decode 每 token 最低访存量约：

```text
13.2 GB + 1.07 GB + 0.001 GB = 14.3 GB/token
```

### Decode 算术强度

```text
算术强度 = 14.3 GFLOPs / 14.3 GB
        约等于 1 FLOP/Byte
```

这很低。

在同一个芯片上：

```text
分界点 = 50 FLOP/Byte
```

Decode 的 1 FLOP/Byte 远小于 50 FLOP/Byte，所以主要是带宽瓶颈。

理论时间下限：

```text
计算时间 = 14.3 GFLOPs / 100 TFLOP/s = 0.000143 s
访存时间 = 14.3 GB / 2000 GB/s = 0.00715 s
```

所以 decode 更应该优化访存：

- 权重量化，减少权重 bytes
- KV cache 量化，减少 KV bytes
- 多 batch 合并，让权重读取被更多 token 复用
- GQA/MQA，减少 KV cache 大小
- fused kernel，减少中间结果反复读写
- 更好的 cache 和 tile，让数据少从 HBM 走

## 怎么用它分析算子

拿到一个算子后，按这个顺序算：

1. 算 FLOPs
2. 算最低 Bytes
3. 算 `FLOPs / Bytes`
4. 和芯片的 `峰值算力 / 内存带宽` 比
5. 算理论时间下限
6. 拿实际运行时间对比

如果实际时间远高于理论下限，常见原因是：

- 数据没有连续访问
- tile 太小或太大
- 并行度不够
- kernel launch 或调度开销太高
- 中间结果反复落 HBM
- shape 不适合硬件矩阵单元
- 数据类型没有用到最快路径

## 最重要的判断

Prefill 像大矩阵乘，算术强度高，重点看算力利用率。

Decode 对 batch size 1 很不友好，算术强度低，重点看带宽、KV cache 和权重复用。

所以同一个模型、同一块芯片：

- prefill 慢，不一定是带宽问题
- decode 慢，不一定是算力问题
- 要先算 roofline，再谈优化方向
