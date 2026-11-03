---
title: 大模型并行训练技术之ZERO系列
description: 整理 ZeRO 系列并行训练技术的背景、各阶段原理、通信与显存开销，以及实践记录。
date: 2026-11-01
authors:
  - taoger
slug: zero-series
draft: false
categories:
  - 训练系统
  - 分布式训练
topics:
  - training-systems
tags:
  - 大模型
  - 并行训练
  - 分布式训练
  - 数据并行
  - ZeRO
  - ZeRO-1
  - ZeRO-2
  - ZeRO-3
  - DeepSpeed
  - 显存优化
---

# 大模型并行训练技术之ZERO系列

用 Adam + 混合精度训练一个参数量为 $\Psi$ 的模型，光是“模型状态”就要占 $16\Psi$ 字节显存；而在普通数据并行（DP）里，每张卡都完整地存一份。ZeRO（Zero Redundancy Optimizer）的思路是：既然所有卡存的都是同一份东西，那就把它切开，每张卡只保管自己负责的那一片。按切分对象的不同，分为 ZeRO-1（优化器状态）、ZeRO-2（+ 梯度）、ZeRO-3（+ 参数）三个阶段。

<!-- more -->

## 背景与动机

### 每个参数的 4 种状态

![Adam 中每个参数对应的四种状态](assets/zero-series/adam-states.png)

最朴素的梯度下降只需要参数 $W$ 和梯度 $\nabla W$：

$$
W \leftarrow W - \alpha \nabla W
$$

而大模型训练普遍使用 Adam，它为每个参数额外维护两个状态：

$$
\begin{aligned}
m &\leftarrow \beta_1 m + (1-\beta_1)\nabla W && \text{（一阶矩 / Momentum）}\\
v &\leftarrow \beta_2 v + (1-\beta_2)(\nabla W)^2 && \text{（二阶矩 / Variance）}\\
W &\leftarrow W - \alpha \frac{m}{\sqrt{v}+\epsilon}
\end{aligned}
$$

所以每个参数在训练时对应 **4 种状态**：

| 状态 | 作用 |
| --- | --- |
| Parameters $W$ | 前向 / 反向计算用的权重 |
| Gradients $\nabla W$ | 反向传播得到的梯度 |
| Momentum $m$ | 梯度的指数滑动平均 |
| Variance $v$ | 梯度平方的指数滑动平均 |

其中 $m$、$v$ 只在优化器 step 时用到，统称**优化器状态（optimizer states）**。

### 混合精度训练

![混合精度训练流程](assets/zero-series/mixed-precision.png)

如果 4 种状态都用 FP32，每个参数要 $4\times 4 = 16$ 字节。混合精度训练把计算密集的部分换成 16 位，但为了数值稳定，保留一份 FP32 的副本：

- **FP16 区域**（前向 + 反向）：参数 $W$ 和梯度 $\nabla W$ 都是 FP16，矩阵乘法可以跑在 Tensor Core 上，速度快、显存省。
- **FP32 区域**（优化器更新）：梯度转成 FP32 后，与 FP32 的 $m$、$v$ 一起更新一份 **FP32 Master Weights**，更新完再 cast 回 FP16 供下一轮前向使用。

为什么必须保留 FP32？有两个精度问题：

1. **梯度下溢**：FP16（1 位符号 / 5 位指数 / 10 位尾数）能表示的最小正数约为 $6\times10^{-8}$，像 $10^{-8}$ 这样的小梯度会直接变成 0。解决办法是 **Loss Scaling**：先把 loss 乘上一个系数（如 $\times 1024$），由链式法则所有梯度也被放大，$10^{-8}\to 1.024\times10^{-5}$，落回 FP16 的可表示范围；更新前再除回去。
2. **更新被吞掉**：FP16 在 1 附近的间隔约为 $10^{-3}$，所以 $1 + 0.0001$ 在 FP16 下仍等于 $1$，微小的更新量直接丢失。在 FP32 的 Master Weights 上做更新，$1+0.0001=1.0001$ 才能被正确累积。

### 数据并行下的显存开销

![数据并行下每张卡都保存完整的模型状态](assets/zero-series/dp-memory.png)

把混合精度下每个参数的显存开销加起来（$\Psi$ 为参数量）：

| 状态 | 精度 | 字节 / 参数 |
| --- | --- | --- |
| Parameters | FP16 | 2 |
| Gradients | FP16 | 2 |
| Master Weights | FP32 | 4 |
| Momentum | FP32 | 4 |
| Variance | FP32 | 4 |
| **合计** | | **16** |

$$
2\Psi + 2\Psi + 12\Psi = 16\Psi
$$

其中 $12\Psi$ 的 FP32 部分（Master + $m$ + $v$）统称优化器状态，占了大头。以 7.5B 模型为例，模型状态需要 $16\times7.5\text{B}=120\text{GB}$，远超一张 32GB 的 V100。

普通数据并行中，每张 GPU 拿不同的数据，但**都完整保存这 120GB**：4 张卡就有 4 份一模一样的优化器状态、梯度和参数，这就是 ZeRO 要消除的冗余。（这里只讨论模型状态，不含激活值、临时缓冲等 residual states。）

## ZeRO 系列概览

### 从 All-reduce 到 Reduce-scatter + All-gather

![All-reduce 拆分为 Reduce-scatter 和 All-gather](assets/zero-series/allreduce-rs-ag.png)

数据并行每一步要做梯度同步：目标是让每张 GPU 都拿到**相同的平均梯度**，用的通信原语是 All-reduce。在 NCCL 的 Ring 实现中，All-reduce 本身就是两步：

1. **Reduce-scatter**：把梯度切成 $N$ 片，第 $i$ 张卡负责第 $i$ 片，把 $N$ 份副本加起来再除以 $N$。结束后**每张卡只有自己那一片是有效的平均梯度**，其余 $N-1$ 片已经“发出去了，不会再用”（图中的死鱼 🐟）。
2. **All-gather**：每张卡把自己那片有效结果广播给其他卡，最终人人都有完整的平均梯度。

设参数量为 $\Psi$、卡数为 $N$，每张卡在每一步的发送量为：

| 原语 | 单卡通信量 |
| --- | --- |
| Reduce-scatter | $\frac{N-1}{N}\Psi \approx \Psi$（图中 $3\times \Psi/4$） |
| All-gather | $\frac{N-1}{N}\Psi \approx \Psi$ |
| All-reduce = RS + AG | $2\frac{N-1}{N}\Psi \approx 2\Psi$ |

注意：把 All-reduce 拆成 RS + AG **本身并不会减少通信量**，仍然是 $2\Psi$。真正的关键观察是 **Reduce-scatter 之后**：

> 每张卡只持有 $1/N$ 的有效平均梯度。如果第 $i$ 张卡只负责更新第 $i$ 片参数，那它根本不需要完整的梯度，也不需要完整的优化器状态。

于是 ZeRO 把 All-gather 从“收集梯度”挪到了“收集更新后的参数”：

$$
\underbrace{\text{RS(梯度)} + \text{AG(梯度)}}_{\text{DP: } 2\Psi} \quad\Longrightarrow\quad \underbrace{\text{RS(梯度)} \to \text{本地更新 } 1/N \to \text{AG(参数)}}_{\text{ZeRO-1/2: } 2\Psi}
$$

通信量不变，但每张卡上的状态可以只存 $1/N$。根据切分的对象不同，就得到了 ZeRO 的三个阶段：

| 阶段 | 切分对象 | 符号 |
| --- | --- | --- |
| ZeRO-1 | 优化器状态 | $P_{os}$ |
| ZeRO-2 | 优化器状态 + 梯度 | $P_{os+g}$ |
| ZeRO-3 | 优化器状态 + 梯度 + 参数 | $P_{os+g+p}$ |

## ZeRO-1

![ZeRO stage 1：切分优化器状态](assets/zero-series/zero-stage1.png)

**核心问题：哪张 GPU 更新哪部分权重？**

ZeRO-1 把占大头的 $12\Psi$ 优化器状态（FP32 Master、$m$、$v$）切成 $N$ 片，第 $i$ 张卡只保存第 $i$ 片（Shard $i$），也只负责更新这一片对应的权重。FP16 的参数和梯度仍然每卡一份完整的。

一个训练 step 的流程：

1. **前向 + 反向**：每张卡用自己的数据和完整的 FP16 参数算出完整的 FP16 梯度。
2. **Reduce-scatter 梯度**：第 $i$ 张卡得到第 $i$ 片的平均梯度（通信 $\Psi$）。
3. **本地更新**：用这片梯度和本地的 $m_i, v_i$ 更新 FP32 Master Weights 的第 $i$ 片，再转成 FP16。
4. **All-gather 参数**：各卡交换自己更新好的 FP16 参数片，拼回完整参数，供下一步前向使用（通信 $\Psi$）。

显存开销：

$$
2\Psi + 2\Psi + \frac{12\Psi}{N}
$$

7.5B 模型、4 张卡：$15 + 15 + 90/4 = 52.5\text{GB}$，相比 DP 的 120GB 降了一半多；当 $N$ 很大时趋近 $4\Psi$，即约 **4 倍**的显存节省。通信量仍为 $2\Psi$，与 DP 相同。

## ZeRO-2

ZeRO-1 中还有一处明显的浪费：Reduce-scatter 之后，每张卡上 $N-1$ 片梯度已经是“死鱼”——发出去以后不再参与任何计算，却还占着显存。

ZeRO-2 在 ZeRO-1 基础上**把梯度也切分**：第 $i$ 张卡只保留第 $i$ 片的平均梯度，其余梯度在 Reduce-scatter 后立即释放。实际实现中，反向传播是逐层进行的，DeepSpeed 会把梯度分桶（bucket），某个桶的梯度一算完就做 Reduce-scatter 并释放非本卡负责的部分，所以峰值显存里不会出现完整的梯度。

显存开销：

$$
2\Psi + \frac{(2+12)\Psi}{N}
$$

7.5B、4 张卡：$15 + 105/4 = 41.25\text{GB}$；$N$ 很大时趋近 $2\Psi$，约 **8 倍**节省。通信流程与 ZeRO-1 一样（RS 梯度 + AG 参数），仍为 $2\Psi$。

## ZeRO-3

到了 ZeRO-2，每张卡上剩下的冗余只有完整的 FP16 参数 $2\Psi$。ZeRO-3 把**参数也切分**，每张卡只常驻 $1/N$ 的参数。

代价是计算时需要“临时借”参数：

1. **前向**：计算到某一层前，All-gather 这一层的完整参数，算完立即丢弃非本卡的部分（通信 $\Psi$）。
2. **反向**：再次逐层 All-gather 参数用于计算梯度，用完即丢（通信 $\Psi$）。
3. **梯度同步**：对梯度做 Reduce-scatter，每张卡拿到自己那片（通信 $\Psi$）。
4. **本地更新**：直接更新本卡那片参数，不再需要最后的 All-gather。

显存开销：

$$
\frac{(2+2+12)\Psi}{N} = \frac{16\Psi}{N}
$$

7.5B、4 张卡：$120/4 = 30\text{GB}$，终于塞进了 32GB 的卡；显存节省与卡数 $N$ 成**线性**关系。通信量变为 $3\Psi$，是 DP 的 1.5 倍。实践中通过预取（prefetch）下一层参数，让 All-gather 和计算重叠来掩盖这部分开销。

## 显存与通信开销对比

以 $\Psi=7.5\text{B}$ 为例（$N=64$ 为 ZeRO 论文中的设置）：

| 方法 | 单卡显存公式 | $N=4$ | $N=64$ | 单卡通信量 |
| --- | --- | --- | --- | --- |
| DP | $16\Psi$ | 120 GB | 120 GB | $2\Psi$ |
| ZeRO-1 | $4\Psi + \frac{12\Psi}{N}$ | 52.5 GB | 31.4 GB | $2\Psi$ |
| ZeRO-2 | $2\Psi + \frac{14\Psi}{N}$ | 41.25 GB | 16.6 GB | $2\Psi$ |
| ZeRO-3 | $\frac{16\Psi}{N}$ | 30 GB | 1.9 GB | $3\Psi$ |

一句话总结三个阶段的演进：

- **ZeRO-1**：Reduce-scatter 后每张卡只需更新 $1/N$ 的权重 → 只存 $1/N$ 的优化器状态。
- **ZeRO-2**：Reduce-scatter 后其余梯度不再使用 → 只存 $1/N$ 的梯度。
- **ZeRO-3**：参数只在计算时才需要完整 → 只存 $1/N$ 的参数，用时 All-gather。

## 实践记录

<!-- 记录实验环境、配置、结果与遇到的问题。 -->

## 总结

ZeRO 没有改变数据并行“每张卡处理不同数据”的本质，而是观察到 DP 中模型状态的冗余，并利用 All-reduce = Reduce-scatter + All-gather 这一结构，让每张卡只保管、只更新自己那 $1/N$ 的状态。ZeRO-1/2 在通信量不变的前提下分别带来约 4 倍和 8 倍的显存节省；ZeRO-3 以 1.5 倍通信为代价，让显存随卡数线性下降。

## 参考资料

- Rajbhandari et al. *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*. SC 2020. [arXiv:1910.02054](https://arxiv.org/abs/1910.02054)
- Micikevicius et al. *Mixed Precision Training*. ICLR 2018. [arXiv:1710.03740](https://arxiv.org/abs/1710.03740)
- [DeepSpeed ZeRO 文档](https://www.deepspeed.ai/tutorials/zero/)
