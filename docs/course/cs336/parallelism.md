---
title: Parallelism
topics:
  - training-systems
comments: false
---

# CS336 · Parallelism

## 为什么需要并行 {#motivation}

<!-- 单卡显存与算力上限；计算、显存、通信三者的权衡。 -->

## 硬件与通信基础 {#hardware}

### GPU 互联：节点内与节点间

### 集合通信原语

<!-- all-reduce、reduce-scatter、all-gather、broadcast、all-to-all；all-reduce = reduce-scatter + all-gather。 -->

## 数据并行 {#data-parallelism}

### 朴素数据并行

### ZeRO / FSDP

<!-- 详细内容可参考博客《大模型并行训练技术之ZERO系列》。 -->

## 模型并行 {#model-parallelism}

### 张量并行（Tensor Parallelism）

### 流水线并行（Pipeline Parallelism）

## 序列并行与上下文并行 {#sequence-parallelism}

## 专家并行 {#expert-parallelism}

## 组合使用：3D / 4D 并行 {#combining}

## 代码实践 {#code}

<!-- torch.distributed 实现 collective、DDP、TP、PP 的最小示例。 -->

## 参考资料

- [CS336 课程官网](https://cs336.stanford.edu/)
- [CS336 GitHub](https://github.com/stanford-cs336)
