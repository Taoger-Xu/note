---
title: Parallelism
topics:
  - training-systems
comments: false
---

# CS336 Lab · Parallelism

<!-- 对应 Assignment 2（Systems）中的分布式训练部分，按作业实际题目调整。 -->

## 环境与 torch.distributed 基础 {#setup}

## 集合通信 Benchmark {#benchmark}

<!-- 不同后端（gloo / NCCL）、数据量、进程数下 all-reduce 的耗时。 -->

## 朴素 DDP {#naive-ddp}

## DDP 优化：扁平化、通信重叠与分桶 {#ddp-optimizations}

## 优化器状态分片 {#optimizer-sharding}

## 实验记录与问题 {#notes}

## 参考资料

- [CS336 课程官网](https://cs336.stanford.edu/)
- [CS336 GitHub](https://github.com/stanford-cs336)
