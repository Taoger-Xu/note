---
title: "RL-Kernel：大模型 RL 后训练中的训推一致性实践"
description: 介绍开源项目 RL-Kernel 如何通过确定性算子与统一数值契约，消除 rollout 引擎与训练引擎之间的 LogP 偏差，实现比特级训推一致性。
date: 2026-11-09
authors:
  - taoger
slug: rl-kernel
draft: true
categories:
  - 后训练
  - 强化学习
  - 训练系统
topics:
  - rlhf
tags:
  - RL-Kernel
  - 训推一致性
  - 强化学习
  - GRPO
  - PPO
  - vLLM
  - Megatron-LM
  - 确定性算子
---

# RL-Kernel：大模型 RL 后训练中的训推一致性实践

<!-- 在这里填写文章摘要，摘要将显示在 Blog 列表中。 -->

<!-- more -->

## 背景：训推不一致问题

<!-- rollout 引擎（vLLM）与训练引擎（Megatron-LM）对同一 token、同一权重算出的 LogP 不同；来源：kernel 实现、batching、规约顺序。影响：重要性采样比率、KL 项。 -->

## RL-Kernel 的定位

<!-- 位于执行引擎与加速器后端之间；运行时适配器按后端选择算子实现，同时保持 rollout 与训练共享同一数值契约；vime 编排 vLLM rollout + Megatron-LM 训练。 -->

## 核心算子

### 确定性 Attention

### 确定性 Dense FFN

### LogP 计算

### GRPO / PPO 目标函数

### 集合通信

## 使用方式

<!-- ./rlk run / ./rlk verify / ./rlk plan；.rlk-profile.json 与 RLK_REPRO_PROFILE；consistency 模式与 --mode native 对比。 -->

## 实验结果

<!-- H100 上 Qwen3-8B 200 步 LogP 完全一致；MI300X 三步检查 58.29 s/step vs native 57.40 s/step（+1.55%），raw-bit logprob 零偏差。 -->

## 硬件与生态支持

<!-- CUDA SM90（H100/H200/GH200）、ROCm gfx942（MI300 系列）已支持；Ascend 部分支持；摩尔线程 MUSA 进行中。路线图：DeepSeek-V4-Flash MoE、Miles、AReaL。 -->

## 总结与思考

## 参考资料

- [RL-Kernel GitHub](https://github.com/RL-Align/RL-Kernel)
- [RL-Kernel 文档](https://rl-align.github.io/RL-Kernel/)
- [RL-Kernel DeepWiki](https://deepwiki.com/RL-Align/RL-Kernel)
- [vLLM](https://github.com/vllm-project/vllm)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
