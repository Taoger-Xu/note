---
title: HW4 · LLM RL
topics:
  - rlhf
  - post-training
  - reasoning
comments: false
---

# CS285 Lab · HW4: LLM Reinforcement Learning

<!-- 对应 Spring 2026 Homework 4（LLM RL），基于 Modal（H100 GPU）在 Qwen 基座模型上从零实现 GR-REINFORCE 与 GRPO 后训练。 -->

## Modal（H100）环境与训练管线 {#setup}

<!-- Rollout 采样、Prompt / Response Masking、逐 Token 对数概率（Per-token Log-Probabilities）计算与梯度累积 Minibatching 实现。 -->

## 结果奖励与组相对优势（GR-REINFORCE） {#gr-reinforce}

<!-- 对同一 Prompt 采样 G 条回复并验证答案/格式正确性，计算组内均值与标准差归一化优势（Group-Relative Advantage），在快速格式复制任务上验证收敛。 -->

## Group Relative Policy Optimization（GRPO）实现 {#grpo}

<!-- 新旧策略重要性比率 r_{i,t}(\theta)、PPO 风格雙边裁剪（Clipped Surrogate Objective）与参考模型无偏 KL 惩罚估计器实现。 -->

## 数学推理评测实验（GSM8K / MATH） {#math-reasoning}

<!-- 不同组大小 G、KL 系数 \beta、裁剪范围或优势归一化方式对数学推理准确率、回复长度与训练稳定性的影响对比。 -->

## 实验记录与问题 {#notes}

## 参考资料

- [CS 185/285 课程官网（Spring 2026）](https://rail.eecs.berkeley.edu/deeprlcourse/)
- [CS 185/285 Homework Spring 2026 - hw4](https://github.com/berkeleydeeprlcourse/homework_spring2026/tree/main/hw4)
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)](https://arxiv.org/abs/2402.03300)
