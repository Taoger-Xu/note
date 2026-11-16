---
title: HW2 · Policy Gradients
topics:
  - rlhf
  - post-training
comments: false
---

# CS285 Lab · HW2: Policy Gradients

<!-- 对应 Spring 2026 Homework 2（Policy Gradients），从 REINFORCE 逐步引入 Reward-to-Go、神经网络 Value Baseline 与 GAE。 -->

## 轨迹采样与折扣回报计算 {#trajectory-returns}

<!-- 完整轨迹回报（Full-trajectory Return）与因果性折扣回报（Reward-to-Go）向量化实现及方差对比。 -->

## 神经网络 Value Baseline {#value-baseline}

<!-- 状态价值网络 V_\phi(s) 回归训练、优势标准化（Advantage Normalization）与多步 Critic 更新。 -->

## 广义优势估计（GAE）实现 {#gae}

<!-- 逆向递推实现 GAE(\gamma, \lambda)，对比不同 \lambda 取值在连续控制任务上的偏差-方差表现。 -->

## 超参数调优实验 {#hyperparameter-tuning}

<!-- Batch Size、Learning Rate 与更新步数对策略收敛速度与稳定性的影响记录。 -->

## 实验记录与问题 {#notes}

## 参考资料

- [CS 185/285 课程官网（Spring 2026）](https://rail.eecs.berkeley.edu/deeprlcourse/)
- [CS 185/285 Homework Spring 2026 - hw2](https://github.com/berkeleydeeprlcourse/homework_spring2026/tree/main/hw2)
