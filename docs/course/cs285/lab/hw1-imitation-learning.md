---
title: HW1 · Imitation Learning
topics:
  - rlhf
  - aigc
  - agent
comments: false
---

# CS285 Lab · HW1: Imitation Learning

<!-- 对应 Spring 2026 Homework 1（Imitation Learning），在 Push-T 平面推移操作环境上实现 Action Chunking 与 Flow Matching 策略。 -->

## 环境与 Push-T 演示数据集 {#setup}

<!-- Push-T 操作任务状态/动作空间、轨迹切片与 Action Chunking 数据加载（预测未来 H 步动作序列）梳理。 -->

## MSE Action-Chunking 行为克隆基线 {#mse-action-chunking}

<!-- 基于均方误差（MSE）回归动作块的策略训练与闭环重规划步数对比，观察多模态绕行分叉处的模式平均现象。 -->

## Flow Matching 策略实现 {#flow-matching-policy}

<!-- 噪声采样 x_0 \sim \mathcal{N}(0, I)、线性插值 x_t = (1-t)x_0 + t x_1、时间条件速度场网络 v_\theta(x_t, t, o) 训练与多步 Euler ODE 采样推理。 -->

## 消融对比：Chunk 长度与去噪步数 {#ablations}

<!-- Action Chunk 预测/执行步长、Euler 积分步数对 Push-T 成功率与轨迹平滑度的影响。 -->

## 实验记录与问题 {#notes}

## 参考资料

- [CS 185/285 课程官网（Spring 2026）](https://rail.eecs.berkeley.edu/deeprlcourse/)
- [CS 185/285 Homework Spring 2026 - hw1](https://github.com/berkeleydeeprlcourse/homework_spring2026/tree/main/hw1)
