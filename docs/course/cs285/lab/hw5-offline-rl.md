---
title: HW5 · Offline RL
topics:
  - rlhf
  - post-training
  - agent
comments: false
---

# CS285 Lab · HW5: Offline Reinforcement Learning

<!-- 对应 Spring 2026 Homework 5（Offline RL），在固定离线数据集上对比标准 Off-Policy RL 退化现象及 SAC+BC、IQL 与 Flow Q-Learning（FQL）。 -->

## 纯离线环境与朴素 SAC 崩溃复现 {#naive-offline-rl}

<!-- 在无环境交互的离线基准数据集上运行标准 SAC，观察分布外动作导致的 Q 值发散与评估回报暴跌。 -->

## 行为克隆正则化：SAC+BC {#sac-bc}

<!-- 在 Actor 目标中叠加数据集动作 BC 约束项，分析权重系数在防止分布偏移与实现轨迹拼接（Stitching）之间的权衡。 -->

## Implicit Q-Learning（IQL）实现 {#iql}

<!-- Expectile Loss（L_2^\tau）拟合状态价值 V(s)、完全样本内更新 Q(s,a)，并结合优势加权回归（AWR）提取策略。 -->

## Flow Q-Learning（FQL）实现 {#flow-q-learning}

<!-- 训练 Flow Matching 行为先验速度场，结合单步蒸馏策略与 Q 值最大化目标，对比高斯策略（SAC+BC / IQL）在复杂多模态离线控制任务上的优势。 -->

## 实验记录与问题 {#notes}

## 参考资料

- [CS 185/285 课程官网（Spring 2026）](https://rail.eecs.berkeley.edu/deeprlcourse/)
- [CS 185/285 Homework Spring 2026 - hw5](https://github.com/berkeleydeeprlcourse/homework_spring2026/tree/main/hw5)
- [Offline Reinforcement Learning with Implicit Q-Learning (IQL)](https://arxiv.org/abs/2110.06169)
- [Flow Q-Learning (FQL)](https://arxiv.org/abs/2502.02538)
