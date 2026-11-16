---
title: Offline RL & Exploration
topics:
  - rlhf
  - post-training
  - agent
comments: false
---

# CS285 Lecture · Offline RL & Exploration

<!-- 对应 Spring 2026 Lecture 17–20 & Lecture 23–24：Offline RL 算法（含 Flow Q-Learning）、探索机制、RL 理论与多任务分层强化学习。 -->

## 离线强化学习挑战：分布外动作与高估 {#offline-rl-challenges}

<!-- 静态数据集 \mathcal{D} 下无法在线交互纠错；轨迹拼接（Trajectory Stitching）潜力与 Bellman 备份查询分布外动作（OOD Actions）引发的 Q 值高估崩溃。 -->

## 策略约束与样本内价值学习（SAC+BC / CQL / IQL） {#sac-bc-cql-iql}

<!-- 行为克隆正则化（TD3+BC / SAC+BC）、保守值函数下界（CQL）、以及完全在数据集动作内做 Expectile 回归的 Implicit Q-Learning（IQL）。 -->

## 生成式离线强化学习：Flow Q-Learning（FQL） {#flow-q-learning}

<!-- 将高表达力 Flow Matching 策略与 Q-Learning 结合；避免穿过 ODE 多步积分反向传播的单步蒸馏策略更新，兼顾多模态行为先验与价值最大化。 -->

## 探索机制（Exploration & Advanced Exploration） {#exploration}

<!-- 多臂赌博机遗憾界、UCB 与后验采样（Thompson Sampling）；高维状态空间下的伪计数（Pseudo-counts）、预测误差内在奖励（RND）与信息增益探索。 -->

## 强化学习理论与多任务/分层强化学习 {#theory-multitask-hrl}

<!-- 样本复杂度、集中性系数（Concentrability）与线性/函数逼近可学习性；目标条件强化学习（Goal-Conditioned RL）、HER 与分层技能发现（Hierarchical RL）。 -->

## 参考资料

- [CS 185/285 课程官网（Spring 2026）](https://rail.eecs.berkeley.edu/deeprlcourse/)
- [Conservative Q-Learning for Offline Reinforcement Learning (CQL)](https://arxiv.org/abs/2006.04779)
- [Offline Reinforcement Learning with Implicit Q-Learning (IQL)](https://arxiv.org/abs/2110.06169)
- [Flow Q-Learning (FQL)](https://arxiv.org/abs/2502.02538)
- [Exploration by Random Network Distillation (RND)](https://arxiv.org/abs/1810.12894)
