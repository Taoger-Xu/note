---
title: Value-Based RL & Q-Learning
topics:
  - rlhf
  - agent
comments: false
---

# CS285 Lecture · Value-Based RL & Q-Learning

<!-- 对应 Spring 2026 Lecture 7–8（Value-Based RL & Q-Learning in Practice）：动态规划、Fitted Q-Iteration、DQN 实践技巧与连续动作控制。 -->

## 值函数与 Bellman 最优算子 {#bellman}

<!-- Policy Iteration 与 Value Iteration、Bellman 最优算子 \mathcal{B}^* 在无穷范数下的 \gamma-压缩映射性质与不动点收敛。 -->

## Fitted Q-Iteration 与在线 Q-Learning {#q-learning}

<!-- 函数逼近下的非线性回归拟合、Semi-gradient 更新本质、离策略 + 函数逼近 + 自举（Deadly Triad）导致的发散分析。 -->

## Q-Learning 实践机制：DQN 家族 {#dqn-in-practice}

<!-- Replay Buffer 去相关性、Target Network（硬更新 / Polyak 软更新）稳定回归目标；最大化高估偏差（Overestimation Bias）与 Double DQN、Multi-step Q-Learning。 -->

## 连续动作空间的 Q-Learning {#continuous-actions}

<!-- 随机采样 / CEM 近似 \max_a Q(s,a)、确定性策略梯度（DDPG）与 TD3（Clipped Double-Q、Delayed Policy Updates、Target Policy Smoothing）。 -->

## Soft Actor-Critic（SAC）实务 {#sac}

<!-- 熵正则化软 Q 更新、重参数化技巧（Reparameterization Trick）低方差策略梯度与自适应温度系数 \alpha 调节。 -->

## 参考资料

- [CS 185/285 课程官网（Spring 2026）](https://rail.eecs.berkeley.edu/deeprlcourse/)
- [Human-level control through deep reinforcement learning (DQN)](https://www.nature.com/articles/nature14236)
- [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
- [Addressing Function Approximation Error in Actor-Critic Methods (TD3)](https://arxiv.org/abs/1802.09477)
- [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905)
