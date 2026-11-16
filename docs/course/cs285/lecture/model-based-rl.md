---
title: Model-Based RL
topics:
  - rlhf
  - agent
comments: false
---

# CS285 Lecture · Model-Based RL

<!-- 对应 Spring 2026 Lecture 15–16（Model-Based RL Part 1 & Part 2）：最优控制、MPC、不确定性感知动力学建模与 MBPO。 -->

## 开环与闭环轨迹优化 {#trajectory-optimization}

<!-- 已知动力学下的有限时域最优控制；免梯度采样优化：随机射击（Random Shooting）与交叉熵方法（CEM）。 -->

## LQR 与迭代 LQR（iLQR） {#lqr-ilqr}

<!-- 线性二次调节器（LQR）后向 Riccati 递推解析解；非线性动力学与代价二阶/一阶局部展开（iLQR / DDP）与线搜索。 -->

## 学习动力学模型与模型预测控制（MPC） {#mpc}

<!-- 分布偏移导致的模型漏洞利用（Model Exploitation）；滚动时域重规划（MPC）与在线数据聚合重训循环。 -->

## 不确定性感知建模（Uncertainty-Aware Models） {#uncertainty-ensembles}

<!-- 偶然不确定性（Aleatoric）与认知不确定性（Epistemic）；深度集成模型（Bootstrap Ensembles / PETS）评估预测分布。 -->

## 基于模型的策略优化：Dyna 与 MBPO {#dyna-mbpo}

<!-- 穿过多步动力学模型反向传播的不稳定性；从真实状态出发生成短步长合成 Rollout（Branched Rollouts）训练无模型策略（MBPO）与潜空间世界模型（World Models）。 -->

## 参考资料

- [CS 185/285 课程官网（Spring 2026）](https://rail.eecs.berkeley.edu/deeprlcourse/)
- [Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models (PETS)](https://arxiv.org/abs/1805.12114)
- [When to Trust Your Model: Model-Based Policy Optimization (MBPO)](https://arxiv.org/abs/1906.08253)
