---
title: Imitation Learning & Flow Policies
topics:
  - rlhf
  - aigc
  - agent
comments: false
---

# CS285 Lecture · Imitation Learning & Flow Policies

<!-- 对应 Spring 2026 Lecture 1–3（Introduction & Behavioral Cloning Part 1–2）：从监督行为克隆到 Action Chunking 与 Flow Matching 策略。 -->

## 符号体系与行为克隆（Behavioral Cloning） {#behavioral-cloning}

<!-- 观测 o_t、状态 s_t、动作 a_t、轨迹 \tau 与最大似然监督训练目标 \max_\theta \mathbb{E}_{(o,a)\sim\mathcal{D}}[\log \pi_\theta(a|o)]。 -->

## 分布偏移与复合误差分析 {#distributional-shift}

<!-- 训练轨迹分布 p_{data}(o_t) 与测试策略轨迹分布 p_{\pi_\theta}(o_t) 失配；单步误差 \epsilon 导致 O(\epsilon T^2) 累积误差的界定，以及通过 DAgger 收敛至 O(\epsilon T)。 -->

## 非马尔可夫行为与 Action Chunking {#action-chunking}

<!-- 人类演示中的时间相关性、停顿与多步协调；一次预测未来 k 步动作序列 a_{t:t+k} 与时域集成（Temporal Ensembling）。 -->

## 多模态动作分布与生成式策略 {#generative-policies}

<!-- MSE / 单高斯策略在多峰专家动作（如 Push-T 左右绕行）上的模式平均（Mode Averaging）失效；离散自回归、扩散策略（Diffusion Policy）对比。 -->

## Flow Matching 策略建模 {#flow-matching}

<!-- 连续时间常微分方程（ODE）传输映射、线性插值路径 x_t = (1-t)x_0 + t x_1、条件向量场回归损失 \|v_\theta(x_t, t, o) - (x_1 - x_0)\|^2 与 Euler 步积分采样。 -->

## 参考资料

- [CS 185/285 课程官网（Spring 2026）](https://rail.eecs.berkeley.edu/deeprlcourse/)
- [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT / Action Chunking)](https://arxiv.org/abs/2304.13705)
- [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137)
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
