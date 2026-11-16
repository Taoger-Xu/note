---
title: 强化学习基础理论
description: 从 MDP、价值函数与贝尔曼方程出发，梳理动态规划、蒙特卡洛、时序差分、策略梯度到 Actor-Critic 与 PPO 的强化学习基础理论。
date: 2026-11-06
authors:
  - taoger
slug: rl-basics
draft: false
categories:
  - 后训练
  - 强化学习
topics:
  - rlhf
tags:
  - 强化学习
  - MDP
  - 贝尔曼方程
  - 策略梯度
  - Actor-Critic
  - PPO
---

# 强化学习基础理论

<!-- 在这里填写文章摘要，摘要将显示在 Blog 列表中。 -->

<!-- more -->

## 路线图

<figure class="rl-roadmap-figure">
--8<-- "docs/blog/posts/post-training/assets/rl-basics/rl-roadmap.svg"
<figcaption>两条主线：基于价值的方法逐步去掉对模型的依赖，最后用神经网络近似 Q；基于策略的方法直接优化策略，并借用 TD 学到的价值函数作为 Critic。</figcaption>
</figure>

## 基本概念：智能体、环境与 MDP

RL 解决的是决策问题。监督学习从给定的 data 里学一个映射，模型的输出不会影响之后看到的数据；而 RL 中 Agent 的 action 会改变环境的 state，进而影响之后所有能拿到的 reward。所以 RL 学的不是去拟合标签，而是一个策略，目标是在长期交互中最大化累积奖励的期望。

### 交互框架

<figure class="rl-roadmap-figure">
--8<-- "docs/blog/posts/post-training/assets/rl-basics/rl-loop.svg"
<figcaption>Agent 在 <i>S</i><sub>t</sub> 下按策略选 <i>A</i><sub>t</sub>，环境返回 <i>S</i><sub>t+1</sub> 和 <i>R</i><sub>t+1</sub>。</figcaption>
</figure>

每个时间步 $t$，Agent 观察到状态 $S_t$，选择动作 $A_t$，环境转移到 $S_{t+1}$ 并给出奖励 $R_{t+1}$。不断重复这个过程，就得到一条轨迹（trajectory）：

$$
\tau = (S_0, A_0, R_1, S_1, A_1, R_2, \dots)
$$

策略（policy）$\pi(a\mid s)$ 表示在状态 $s$ 下选择动作 $a$ 的概率，确定性策略可以写成 $a=\pi(s)$。奖励 $R_{t+1}$ 只评价单步动作的好坏，而决策关心的是长期收益，因此定义从 $t$ 时刻开始的回报（return）：

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
$$

RL 的目标就是找到使期望回报最大的策略 $\pi^* = \arg\max_{\pi} \mathbb{E}_{\tau \sim \pi}[G_0]$。

### 为什么用 MDP

如果下一个状态依赖于全部历史 $S_1,\dots,S_t$，那么随着交互进行，需要考虑的信息会越来越长，没法处理。MDP 假设环境的转移满足马尔可夫性质：

$$
\mathbb{P}[S_{t+1}\mid S_t] = \mathbb{P}[S_{t+1}\mid S_1, \dots, S_t]
$$

即当前状态是未来的充分统计量，知道 $S_t$ 之后历史就可以丢掉了。这也是后面贝尔曼方程能写成一步递推的前提。

MDP 可以由一个五元组 $(S, A, P_{sa}, \gamma, R)$ 表示：

- $S$ 是状态的集合，比如迷宫中的位置、Atari 游戏中的当前屏幕
- $A$ 是动作的集合，比如向 N、E、S、W 移动，手柄操纵杆方向和按钮
- $P_{sa}$ 是状态转移概率：对每个 $s\in S$ 和 $a\in A$，$P_{sa}$ 是下一个状态在 $S$ 上的概率分布
- $\gamma\in[0,1]$ 是对未来奖励的折扣因子
- $R: S\times A \mapsto \mathbb{R}$ 是奖励函数，有时奖励只和状态相关，写作 $R(s)$

## 价值函数与贝尔曼方程

<!-- V(s)、Q(s,a)、优势函数 A(s,a)，贝尔曼期望方程与最优方程。 -->

## 动态规划：策略迭代与价值迭代

## 蒙特卡洛与时序差分

<!-- MC、TD(0)、TD(λ)、GAE；偏差与方差的权衡；SARSA 与 Q-learning。 -->

## 策略梯度

<!-- 策略梯度定理、REINFORCE、baseline 与方差缩减、重要性采样。 -->

## Actor-Critic

## 信任域方法：TRPO 与 PPO

## 从 RL 到 LLM 后训练

<!-- RLHF、GRPO 等方法与经典 RL 的对应关系。 -->

## 总结

## 参考资料

- [David Silver：UCL Course on Reinforcement Learning](https://davidstarsilver.wordpress.com/teaching/)
- [动手学强化学习](https://hrl.boyuai.com/)

<!-- 本文配图放在 assets/rl-basics/ 下，用相对路径引用。 -->
