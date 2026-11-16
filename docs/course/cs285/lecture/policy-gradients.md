---
title: Policy Gradients & Actor-Critic
topics:
  - rlhf
  - post-training
comments: false
---

# CS285 Lecture · Policy Gradients & Actor-Critic

<!-- 对应 Spring 2026 Lecture 4–6 & Lecture 9–10：MDP、策略梯度、Actor-Critic、GAE 与信赖域高级策略梯度（TRPO / PPO）。 -->

## 马尔可夫决策过程（MDP）与强化学习目标 {#mdp}

<!-- 状态空间、动作空间、转移核 p(s_{t+1}|s_t, a_t)、折扣回报与轨迹分布 p_\theta(\tau) 最大化目标 J(\theta)。 -->

## REINFORCE 与对数导数技巧 {#reinforce}

<!-- Likelihood Ratio Trick 推导 \nabla_\theta J(\theta)，与加权最大似然的联系，高方差来源剖析。 -->

## 方差缩减：因果性与 Baseline {#variance-reduction}

<!-- Causality（Reward-to-Go）剔除无关历史奖励；状态相关 Baseline b(s_t) 无偏性证明与最小方差最优常数 Baseline。 -->

## Actor-Critic 与广义优势估计（GAE） {#actor-critic-gae}

<!-- 价值网络 V^\pi_\phi(s) 拟合、TD 残差 \delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)、n-step 回报与 GAE(\gamma, \lambda) 指数加权折中。 -->

## Off-Policy 策略梯度与重要性采样 {#importance-sampling}

<!-- 轨迹级与边缘状态分布重要性权重，策略改进下界（Performance Difference Lemma）推导。 -->

## 高级策略梯度：Natural Gradient、TRPO 与 PPO {#trpo-ppo}

<!-- 参数空间欧氏约束 vs 策略分布 KL 约束、Fisher 信息矩阵与共轭梯度法（TRPO）、PPO Clipped Surrogate Objective 与自适应 KL 惩罚。 -->

## 参考资料

- [CS 185/285 课程官网（Spring 2026）](https://rail.eecs.berkeley.edu/deeprlcourse/)
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation (GAE)](https://arxiv.org/abs/1506.02438)
- [Trust Region Policy Optimization (TRPO)](https://arxiv.org/abs/1502.05477)
- [Proximal Policy Optimization Algorithms (PPO)](https://arxiv.org/abs/1707.06347)
