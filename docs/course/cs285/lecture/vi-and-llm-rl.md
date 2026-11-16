---
title: Control as Inference & LLM RL
topics:
  - rlhf
  - post-training
  - reasoning
comments: false
---

# CS285 Lecture · Control as Inference & LLM RL

<!-- 对应 Spring 2026 Lecture 11–14（Variational Inference, VI in RL, Control as Inference & RL with Sequences and LLMs）。 -->

## 变分推断基础（Variational Inference） {#variational-inference}

<!-- 隐变量模型最大似然、证据下界（ELBO）推导、均场近似与摊销变分推断（Amortized VI / VAE）。 -->

## Control as Inference 与最大熵强化学习 {#control-as-inference}

<!-- 引入最优性二元随机变量 \mathcal{O}_t（p(\mathcal{O}_t=1|s_t,a_t) = \exp(r(s_t,a_t))）、后验轨迹推断 p(\tau|\mathcal{O}_{1:T})、软值函数消息传递与 KL 正则化 RL 目标的等价性。 -->

## 序列生成与大语言模型 MDP 建模 {#sequence-mdp}

<!-- 自回归 Token 生成作为确定性状态转移 MDP：Prompt 作为初始状态、Token 作为动作、序列级结果奖励（Outcome Reward）与过程奖励（PRM）。 -->

## RLHF 与 KL 约束对齐目标 {#rlhf-kl}

<!-- 偏好学习（Bradley-Terry 奖励模型）、参考策略 KL 惩罚 \beta D_{\mathrm{KL}}(\pi_\theta \| \pi_{\mathrm{ref}}) 防止 Reward Hacking，以及与重加权/DPO 的联系。 -->

## 可验证奖励推理与无 Critic 组相对策略优化（GRPO） {#rlvr-grpo}

<!-- 数学/代码可验证奖励（RLVR）；省去独立 Value Model 显存开销，对同一 Prompt 采样 G 条输出计算组内相对优势（GR-REINFORCE 与 GRPO Clipping + KL 目标）。 -->

## 参考资料

- [CS 185/285 课程官网（Spring 2026）](https://rail.eecs.berkeley.edu/deeprlcourse/)
- [Reinforcement Learning and Control as Probabilistic Inference: Tutorial and Review](https://arxiv.org/abs/1805.00909)
- [Training language models to follow instructions with human feedback (InstructGPT)](https://arxiv.org/abs/2203.02155)
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)](https://arxiv.org/abs/2402.03300)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
