---
title: CS285
topics:
  - rlhf
  - post-training
  - reasoning
  - agent
comments: false
---

# CS285 · Deep Reinforcement Learning (Spring 2026)

UC Berkeley 的深度强化学习课程（CS 185/285，Sergey Levine 主讲，Seohong Park 任 Head GSI）。2026 春季版在经典深度强化学习框架之上，全面融入了 **Action Chunking 与 Flow Matching 策略**、**大语言模型强化学习（LLM RL / GRPO）** 以及 **Flow Q-Learning（FQL）离线强化学习** 等前沿内容。

## 课程资料

- [课程官网（Spring 2026）](https://rail.eecs.berkeley.edu/deeprlcourse/)
- [课程作业 GitHub（homework_spring2026）](https://github.com/berkeleydeeprlcourse/homework_spring2026)
- [课程视频（RAIL YouTube 频道）](https://www.youtube.com/@rail7462/playlists)

## 学习笔记

### Lecture

- [Imitation Learning & Flow Policies](lecture/imitation-learning.md)：Behavioral Cloning、分布偏移、DAgger、Action Chunking 与 Flow Matching 生成式策略（L1–L3）。
- [Policy Gradients & Actor-Critic](lecture/policy-gradients.md)：MDP、REINFORCE、方差缩减、Actor-Critic、GAE 与信赖域优化 TRPO / PPO（L4–L6, L9–L10）。
- [Value-Based RL & Q-Learning](lecture/value-functions.md)：Bellman 算子、Fitted Q-Iteration、DQN 实践技巧（Replay / Target / Double Q）与连续动作 SAC（L7–L8）。
- [Control as Inference & LLM RL](lecture/vi-and-llm-rl.md)：变分推断、最大熵控制（Control as Inference）与序列/大语言模型强化学习（RLVR / GR-REINFORCE / GRPO）（L11–L14）。
- [Model-Based RL](lecture/model-based-rl.md)：最优控制（LQR / iLQR）、MPC、不确定性感知动力学集成与 Dyna / MBPO（L15–L16）。
- [Offline RL & Exploration](lecture/offline-rl-exploration.md)：离线分布偏移、SAC+BC / CQL / IQL / Flow Q-Learning、内在探索、RL 理论与多任务分层强化学习（L17–L20, L23–L24）。

### Lab

- [HW1 · Imitation Learning](lab/hw1-imitation-learning.md)：Push-T 环境下的 Action Chunking MSE 策略与 Flow Matching 策略实现。
- [HW2 · Policy Gradients](lab/hw2-policy-gradients.md)：REINFORCE、Reward-to-Go、神经网络 Value Baseline 与 GAE 实现及调优。
- [HW3 · Q-Learning & Actor-Critic](lab/hw3-dqn-sac.md)：DQN、Double DQN 与 Soft Actor-Critic（SAC）实现。
- [HW4 · LLM Reinforcement Learning](lab/hw4-llm-rl.md)：基于 Modal（H100）与 Qwen 基座模型的 GR-REINFORCE 与 GRPO 后训练实验（GSM8K / MATH）。
- [HW5 · Offline RL](lab/hw5-offline-rl.md)：离线数据集上的 SAC+BC、Implicit Q-Learning（IQL）与 Flow Q-Learning（FQL）实现与对比。

> **Final Project 默认赛道（Spring 2026）**：`LLM RL Track`（大模型推理强化学习）与 `Offline-to-Online RL Track`（离线预训练到在线微调）。

后续在这里整理课程笔记与作业记录。
