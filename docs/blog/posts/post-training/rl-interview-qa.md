---
title: 强化学习面经 QA
description: 整理大模型强化学习方向的常见面试问题，覆盖 RL 基础、PPO / GRPO 系列算法、训练实践与 RL Infra。
date: 2026-11-06
authors:
  - taoger
slug: rl-interview-qa
draft: false
categories:
  - 后训练
  - 强化学习
topics:
  - rlhf
  - training-systems
tags:
  - 强化学习
  - 面经
  - PPO
  - GRPO
  - DAPO
  - GSPO
  - RL Infra
  - 异步训练
---

# 强化学习面经 QA

<!-- 在这里填写文章摘要，摘要将显示在 Blog 列表中。 -->

<!-- more -->

## 算法部分

### Q1：为什么要用 Actor-Critic 而不是纯 Critic？

<!-- 回答 -->

### Q2：KL 散度和交叉熵、MLE 的关系？

<!-- 回答 -->

### Q3：不同 RL 场景应该如何设计 Reward？

<!-- 回答 -->

### Q4：如何理解 RL 中的 importance sampling / rejection sampling 等 Monte Carlo 方法？

<!-- 回答 -->

### Q5：PPO / GRPO 的 advantage 是怎么算的，为什么要减去 baseline，这里一定要除以 std 吗？

<!-- 回答 -->

### Q6：RL training 和 test-time scaling 各自是如何 explore 的？

<!-- 回答 -->

### Q7：PPO 是如何 clip 的，为什么要取 min，不 clip 会怎么样，CISPO 是怎么做的？

<!-- 回答 -->

### Q8：GRPO 为什么加上 KL 散度，KL 散度怎么计算，为什么 DAPO、GSPO 又去掉了 KL 散度？

<!-- 回答 -->

### Q9：在 LLM 训练时，如果不小心多 All Reduce 了几次 loss，会发生什么？

<!-- 回答 -->

### Q10：DPO 的 reward 是什么，会不会 reward hacking，如何解决？

<!-- 回答 -->

### Q11：有哪些解决 MoE 训推不一致问题的算法，各自是什么原理？

<!-- 回答 -->

### Q12：RL 训练时，group size / learning rate / ppo epoch / generation length 如何设置？

<!-- 回答 -->

### Q13：相比 GRPO，Dr.GRPO / DAPO / GSPO / CISPO / SAPO / DPPO / MaxRL / SimKO 是如何改进的，各自又有什么缺点？

<!-- 回答 -->

### Q14：TRPO / DPPO / AReaL 是如何用 trust region 约束 RL objective 的？

<!-- 回答 -->

### Q15：RL 能否拓展 LLM 的能力边界？

<!-- 回答 -->

### Q16：结合 ProRL 等工作，谈谈如何 scale RL 训练边界？

<!-- 回答 -->

### Q17：OPD 相比于传统 RL / SFT 的改进，有哪些 OPD 的应用？

<!-- 回答 -->

### Q18：LLM 推理能力是在哪一个训练阶段产生的？

<!-- 回答 -->

### Q19：DeepSeek R1 到 V3.2 / V4，RL 部分有哪些改进，MoE RL 有什么不同？

<!-- 回答 -->

## Infra 部分

### Q20：不考虑 CPU offload，GRPO 训练时显存里有几个模型，考虑了能省多少显存？

<!-- 回答 -->

### Q21：分布式推理：KV cache 传输优化、多卡通信优化策略

<!-- 回答 -->

### Q22：INT8 与 FP8 优劣对比，训推分别用什么精度？

<!-- 回答 -->

### Q23：RL rollout 中的长尾问题是什么，有哪些解决方案？

<!-- 回答 -->

### Q24：continuous batching 在 RL 训练时会有什么问题，vLLM 和 SGLang 的区别？

<!-- 回答 -->

### Q25：vLLM / SGLang 怎么看利用率，KV cache 在训练里的利用率怎么看？

<!-- 回答 -->

### Q26：多机多卡 RL 训练时如何实现反向传播？

<!-- 回答 -->

### Q27：RL 训练有哪些异步框架，解决了同步训练的什么问题？

<!-- 回答 -->

### Q28：AReaL 或者其他 partial rollout 框架，在 rollout 时，会不会保存之前 policy 的 KV cache？

<!-- 回答 -->

### Q29：MoE 的 EP 对 throughput 的影响

<!-- 回答 -->

### Q30：Long context 场景下的 compute-communication overlap，Megatron 和 FSDP 各自的 parallelism

<!-- 回答 -->

### Q31：确定性模式怎么开，什么是 batch invariance，是什么导致的，有没有 atom add，atom add 能解决吗？

<!-- 回答 -->

### Q32：AReaL 和 slime 对 RL rollout bottleneck 的理解有什么不同？

<!-- 回答 -->

### Q33：full async staleness 怎么看，训练时大概是多少？

<!-- 回答 -->

### Q34：slime 里 data 怎么流，Megatron 怎么结合，loss 怎么算？

<!-- 回答 -->

### Q35：VeRL / TRL / Unsloth / AReaL / slime 你会选哪个？

<!-- 回答 -->

## 参考资料

- [知乎：大模型强化学习面经](https://zhuanlan.zhihu.com/p/2046740446353811230)

<!-- 本文配图放在 assets/rl-interview-qa/ 下，用相对路径引用。 -->
