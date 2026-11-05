---
title: "Uni-Agent: Train Long-Horizon Agents at Scale"
description: 从大规模 Agentic RL 的挑战出发，理解 Uni-Agent 与 verl 的分工，以及统一任务接口和 Agent Gateway 的目标。
date: 2026-11-04
authors:
  - taoger
slug: uni-agent
draft: false
categories:
  - Agent
  - 训练系统
topics:
  - agent
tags:
  - Uni-Agent
  - verl
  - Agentic RL
  - 强化学习
---

# Uni-Agent: Train Long-Horizon Agents at Scale

训练一个能够完成实际任务的 Agent，需要让模型反复调用工具、观察环境、调整行动，最后根据任务结果学习。[Uni-Agent](https://github.com/verl-project/uni-agent) 建立在 verl 之上，将 Agent 执行、任务环境、轨迹采集和强化学习连接起来，目标是让长程 Agent 能够在统一框架中规模化训练。

<!-- more -->

## 为什么需要 Uni-Agent？

以修复代码为例，一条训练样本可能包含完整的执行过程：

```text
阅读 issue → 搜索代码 → 修改文件 → 运行测试 → 根据报错继续修改
                                                   ↓
                                      验证结果 → 奖励 → 模型更新
```

这个过程包含多轮模型调用，还依赖持续存在的文件系统、工具进程和任务状态。并发运行成百上千个任务后，训练系统需要同时处理环境资源、长上下文计算、执行故障以及训练数据的正确性。

![大规模 Agentic RL 的五类挑战及应对方向](assets/uni-agent/challenges.svg)

*根据本文讨论的演示截图复刻。右侧为应对方向，不表示所有能力都已在 Uni-Agent 中完整实现。*

这些挑战可以分成五个层面：

| 挑战 | 为什么会出现 | 需要解决什么 |
| --- | --- | --- |
| 大规模沙箱 | 每个任务需要独立的文件、依赖和进程 | 环境启动、资源隔离、超时处理与回收 |
| 长上下文推理 | 工具结果不断累积，任务可能持续上百轮 | KV cache 复用、负载均衡和长尾等待 |
| 长上下文训练 | 长轨迹的激活和反向传播开销很大 | 并行计算、重计算、激活卸载等 |
| Harness 接入 RL | Agent 会压缩历史、重试、分支或调用子 Agent | 准确还原模型采样轨迹并区分上下文与可训练输出 |
| 有效训练 | 最终奖励稀疏，长轨迹难以分配信用 | 探索、奖励设计、策略滞后与优化目标 |

例如，Agent 等待测试时暂时不调用模型，但下一轮仍需要之前的上下文。保留所有会话的 KV cache 会占用大量显存；驱逐缓存后，又要付出重新 prefill 的成本。因此，扩大吞吐需要同时考虑推理缓存、工具等待和环境资源。[官方技术介绍](https://verl-project.github.io/posts/2026-09-23-uni-agent-v0-1-0/)

### 已经有 verl，为什么还需要一层框架？

verl 已经提供多轮 rollout 接口，可以通过自定义 `AgentLoopBase` 调用模型、工具和环境，再返回训练所需的 token 与 mask。它具备训练 Agent 的基础能力。[verl Agent Loop 文档](https://verl.readthedocs.io/en/latest/advance/agent_loop.html)

Uni-Agent 将 Agent 运行和训练接入所需的公共工程抽出来：任务如何启动，工具在哪里执行，沙箱怎样释放，现有 harness 怎样连接模型服务，以及运行过程如何形成训练轨迹。开发者可以复用这些组件，减少每个任务重复搭建基础设施的工作。

可以把两者的分工理解为：**verl 提供模型优化和分布式训练底座，Uni-Agent 在其上组织实际任务的交互过程，并把执行结果交付给训练系统。**

## 项目目标：All in One Stack

![Uni-Agent 的统一任务与统一 Gateway 目标](assets/uni-agent/goals.svg)

*根据本文讨论的演示截图复刻。Shell、Browser、Desktop 代表统一框架面向的任务方向；图中的 harness 名称是接入示例，具体支持范围以项目当前文档为准。*

这张图从研究与应用两个角度解释了“统一”：研究侧希望用同一组抽象表达不同任务，应用侧希望让已有 Agent harness 接入同一套训练基础设施。

### 统一任务和交互环境

代码 Agent 使用 Shell，网页 Agent 操作 Browser，桌面 Agent 则需要截图、鼠标和键盘。它们的动作与观察不同，但都包含任务初始化、模型决策、环境交互和结果验证。

Uni-Agent 将这些职责拆成可组合的接口：

| 组件 | 负责什么 |
| --- | --- |
| Agent | 解决任务的策略与执行循环 |
| Tool | 模型能够调用的具体动作 |
| Task | 任务生命周期、数据、验证与奖励 |
| Sandbox | 执行环境、文件和进程管理 |
| Gateway | 模型协议适配与 token 级轨迹采集 |
| Framework | 会话编排、故障隔离与训练数据交付 |

通过这些边界，替换沙箱后端不必重新编写 Agent 策略，增加任务也可以复用已有工具与运行框架。[组件职责说明](https://github.com/verl-project/uni-agent/blob/main/CONTRIBUTING.md)

统一的接口还为多模态信息和多 Agent 协作提供扩展空间。不过，不同环境仍需要自己的工具、观察表示和验证器；接口统一并不意味着所有任务已经开箱即用。

### 统一 Gateway，接入已有 Agent harness

成熟的 harness 已经包含提示词、工具、记忆、重试和上下文管理。为了训练重新实现它，成本较高，而且细节变化可能改变模型行为。

Uni-Agent 的目标是让兼容的 harness 通过熟悉的模型 API 接入 Gateway：

```text
Task / Sandbox 中的 Agent harness
                ↓ 模型 API 请求
         Uni-Agent Gateway
                ↓
      verl 管理的推理后端
                ↓
     token 轨迹与任务奖励 → RL 训练
```

这里的“黑盒接入”指保留 harness 内部执行逻辑，通过模型调用边界采集数据；被更新的仍是接入的模型参数。外部模型 API 可以用于评测，但不会因此获得对外部模型权重的训练能力。[Gateway 文档](https://github.com/verl-project/uni-agent/blob/main/docs/source/concepts/gateway-and-trajectories.md)

Gateway 的关键工作是保留真实采样的 token、mask 和 log probability，并处理上下文变化。仅保存最终对话再重新 tokenize，可能因为工具调用解析、解码与编码而偏离原始采样序列；历史压缩和分支还会使多次请求无法直接拼接。正确的轨迹记录是训练有效性的前提。[verl 对 token 一致性的说明](https://verl.readthedocs.io/en/latest/advance/agent_loop.html)

### 缩小训练与推理之间的 harness 差异

如果模型在简化循环中训练，部署时却使用另一套提示词、工具和记忆管理方式，训练收益可能无法完整迁移。复用同一套交互栈，可以减少这类差异，也便于在相同条件下评测训练前后的变化。

Uni-Agent 因此将执行、评测和训练放在可复用的运行框架中，并提供可运行配方作为实验起点。其统一框架面向的核心问题是：**让真实 Agent 的任务执行过程，能够准确、可追踪且高效地成为模型学习的数据。**

## 整体架构：从任务执行到训练轨迹

![Uni-Agent 的 Framework、Task Runner、Gateway 与工具执行架构](assets/uni-agent/architecture.svg)

*根据演示截图复刻。图中 Agent 和沙箱名称用于说明可插拔的组织方式，具体可用实现以项目当前版本为准。*

这张图分为两部分：左侧展示任务如何被分发和执行，右侧展开白盒 Agent 内部的模型交互与工具执行。右侧是局部放大，不是左侧之后的额外流水线阶段。

### 左侧：Framework 编排任务，Sandbox 承载环境

输入数据进入 Framework 后，由 FrameworkWorker 驱动任务执行。Task Runner 将样本对应的 Agent、Sandbox 和验证流程组织成一个完整 episode；多个 Worker 可以并发推进不同会话。

Agent 负责决定如何解决任务。黑盒接入可以复用 Claude Code、Mini-SWE-Agent 等已有 harness；白盒接入则允许直接定制交互循环、工具和错误处理。两种方式都需要通过 Sandbox 操作任务环境，执行命令、读写文件并保留任务状态。

Sandbox 将环境能力与后端实现分离。图中的 veFaaS、YuanRong、Modal 表示不同基础设施选择，上层尽量通过一致的接口执行操作。图中的 `Sandbox.exec()` 和 `Sandbox.mount()` 表达命令执行与文件挂载等能力；具体方法及其支持范围需要以所用后端的接口为准。

Gateway Manager 管理模型调用会话。任务运行过程中，Agent 的模型请求经过对应的 Gateway，形成可以追踪的 token 级记录；任务验证产生奖励，Framework 再组织输出并交付给训练侧。图中 Sandbox 与左侧模块之间的连线是简化示意，不能理解为工具执行结果直接就是训练轨迹。

### 右侧：模型请求和工具执行走两条路径

白盒 Agent 的 Interaction System 维护交互循环与上下文，Toolbox 提供可调用工具。一次典型循环包含两条不同的路径：

```text
模型决策：Interaction System → Agent Gateway → 模型推理后端
环境执行：Interaction System → Toolbox → Tool → Sandbox
```

模型生成工具调用后，交互系统分发动作；工具在环境中执行，再将观察结果送回上下文，发起下一轮模型调用。Stateful Shell 保留命令执行状态，Browser 和 Desktop 提供网页与桌面操作能力，MCP 则是连接外部工具的一种协议入口。这些工具的底层实现不必完全相同，也不一定都依赖相同的 Sandbox 操作。

**Gateway 负责模型调用与轨迹，Sandbox 负责环境执行与状态。** Gateway 本身不生成 token，实际生成由 vLLM、SGLang 等推理后端完成；它也不代替工具执行器。区分这两条路径，才能理解框架如何同时接入不同 harness、工具和运行环境。[Gateway 职责说明](https://github.com/verl-project/uni-agent/blob/main/docs/source/concepts/gateway-and-trajectories.md)

以代码修复为例，Worker 启动任务后，Agent 先经 Gateway 请求模型，模型提出 Shell 或编辑工具调用，工具在 Sandbox 中执行。测试结果作为观察返回 Agent；任务结束时，Task 负责验证与奖励，Gateway 提供采样轨迹，Framework 将二者关联起来。这就把一次实际任务执行连接到了 RL 更新所需的数据。

## Gateway 设计：Message In，Training Trajectories Out

![Gateway 在 message 接口和 token 推理接口之间组织训练轨迹](assets/uni-agent/gateway-design.svg)

*根据演示截图复刻。TrajectoryBuffer 表示轨迹缓冲的概念，具体类名与接口以项目版本为准。*

Gateway 的核心目的之一，是将 Agent 在线交互中产生的模型调用组织成训练侧可以消费的 token 级轨迹。它连接两种接口：**面向 Agent 使用 messages，面向推理后端使用 token IDs。**

### 两侧接口：兼容现有 Agent，保留真实采样记录

Agent 侧继续使用熟悉的 OpenAI 或 Anthropic 兼容接口。图中给出的会话路径是 `/sessions/{id}/v1/chat/completions` 和 `/sessions/{id}/v1/messages`，请求仍然包含对话内容、工具定义与采样参数。

推理侧则使用 token-in/token-out 的生成接口，概念上可以写成：

```python
result = backend.generate(prompt_ids, sampling_params)
```

Gateway 将消息编码为模型输入，并保留推理后端返回的实际采样 token 和相关概率，再将结果转换为 Agent 能使用的回复。这样，harness 可以继续操作 messages，训练侧也能获得准确的采样记录，无需仅依赖最终对话文本重新构造输出 token。[Gateway 文档](https://github.com/verl-project/uni-agent/blob/main/docs/source/concepts/gateway-and-trajectories.md)

### Session：一次任务可以包含多条轨迹

一次任务执行通常对应一个 Gateway session，其中包含多次模型调用。对于不断追加的对话，新的调用可以延续已有轨迹；如果 harness 压缩、裁剪或改写历史，新的上下文不一定兼容原来的 token 序列，就需要按规则拆分或回退处理。

因此，**一个 session 不一定只产生一条轨迹**。Session 关联同一次任务的调用，轨迹则描述可以一致组织的 token 序列；二者不是同一个粒度。

### 轨迹缓冲：记录训练真正需要的信息

| 字段 | 用途 |
| --- | --- |
| `prompt_ids` | 保存该条轨迹的输入 token |
| `response_ids` | 保存后续序列，可能同时包含模型输出和追加的工具上下文 |
| `response_mask` | 标明哪些位置属于可训练的模型输出，工具返回等上下文通常不参与策略损失 |
| `logprobs` | 保存实际采样输出的行为策略 log probability，供 RL 目标使用 |

这些字段表达的是概念上的训练记录，最终格式由具体的数据契约决定。任务结束后，Framework 将轨迹与 Task 验证得到的奖励关联，通过 TransferQueue 等数据通道交给 verl。**Gateway 负责模型协议和轨迹，Task 负责奖励，verl 负责参数更新。**

## 参考资料

- [Uni-Agent 项目仓库](https://github.com/verl-project/uni-agent)
- [Uni-Agent 官方技术介绍](https://verl-project.github.io/posts/2026-09-23-uni-agent-v0-1-0/)
- [Uni-Agent Gateway 与轨迹文档](https://github.com/verl-project/uni-agent/blob/main/docs/source/concepts/gateway-and-trajectories.md)
- [verl Agent Loop 文档](https://verl.readthedocs.io/en/latest/advance/agent_loop.html)
