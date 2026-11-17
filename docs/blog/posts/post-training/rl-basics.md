---
title: 强化学习基础之从零到GRPO
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

### 价值函数与最优策略

前面把 RL 的目标写成最大化 $\mathbb{E}_{\tau\sim\pi}[G_0]$，这只是一个从初始状态出发的标量，没法衡量策略在各个状态下的好坏。因此在 MDP 中引入状态价值函数，表示从状态 $s$ 出发按策略 $\pi$ 行动的期望回报：

$$
V_\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s]
$$

为了进一步比较同一状态下不同动作的好坏，再定义动作价值函数，表示在 $s$ 先执行 $a$、之后按 $\pi$ 行动的期望回报：

$$
Q_\pi(s,a) = \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a]
$$

两者的关系是 $V_\pi(s) = \sum_a \pi(a\mid s)\,Q_\pi(s,a)$，即 $V$ 是 $Q$ 在策略下的平均。$A_\pi(s,a) = Q_\pi(s,a) - V_\pi(s)$ 称为优势函数（advantage），衡量动作 $a$ 比平均水平好多少，后面策略梯度会用到。

有了价值函数，就可以比较策略的优劣：若对所有状态 $s\in S$ 都有 $V_\pi(s) \ge V_{\pi'}(s)$，就称策略 $\pi$ 不差于 $\pi'$。在有限 MDP 中，一定存在至少一个策略不差于所有其他策略，这就是我们要找的最优策略 $\pi^*$。所有最优策略共享相同的最优状态价值函数和最优动作价值函数：

$$
V^*(s) = \max_\pi V_\pi(s), \qquad Q^*(s,a) = \max_\pi Q_\pi(s,a)
$$

一旦知道了 $Q^*$，每个状态直接取贪心动作就能得到最优策略：$\pi^*(s) = \arg\max_a Q^*(s,a)$。

### 策略评估与策略提升

目标明确了是找 $\pi^*$，但直接在所有策略里穷举 $\max_\pi$ 是不现实的。更自然的思路是迭代：先拿一个策略 $\pi$，算出它的价值（策略评估，policy evaluation），再根据评估结果改进它（策略提升，policy improvement），反复交替直到收敛。

有了 $Q_\pi$，改进的方式很直接：每个状态都选当前看来最好的动作

$$
\pi'(s) = \arg\max_a Q_\pi(s,a)
$$

策略提升定理保证这样做不会变差：若对所有 $s$ 有 $Q_\pi(s,\pi'(s)) \ge V_\pi(s)$，则 $V_{\pi'}(s) \ge V_\pi(s)$。贪心策略显然满足条件，因为 $\max_a Q_\pi(s,a)$ 不小于按 $\pi$ 的平均 $V_\pi(s)$。

<figure class="rl-roadmap-figure">
--8<-- "docs/blog/posts/post-training/assets/rl-basics/policy-improvement.svg"
<figcaption>价值评估指导策略提升：交替做策略评估与贪心策略提升，逐步收敛至最优的 V* 和 π*。</figcaption>
</figure>

不断交替进行价值评估与贪心策略提升，最终就能逼近最优的 $V^*$ 和 $\pi^*$。所以现在的问题归结为：如何求解出 $V_\pi$ 和 $Q_\pi$，有了它们就可以直接用来决策。

### 贝尔曼期望方程：怎么算价值函数

按定义直接算，要对所有可能的轨迹求期望。但回报本身有递推结构 $G_t = R_{t+1} + \gamma G_{t+1}$，再加上马尔可夫性质（未来只依赖 $S_{t+1}$），就能把价值写成"一步奖励 + 下一状态价值"。用 backup 图表示就是往下展开一层再求平均：

<figure class="rl-roadmap-figure">
--8<-- "docs/blog/posts/post-training/assets/rl-basics/bellman-backup.svg"
<figcaption>空心圆是状态，实心点是动作。V 先按 π 对动作平均，再按 P 对下一状态平均；Q 反过来。</figcaption>
</figure>

$$
V_\pi(s) = \sum_a \pi(a\mid s)\Big[R(s,a) + \gamma \sum_{s'} P_{sa}(s')\,V_\pi(s')\Big]
$$

$$
Q_\pi(s,a) = R(s,a) + \gamma \sum_{s'} P_{sa}(s') \sum_{a'} \pi(a'\mid s')\,Q_\pi(s',a')
$$

这就是贝尔曼期望方程。它把对整条轨迹的期望变成了状态之间的一步递推关系，有限状态下就是一个线性方程组，策略评估就是在解这个方程。

### 贝尔曼最优方程

当策略收敛到最优策略 $\pi^*$ 时，策略本身已经是贪心的，期望方程中对 $\pi$ 的平均直接变成取 max：

$$
V^*(s) = \max_a \Big[R(s,a) + \gamma \sum_{s'} P_{sa}(s')\,V^*(s')\Big]
$$

$$
Q^*(s,a) = R(s,a) + \gamma \sum_{s'} P_{sa}(s') \max_{a'} Q^*(s',a')
$$

因为有 max，这个方程是非线性的，没有闭式解，只能迭代求解。在 $P_{sa}$ 和 $R$ 已知时，可以用下一节的动态规划求解。

## 动态规划：策略迭代与价值迭代

回到前面留下的问题：怎么算出 $V_\pi$。当当前策略 $\pi$ 已知，并且环境模型 $P_{sa}$ 和 $R$ 也已知（model-based）时，贝尔曼期望方程里的所有系数都是已知的，不需要和环境交互。虽然可以直接当线性方程组解，但矩阵求逆代价太高（$O(|S|^3)$），动态规划（Dynamic Programming, DP）的做法是直接把贝尔曼方程当成更新式，从任意初值出发反复迭代直到收敛出 $V_\pi$。

在此基础上结合策略提升，就得到两种求解最优策略的方法。

### 策略迭代

策略迭代就是把前面"评估 → 提升"的循环直接落地：

1. 策略评估：固定 $\pi$，用贝尔曼期望方程反复更新所有状态，直到 $V_k$ 收敛到 $V_\pi$

    $$
    V_{k+1}(s) \leftarrow \sum_a \pi(a\mid s)\Big[R(s,a) + \gamma \sum_{s'} P_{sa}(s')\,V_k(s')\Big]
    $$

2. 策略提升：用 $V_\pi$ 算出 $Q_\pi(s,a) = R(s,a) + \gamma \sum_{s'} P_{sa}(s')V_\pi(s')$，再贪心地取 $\pi'(s) = \arg\max_a Q_\pi(s,a)$

重复这两步，直到策略不再变化。有限 MDP 中策略数量有限，而每次提升都不会变差，所以一定会收敛到 $\pi^*$。

### 价值迭代

策略迭代每一轮都要把评估跑到收敛，代价很大。实际上评估没必要算准，只更新一步就提升策略也可以收敛。把这两步合在一起，就是直接迭代贝尔曼最优方程：

$$
V_{k+1}(s) \leftarrow \max_a \Big[R(s,a) + \gamma \sum_{s'} P_{sa}(s')\,V_k(s')\Big]
$$

收敛得到 $V^*$ 后，再做一次贪心就能得到 $\pi^*$。

DP 的问题在于强依赖环境模型。大多数实际场景中根本不知道 $P_{sa}$，贝尔曼方程里对下一状态的期望 $\sum_{s'}P_{sa}(s')[\cdot]$ 也就没法算；而且每次都要遍历全部状态，状态空间一大就做不了。

## 蒙特卡洛与时序差分

不知道 $P_{sa}$ 时（model-free），只能让 Agent 按策略和环境交互，用采样出的数据去近似期望。思路仍然是"评估 → 提升"，只是两步都要改成基于样本的版本。

### 策略评估：用样本估计价值

蒙特卡洛（Monte Carlo, MC）直接从定义 $V_\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s]$ 出发，期望用样本平均代替。按 $\pi$ 跑完一整条轨迹，得到真实回报 $G_t$，用它更新 $V(S_t)$：

$$
V(S_t) \leftarrow V(S_t) + \alpha\big(G_t - V(S_t)\big)
$$

其中 $\alpha$ 为学习率。$G_t$ 是 $V_\pi(S_t)$ 的无偏估计，但它累积了整条轨迹上所有动作和转移的随机性，方差很大，而且必须等 episode 结束才能更新。

时序差分（Temporal Difference, TD）结合了 MC 与 DP 的思想：像 MC 一样采样，不需要模型；像 DP 一样利用贝尔曼方程做一步 bootstrapping。只交互一步得到 $(R_{t+1}, S_{t+1})$，后面的回报用当前估计 $V(S_{t+1})$ 代替：

$$
V(S_t) \leftarrow V(S_t) + \alpha\big(R_{t+1} + \gamma V(S_{t+1}) - V(S_t)\big)
$$

其中 $R_{t+1} + \gamma V(S_{t+1})$ 称为 TD target，$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ 称为 TD error。由于 $V(S_{t+1})$ 本身是估计值，TD target 是有偏的；但它只包含一步的随机性，方差比 MC 小得多，并且每走一步就能在线更新。

两者之间还可以折中：往后看 $n$ 步真实奖励再接 $\gamma^n V(S_{t+n})$，就是 $n$-step TD；对不同 $n$ 的回报做指数加权平均，就是 TD($\lambda$)。后面 Actor-Critic 中的 GAE 也是同样的思路。

### 策略提升：从数据中改进策略

有了样本估计的价值函数，下一步是提升策略。但 model-free 下只学 $V$ 是不够的：贪心提升需要 $\arg\max_a \big[R(s,a) + \gamma\sum_{s'}P_{sa}(s')V(s')\big]$，这里又用到了未知的 $P_{sa}$。所以要直接估计 $Q(s,a)$，提升时只需要 $\pi'(s) = \arg\max_a Q(s,a)$，不再依赖模型。

另一个问题是探索。数据是策略自己采出来的，如果每次都贪心选动作，没被选过的动作就永远没有样本，$Q$ 也永远估不准。常用的做法是 $\varepsilon$-greedy：以 $1-\varepsilon$ 的概率选 $\arg\max_a Q(s,a)$，以 $\varepsilon$ 的概率随机选一个动作。

把 TD 的思想用到 $Q$ 上，每交互一步就更新一次 $Q$，并用 $\varepsilon$-greedy 采集下一步数据，就得到两个经典算法。

SARSA 用实际采样到的下一个动作 $A_{t+1}$ 构造 TD target，更新所需的五元组 $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$ 也是名字的由来：

$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha\big(R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)\big)
$$

它对应贝尔曼期望方程，估计的是当前 $\varepsilon$-greedy 策略自身的 $Q_\pi$，采样和学习的是同一个策略，属于 on-policy。

Q-learning 不管下一步实际选了什么动作，直接用 $\max_{a'}$ 构造 TD target：

$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha\big(R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t)\big)
$$

它对应贝尔曼最优方程，直接逼近 $Q^*$。采样用的是 $\varepsilon$-greedy 策略，学习的却是贪心策略，属于 off-policy，因此也可以用其他策略或历史数据来学习。

当状态空间很大时（比如 Atari 的屏幕像素），$Q$ 没法用表格存下，就需要用神经网络 $Q_\theta(s,a)$ 来近似，这就是 DQN。

## 策略梯度

### 直接优化策略的目标

前面的方法都是先学 $Q(s,a)$，再用 $\arg\max_a Q(s,a)$ 间接得到策略。这一步在两种情况下会出问题：动作空间连续或很大时（比如 LLM 的词表），$\max_a$ 很难算；$Q$ 有一点误差，argmax 选出的动作就可能突变，策略不稳定。

既然最终要的是策略，更直接的做法是把策略本身参数化为 $\pi_\theta(a\mid s)$，直接最大化期望轨迹回报 $J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)}[G(\tau)]$，其中 $G(\tau) = \sum_{t=0}^{T-1} R(s_t, a_t)$。整个优化过程就是一个“采样轨迹 → 估计回报 → 沿梯度 $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$ 改进策略”的循环：

<figure class="rl-roadmap-figure">
--8<-- "docs/blog/posts/post-training/assets/rl-basics/pg-loop.svg"
<figcaption>直接优化策略的目标与三步循环：运行策略采样轨迹、估计期望回报、沿梯度上升方向更新参数 θ。</figcaption>
</figure>

### 怎么算梯度

上面的循环里前两步采样算回报很直接，核心是怎么算第三步的 $\nabla_\theta J(\theta)$。把期望写成积分 $J(\theta) = \int p_\theta(\tau)\,G(\tau)\,d\tau$ 直接求导会遇到两个卡点：一是 $\int \nabla_\theta p_\theta(\tau)\,G(\tau)\,d\tau$ 不再是期望形式，没法用采样来估计；二是轨迹生成概率 $p_\theta(\tau) = p(s_0)\prod_{t=0}^{T-1} \pi_\theta(a_t\mid s_t)\,P(s_{t+1}\mid s_t,a_t)$ 里连乘了未知的环境转移概率 $P$。

第一步，利用对数导数恒等式 $\nabla_\theta p_\theta(\tau) = p_\theta(\tau)\,\nabla_\theta \log p_\theta(\tau)$ 替换积分里的导数项，外面重新凑出了密度 $p_\theta(\tau)$，把积分变回关于轨迹分布的期望：

<figure class="rl-roadmap-figure">
--8<-- "docs/blog/posts/post-training/assets/rl-basics/pg-diff-1.svg"
<figcaption>第一步：利用对数导数恒等式（蓝线换橙线），把分布导数的积分变回轨迹期望。</figcaption>
</figure>

第二步，对轨迹概率 $p_\theta(\tau)$ 两侧取 log，概率连乘变成求和。代入 $\nabla_\theta \log p_\theta(\tau)$ 求导时，与 $\theta$ 无关的初始分布 $\log p(s_0)$ 和未知的环境转移 $\log P(s_{t+1}\mid s_t, a_t)$ 直接归零消掉，只剩下策略自己的对数梯度之和：

<figure class="rl-roadmap-figure">
--8<-- "docs/blog/posts/post-training/assets/rl-basics/pg-diff-2.svg"
<figcaption>第二步：两侧取 log 后，初始分布与环境转移项对 θ 求导为 0（红线划去），得到无模型策略梯度。</figcaption>
</figure>

### REINFORCE 算法与对比 SFT

把上面推出来的期望梯度换成 $N$ 条采样轨迹的均值，代回开头的三步循环，就得到最基础的 REINFORCE 算法：

1. 按当前策略 $\pi_\theta(a_t\mid s_t)$ 与环境交互，采样 $N$ 条轨迹 $\{\tau^{(i)}\}_{i=1}^N$
2. 用样本平均估计策略梯度：$\nabla_\theta J(\theta) \approx \frac{1}{N}\sum_{i=1}^N \big(\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t^{(i)}\mid s_t^{(i)})\big)\big(\sum_{t=0}^{T-1} R(s_t^{(i)}, a_t^{(i)})\big)$
3. 沿梯度上升更新参数：$\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$

怎么直观理解这个式子？把它和普通 SFT（最大似然估计，Maximum Likelihood）的梯度放在一起对比：

$$
\begin{aligned}
\text{Policy Gradient:} \quad \nabla_\theta J(\theta) &\approx \frac{1}{N}\sum_{i=1}^{N}\left(\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t^{(i)} \mid s_t^{(i)})\right)\!\left(\sum_{t=0}^{T-1} R(s_t^{(i)}, a_t^{(i)})\right) \\
\text{Maximum Likelihood (SFT):} \quad \nabla_\theta J_{\text{ML}}(\theta) &\approx \frac{1}{N}\sum_{i=1}^{N}\left(\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t^{(i)} \mid s_t^{(i)})\right)
\end{aligned}
$$

两式的唯一区别就是后面多乘了一个轨迹总回报 $G(\tau^{(i)}) = \sum_{t=0}^{T-1} R(s_t^{(i)}, a_t^{(i)})$。普通 SFT 的数据来自现成的专家示范，默认每条样本都是对的（权重恒为 $+1$），无差别推高轨迹上每一步动作的对数概率；而 REINFORCE 没有标准答案，数据靠模型自己采样试错（trial-and-error），本质上就是按回报加权的 SFT：高回报轨迹推高概率，负回报轨迹反向压低概率。放到自回归 LLM 上，把上下文 $(x, y_{<t})$ 看作状态 $s_t$、第 $t$ 个 token $y_t$ 看作动作 $a_t$，对输出 logits 做一次 `log_softmax` 就拿到 $\log \pi_\theta(a_t\mid s_t)$，训练目标就是给逐 token 的交叉熵 loss 乘上回报权重。

### 朴素策略梯度的两个问题

从“按回报加权的 SFT”这个视角看，上面的基础 REINFORCE 循环有两个明显问题，后面的改进都是围绕这两点展开的：

<figure class="rl-roadmap-figure">
--8<-- "docs/blog/posts/post-training/assets/rl-basics/pg-roadmap.svg"
<figcaption>朴素策略梯度的循环，以及沿两个问题展开的改进路线。</figcaption>
</figure>

第一个问题是方差大。梯度估计虽然无偏，但每个动作的权重都是整条轨迹的回报 $G(\tau)$：

- 一条轨迹上每一步的动作采样和状态转移都有随机性，全部累加到 $G(\tau)$ 里，而采样的轨迹数 $N$ 又有限
- $t$ 时刻之前拿到的奖励和动作 $a_t$ 无关，却也算进了它的权重
- 权重是 $G(\tau)$ 的绝对值，如果奖励都是正的，差动作乘上正的回报依然像 SFT 一样被推高概率，只是推高得少一点

方差大意味着梯度方向抖动严重。学习率大了，一批坏样本就可能把策略推坏，坏策略又采出更差的数据；学习率小了，收敛又很慢。

第二个问题是数据只能用一次。普通 SFT 的静态数据集可以反复训多个 epoch；而这里的期望要求轨迹来自当前的 $\pi_\theta$，$\theta$ 更新一步之后数据就不再服从新策略的分布了，只能扔掉重采。

### 降方差：reward-to-go 与 baseline

针对方差，有两个不引入偏差的改法。

一是利用因果性。$a_t$ 影响不了之前的奖励，所以把权重从整条轨迹总回报 $G(\tau)$ 收紧为 $t$ 之后的未来回报 $G_t = \sum_{k=t}^{T-1} R(s_k, a_k)$（即 reward-to-go）。

二是减去一个只和状态有关的 baseline $b(s_t)$，权重变成 $G_t - b(s_t)$。这样做不改变梯度的期望，因为

$$
\mathbb{E}_{a_t\sim\pi_\theta}\big[\nabla_\theta \log \pi_\theta(a_t\mid s_t)\,b(s_t)\big] = b(s_t)\,\nabla_\theta \sum_{a_t}\pi_\theta(a_t\mid s_t) = b(s_t)\,\nabla_\theta 1 = 0
$$

最自然的选择是 $b(s_t) = V_{\pi_\theta}(s_t)$。又因为 $\mathbb{E}[G_t\mid s_t,a_t] = Q_{\pi_\theta}(s_t,a_t)$，权重在期望上就是前面定义的优势函数 $A_{\pi_\theta} = Q_{\pi_\theta} - V_{\pi_\theta}$：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, A_{\pi_\theta}(s_t, a_t)\right]
$$

比平均水平好的动作（$A>0$）提高概率，比平均水平差的（$A<0$）降低概率。

### 复用数据：重要性采样

针对数据只能用一次，可以用重要性采样 $\mathbb{E}_{x\sim p}[f(x)] = \mathbb{E}_{x\sim q}\big[\tfrac{p(x)}{q(x)}f(x)\big]$ 修正分布差异，用旧策略 $\pi_{\theta_{\text{old}}}$ 采的数据来更新 $\pi_\theta$。在两者足够接近时，只修正当前动作的概率比，得到代理目标：

$$
J^{\text{IS}}(\theta) = \mathbb{E}_{(s_t, a_t) \sim \pi_{\theta_{\text{old}}}}\!\left[\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}\, A_{\pi_{\theta_{\text{old}}}}(s_t, a_t)\right]
$$

到这里还剩两个问题：一是优势函数 $A$ 怎么估，如果还用 MC 回报，方差依然很大，下一节的 Actor-Critic 用 TD 学一个价值函数来估；二是 $\pi_\theta$ 离 $\pi_{\theta_{\text{old}}}$ 太远时概率比会很大，近似失效，策略容易崩，这需要限制每次更新的幅度，就是后面的 TRPO 和 PPO。

## Actor-Critic

## 信任域方法：TRPO 与 PPO

## 从 RL 到 LLM 后训练

<!-- RLHF、GRPO 等方法与经典 RL 的对应关系。 -->

## 总结

## 参考资料

- [David Silver：UCL Course on Reinforcement Learning](https://davidstarsilver.wordpress.com/teaching/)
- [动手学强化学习](https://hrl.boyuai.com/)

<!-- 本文配图放在 assets/rl-basics/ 下，用相对路径引用。 -->
