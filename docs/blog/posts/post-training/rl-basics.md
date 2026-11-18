---
title: 强化学习基础之从随机变量到GRPO
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

# 强化学习基础之从随机变量到GRPO

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

上面的循环里前两步采样算回报很直接，核心是怎么求第三步的 $\nabla_\theta J(\theta)$。推导只靠两个关键点：一是利用对数变换技巧（$\nabla_\theta p_\theta(\tau) = p_\theta(\tau)\nabla_\theta \log p_\theta(\tau)$）把对分布的求导还原成轨迹期望；二是利用初始分布和环境转移概率都与策略参数 $\theta$ 无关，取 log 把连乘拆成求和后，未知的环境项求导直接归零：

<figure class="rl-roadmap-figure">
--8<-- "docs/blog/posts/post-training/assets/rl-basics/pg-diff-1.svg"
<figcaption>第一步：利用对数导数恒等式（蓝线换橙线），把分布导数的积分变回轨迹期望。</figcaption>
</figure>

<figure class="rl-roadmap-figure">
--8<-- "docs/blog/posts/post-training/assets/rl-basics/pg-diff-2.svg"
<figcaption>第二步：两侧取 log 后，初始分布与环境转移项对 θ 求导为 0（红线划去）。</figcaption>
</figure>

由此得到不依赖环境模型的策略梯度公式：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)}\!\left[\left(\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\right)\!\left(\sum_{t=0}^{T-1} R(s_t, a_t)\right)\right]
$$

### REINFORCE 算法与对比 SFT

对上面的期望梯度直接采用蒙特卡洛（MC）采样来估计——即按当前策略 $\pi_\theta$ 采样 $N$ 条完整轨迹 $\{\tau^{(i)}\}_{i=1}^N$，用样本均值代替期望，再代回开头的三步循环更新 $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$，就得到最基础的 REINFORCE 算法。

怎么直观理解这个梯度估计式？把它和普通 SFT（最大似然估计，Maximum Likelihood）的梯度摆在一起对比：

<figure class="rl-roadmap-figure">
--8<-- "docs/blog/posts/post-training/assets/rl-basics/pg-vs-sft.svg"
<figcaption>REINFORCE 策略梯度与 SFT 最大似然梯度对比：唯一区别是右侧多乘了轨迹总回报权重。</figcaption>
</figure>

两式的唯一区别就是后面多乘了一个轨迹总回报 $G(\tau^{(i)}) = \sum_{t=0}^{T-1} R(s_t^{(i)}, a_t^{(i)})$。普通 SFT 的数据来自现成的专家示范，默认每条样本都是对的（权重恒为 $+1$），无差别推高轨迹上每一步动作的对数概率；而 REINFORCE 没有标准答案，数据靠模型自己采样试错（trial-and-error），本质上就是按回报加权的 SFT：高回报轨迹推高概率，负回报轨迹反向压低概率。

把这个式子落到自回归 LLM 上，左边的对数似然梯度求和与右边的累积奖励求和各自有着非常具体的对应关系：

<figure class="rl-roadmap-figure">
--8<-- "docs/blog/posts/post-training/assets/rl-basics/pg-llm-mapping.svg"
<figcaption>策略梯度在自回归 LLM 中的逐项对应：左项是生成回答的逐 token 对数似然梯度（同 SFT 交叉熵），右项是句末由 Reward Model 或验证器算出的整条回答回报权重。</figcaption>
</figure>

### 高级策略梯度

基础 REINFORCE 给出了直接优化策略的完整闭环，但直接拿来训练会遇到两个突出瓶颈：一是**梯度极不稳定（高方差）**，二是**学习效率太低（样本用一次就废）**。后续的高级策略梯度算法，主要就是从稳定梯度和提升样本效率这两个方向展开改进的：

<figure class="rl-roadmap-figure">
--8<-- "docs/blog/posts/post-training/assets/rl-basics/pg-roadmap.svg"
<figcaption>高级策略梯度的两个改进方向：左线通过因果性与 Baseline 缓解梯度不稳定，右线通过 Off-Policy 重要性采样与步长约束提升学习效率。</figcaption>
</figure>

#### 缓解梯度不稳定：reward-to-go 与 baseline

基础 REINFORCE 的梯度之所以不稳定，是因为它给每个动作乘的权重都是整条轨迹的总回报 $G(\tau)$，这带来三层噪声：

- 一条轨迹上每步动作采样和状态转移都有随机性，全部累加进单一标量 $G(\tau)$ 里，而在线采样的轨迹数 $N$ 又有限
- $t$ 时刻之前拿到的奖励和当前动作 $a_t$ 无关，却也算进了它的权重
- 权重直接取决于 $G(\tau)$ 的绝对值，如果奖励恒为正数，差动作乘上正回报依然像 SFT 一样被推高概率，只是推高得少一点

方差过大会让每次估计出的梯度方向剧烈抖动：学习率大了，一批坏样本就可能把策略推崩，变差的策略又采出更差的数据；学习率小了，收敛又极慢。既然在线交互没法靠无限增大采样数 $N$ 来暴力平摊噪声，**降低方差的目的就是只用少量样本也能得到稳定的梯度估计**。在不引入偏差的前提下，通常分两步改进：

1. **利用因果性（reward-to-go）**：改之前外层的 $G(\tau)$ 乘进求和号，相当于每一步动作都乘同一个整条轨迹总分 $G(\tau)$；而当前动作 $a_t$ 影响不了过去的奖励，去掉 $k < t$ 的无关历史项后，第 $t$ 步就只加权从 $t$ 往后的未来回报 $G_t = \sum_{k=t}^{T-1} R(s_k, a_k)$（权重随 $t$ 变化，写进求和号内）：

    $$
    \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\cdot G(\tau) \;\longrightarrow\; \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\cdot G_t
    $$

    （注：若在单轮 LLM 中只有句末 `<eos>` 给一个结果分、中间步奖励全为 $0$ 且 $\gamma=1$，那每个位置往后加出的 $G_t$ 仍等于同一个句末总分；一旦有多轮工具交互奖励、过程奖励 PRM 或 $\gamma < 1$，每一步的 $G_t$ 就各不相同了。）
2. **减去状态基线 $b(s_t)$**：把权重改成 $G_t - b(s_t)$ 来消除奖励整体偏移带来的波动。这样做不改变梯度的期望，因为

    $$
    \mathbb{E}_{a_t\sim\pi_\theta}\big[\nabla_\theta \log \pi_\theta(a_t\mid s_t)\,b(s_t)\big] = b(s_t)\,\nabla_\theta \sum_{a_t}\pi_\theta(a_t\mid s_t) = b(s_t)\,\nabla_\theta 1 = 0
    $$

    最自然的选择是取 $b(s_t) = V_{\pi_\theta}(s_t)$。结合 $\mathbb{E}[G_t\mid s_t,a_t] = Q_{\pi_\theta}(s_t,a_t)$，权重在期望上正好变成前面定义的优势函数 $A_{\pi_\theta}(s_t,a_t) = Q_{\pi_\theta}(s_t,a_t) - V_{\pi_\theta}(s_t)$：

    $$
    \nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, A_{\pi_\theta}(s_t, a_t)\right]
    $$

此时比平均水平好的动作（$A>0$）提高概率，比平均水平差的（$A<0$）降低概率。

#### 提升学习效率：Off-Policy 与重要性采样

另一个瓶颈是学习效率。普通 SFT 的离线数据集采集一次就能反复训多个 epoch；而策略梯度里的期望 $\mathbb{E}_{\tau \sim \pi_\theta}[\cdot]$ 要求样本必须来自当前正在优化的 $\pi_\theta$（on-policy）。一旦拿数据算了一次梯度、把参数从 $\theta_{\text{old}}$ 更新到 $\theta$，策略分布就变了，刚采出来的轨迹不再服从新分布，只能扔掉重采。在交互或生成开销很大的场景（如 LLM 长序列 rollout），一批数据只更新一步太浪费。

要想复用旧策略 $\pi_{\theta_{\text{old}}}$ 采出的数据来多步更新当前的 $\pi_\theta$（即转向 Off-policy 策略梯度），核心问题就是：**样本是从旧分布 $q$ 采出来的，怎么用它去估计新分布 $p$ 下的期望？** 这正是重要性采样（Importance Sampling）解决的问题：

$$
\mathbb{E}_{x \sim p}[f(x)] = \int p(x)\,f(x)\,dx = \int q(x)\,\frac{p(x)}{q(x)}\,f(x)\,dx = \mathbb{E}_{x \sim q}\!\left[\frac{p(x)}{q(x)}\,f(x)\right]
$$

即乘上新旧分布的概率比 $\frac{p(x)}{q(x)}$（重要性权重）来修正采样分布的差异。如果直接对整条轨迹套用，概率比 $\frac{p_\theta(\tau)}{p_{\theta_{\text{old}}}(\tau)}$ 中未知的环境转移项会上下约掉，变成每步动作概率比的连乘 $\prod_{t=0}^{T-1}\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}$，但多步连乘会让方差指数爆炸。实际做法是当 $\pi_\theta$ 与 $\pi_{\theta_{\text{old}}}$ 足够接近时，近似认为两者的状态访问分布一致，只对当前动作做单步修正：

$$
\nabla_\theta J(\theta) \approx \mathbb{E}_{(s_t, a_t) \sim \pi_{\theta_{\text{old}}}}\!\left[\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}\,\nabla_\theta \log \pi_\theta(a_t \mid s_t)\, A_{\pi_{\theta_{\text{old}}}}(s_t, a_t)\right]
$$

再利用 $\frac{\pi_\theta}{\pi_{\theta_{\text{old}}}}\nabla_\theta \log \pi_\theta = \frac{\nabla_\theta \pi_\theta}{\pi_{\theta_{\text{old}}}} = \nabla_\theta\!\left(\frac{\pi_\theta}{\pi_{\theta_{\text{old}}}}\right)$ 把求导提出来，对应的代理优化目标（surrogate objective）就可以写成（注意 $\log$ 已经被吸收进了比值里）：

$$
J^{\text{IS}}(\theta) = \mathbb{E}_{(s_t, a_t) \sim \pi_{\theta_{\text{old}}}}\!\left[\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}\, A_{\pi_{\theta_{\text{old}}}}(s_t, a_t)\right]
$$

有了 $J^{\text{IS}}(\theta)$，同一批 $\pi_{\theta_{\text{old}}}$ 采出的数据就能反复做多步梯度更新。顺着上面两个方向往下走，还剩最后两个关键拼图：一是优势函数 $A$ 如果仍用 MC 回报 $G_t$ 来算，长轨迹下方差还是偏大，下一节的 Actor-Critic 用 TD 学一个价值网络来低方差估计它；二是多步更新一旦让 $\pi_\theta$ 偏离 $\pi_{\theta_{\text{old}}}$ 太远，重要性权重就会剧烈失控导致策略崩溃，后面的 TRPO 和 PPO 就是通过限制每次更新的幅度来解决这一点的。

## Actor-Critic

### 从单次随机采样到价值函数拟合

回到上一节带 baseline 的策略梯度：在实际采样出一条轨迹后，如果还像 REINFORCE 那样用单条轨迹算出的 $G_t^{(i)} - V(s_t^{(i)})$ 当作优势估计，方差其实依然很大。从随机变量的视角看原因很直接——从 $(s_t, a_t)$ 往后走，由于策略采样和环境转移都是随机的，未来会展开成一整片扇形的可能轨迹，未来回报 $G_t$（以及基于单样本算出的优势）本身是一个随机变量。REINFORCE 每次只沿着实际发生的那一条分支把奖励加起来，碰上运气好或运气差的分支，数值就会剧烈跳动。

我们真正需要的不是单条分支的随机样本 $G_t^{(i)}$，而是它在整片未来分支扇面上的真实期望 $Q_{\pi_\theta}(s_t, a_t) = \mathbb{E}_{\pi_\theta}[G_t \mid s_t, a_t]$，进而得到确定性的优势函数 $A_{\pi_\theta}(s_t, a_t) = Q_{\pi_\theta}(s_t, a_t) - V_{\pi_\theta}(s_t)$：

<figure class="rl-roadmap-figure">
--8<-- "docs/blog/posts/post-training/assets/rl-basics/ac-q-estimate.svg"
<figcaption>REINFORCE 只沿单条采样分支（红线）累加回报，而 Actor-Critic 用价值函数拟合整片扇面的期望价值，从而给出低方差的优势估计。</figcaption>
</figure>

Actor-Critic 的本质就是专门引入一个参数化的价值网络（Critic）去拟合期望价值，用来更准、更低方差地估计 $A_{\pi_\theta}(s_t, a_t)$，把前面两条主线（Value-based 的 TD 价值估计与 Policy-based 的策略梯度）结合起来：

- **Actor（策略网络 $\pi_\theta(a\mid s)$）**：负责与环境交互选动作，并按 Critic 给出的优势估计 $\hat{A}_t$ 更新策略参数 $\theta$
- **Critic（价值网络 $V_\phi(s)$）**：负责评估状态好坏，通过回归损失最小化价值预测误差来更新参数 $\phi$

### 用单网络 $V_\phi$ 估计优势与 GAE

表面上看算 $A_{\pi_\theta}(s_t, a_t) = Q_{\pi_\theta}(s_t, a_t) - V_{\pi_\theta}(s_t)$ 似乎要同时学 $Q$ 和 $V$ 两个网络，实际上只学一个状态价值网络 $V_\phi(s)$ 就够了。利用贝尔曼期望关系 $Q_{\pi_\theta}(s_t, a_t) = \mathbb{E}[R(s_t, a_t) + \gamma V_{\pi_\theta}(s_{t+1}) \mid s_t, a_t]$，拿交互采到的真实一步奖励加上下一状态的价值估计做 bootstrapping，就能直接得到一步优势估计：

$$
\hat{A}_t = \underbrace{R(s_t, a_t) + \gamma V_\phi(s_{t+1})}_{\approx\, Q_{\pi_\theta}(s_t, a_t)} - V_\phi(s_t) = \delta_t
$$

这个式子正好就是前面时序差分一节里的 TD error $\delta_t$。它只保留了当前一步转移的随机性，把 $s_{t+1}$ 之后整片未来分支的随机性都用确定性的函数值 $V_\phi(s_{t+1})$ 替代了，所以方差极低，而且每走一步就能在线计算，不必等整条轨迹结束。

当然，如果 Critic $V_\phi$ 还没学准，只看一步的 $\delta_t$ 会引入偏差；反过来一直展开到轨迹末尾的 MC 估计 $\hat{A}_t^{\text{MC}} = G_t - V_\phi(s_t) = \sum_{l=0}^{T-t-1} \gamma^l \delta_{t+l}$ 无偏但方差大。实际训练（如 PPO）普遍采用广义优势估计 GAE（Generalized Advantage Estimation），引入 $\lambda \in [0, 1]$ 对未来各步的 TD error 做指数加权平均：

$$
\hat{A}_t^{\text{GAE}(\gamma,\lambda)} = \sum_{l=0}^{T-t-1} (\gamma\lambda)^l \delta_{t+l}
$$

$\lambda = 0$ 时退化为一步 TD 优势 $\delta_t$（低方差、有偏），$\lambda = 1$ 时退化为 MC 优势 $G_t - V_\phi(s_t)$（无偏、高方差）；实践中常取 $\lambda \approx 0.95$ 在偏差与方差之间折中，实现时只需从轨迹末尾往前按 $\hat{A}_t = \delta_t + \gamma\lambda \hat{A}_{t+1}$ 递推一遍即可算出。Critic 网络则通过最小化均方误差 $\mathcal{L}(\phi) = \frac{1}{2}\big(V_\phi(s_t) - \hat{V}_t^{\text{target}}\big)^2$（目标值常取 $\hat{V}_t^{\text{target}} = \hat{A}_t^{\text{GAE}} + V_{\phi_{\text{old}}}(s_t)$）不断提升价值估计精度。

## 信任域方法：TRPO 与 PPO

## 从 RL 到 LLM 后训练

<!-- RLHF、GRPO 等方法与经典 RL 的对应关系。 -->

## 总结

## 参考资料

- [David Silver：UCL Course on Reinforcement Learning](https://davidstarsilver.wordpress.com/teaching/)
- [动手学强化学习](https://hrl.boyuai.com/)

<!-- 本文配图放在 assets/rl-basics/ 下，用相对路径引用。 -->
