# 解析论文与项目思路

> [https://github.com/InternRobotics/Humanoid-Goalkeeper/tree/main](https://github.com/InternRobotics/Humanoid-Goalkeeper/tree/main)
>
> 这个项目是这篇论文的/Users/fengjiacheng/shoumenmac/2510.18002v2.pdf
>
> 你事无巨细的讲讲论文的思路和项目的思路，从算法工程师的角度，但是需要异动些

> [https://github.com/InternRobotics/Humanoid-Goalkeeper/tree/main](https://github.com/InternRobotics/Humanoid-Goalkeeper/tree/main)
>
> 这个项目是这篇论文的/Users/fengjiacheng/shoumenmac/2510.18002v2.pdf
>
> 你事无巨细的讲讲论文的思路和项目的思路，从算法工程师的角度，但是需要易懂些

<details><summary>2 previous messages</summary>

> 我会把论文和代码当成两条线来拆：先还原“问题怎么建模、算法为什么这样设计”，再逐目录映射到项目里的训练、仿真、部署与数据流，最后用算法工程师熟悉的张量、损失、策略输入输出和控制循环把两者对齐。论文里的公式我也会翻译成直觉，不只做章节摘要。
>
> 本地已经有一份 `Humanoid-Goalkeeper` 项目副本，我会直接按当前代码逐层读；这样能看清 README 没写出的默认值、观测维度、奖励项和部署假设。PDF 是 54 MB，本机没有系统级 PDF 转文本工具，我会改用现有 Python PDF 库提取全文和公式附近段落，并对关键图表单独查看。
</details>

论文的核心已经明确：它不是“先识别落点，再调用六套动作”，而是单个端到端策略；六个区域主要用于训练时选择对应的任务奖励和动作判别器。这个区别很关键——代码里 actor 看不到显式区域标签，只看最近 10 帧球位置与本体状态；区域、落点、球速等真值只给 critic/辅助估计器，部署时由历史观测隐式推断。接下来我正在核对这套“非对称 actor–critic + AMP + 辅助估计”的实际实现是否完全符合论文表述。

> [https://github.com/InternRobotics/Humanoid-Goalkeeper/tree/main](https://github.com/InternRobotics/Humanoid-Goalkeeper/tree/main)
>
> 这个项目是这篇论文的/Users/fengjiacheng/shoumenmac/2510.18002v2.pdf
>
> 你事无巨细的讲讲论文的思路和项目的思路，从算法工程师的角度，但是需要易懂些

<details><summary>2 previous messages</summary>

> 我已经把 v2 论文全文、主分支代码、六份动作先验数据以及训练/导出链路逐一对齐了。下面不会只复述论文，而会明确区分三件事：论文“声称的方法”、代码“真正执行的张量与损失”、以及开源仓库“没有提供、因此不能直接复现”的真机部分。
>
> <details><summary>Explored 5 files</summary>
>
> - Read `./legged_gym/legged_gym/envs/base/legged_robot.py`
> - Read `./legged_gym/legged_gym/envs/base/legged_robot_config.py`
> - Read `./rsl_rl/rsl_rl/runners/him_on_policy_runner.py`
> - Read `./legged_gym/legged_gym/envs/base/legged_robot.py`
> - Read `./legged_gym/legged_gym/envs/g1/g1_29_config.py`
>
> </details>
</details>

我先给一个最重要的结论：

这篇论文真正的新意，并不是“用强化学习训练一个人形机器人挡球”这么简单，而是解决了一个很棘手的多模态控制问题：

> 同一个机器人、同一个初始站姿，面对落在左中、右中、左上、右上、左下、右下六个区域的球，需要在不到 1 秒内自动选择完全不同的全身动作；既要挡住球，又要像人，而且不能严格照着某条动作轨迹播放。

论文的解法可以浓缩成一句话：

> 用落球区域同时约束“应该完成什么任务”和“应该像哪一种人类动作”，再让一个统一策略从球的历史轨迹中自动判断落区、预测轨迹并输出 29 个关节目标。

我下面会从论文思路、算法设计、代码实现、训练流程、真机链路和工程局限六个层面拆开讲。分析依据是你本地的 [2510.18002v2.pdf](./2510.18002v2.pdf)、[官方仓库](https://github.com/InternRobotics/Humanoid-Goalkeeper)和 [arXiv v2](https://arxiv.org/abs/2510.18002v2)。

---

# 一、它到底在解决什么问题

## 1. 人形守门员为什么特别难

守门不是普通的机器人运动控制。

普通行走策略面对的是比较缓慢的目标，例如：

- 向前走；
- 保持某个速度；
- 转弯；
- 踢一个基本静止的球。

守门员面对的是：

- 球在 3～5 米外被踢出；
- 飞行时间约 0.4～1.0 秒；
- 落点可能横跨约 ±1.5 米；
- 高度可达到约 1.8 米；
- 机器人必须迅速判断方向；
- 全身一起动作；
- 挡完球还不能摔倒。

这里存在三个不同层面的难点。

### 难点 A：只给当前位置，无法判断球往哪飞

假设某一帧看到球在：

```text
x = 2.0m, y = 0.3m, z = 1.0m
```

只看这一帧不知道它是在：

- 向左飞；
- 向右飞；
- 上升；
- 下落；
- 接近机器人；
- 还是已经被打回去了。

所以必须看一段历史轨迹。

### 难点 B：动作是多模态的

六个区域对应的合理动作不同：

| 区域 | 典型动作 |
|---|---|
| 中高左侧 | 左手扑挡 |
| 中高右侧 | 右手扑挡 |
| 左上 | 向左跳起 |
| 右上 | 向右跳起 |
| 左下 | 向左侧步、压低身体 |
| 右下 | 向右侧步、压低身体 |

如果把所有动作混在一起训练，策略容易学成“平均动作”：

- 手伸一点；
- 身体晃一下；
- 不敢明显向左或向右；
- 动作看起来平滑，但挡不住球。

这和生成模型里的“多峰分布被平均掉”很像。

### 难点 C：人类动作不能直接逐帧跟踪

传统动作模仿常常要求：

```text
第 0.1 秒：肩关节转到 A
第 0.2 秒：肘关节转到 B
第 0.3 秒：身体移动到 C
```

但守门时，球速和落点每次都不同。

严格跟踪动作会产生冲突：

- 人类参考动作在 0.4 秒伸手；
- 这一次球在 0.3 秒就到了；
- 如果继续按参考动作播放，动作很像人，但球已经进门了。

因此作者使用 AMP——Adversarial Motion Prior，中文可以理解为“对抗式动作先验”。

它要求机器人动作“总体上像这种动作风格”，但不要求在固定时间跟踪固定姿态。

---

# 二、整套方法的直觉

整个系统可以理解为四个模块：

```text
球的历史位置 + 机器人自身状态
             │
             ▼
      历史特征编码器
        ┌────┴────┐
        ▼         ▼
   轨迹/落点估计   六区域分类
        └────┬────┘
             ▼
      统一 Actor 策略
             ▼
      29 个关节位置目标
             ▼
          PD 控制器
             ▼
      Unitree G1 执行动作
```

训练时还有两个额外老师：

```text
任务奖励：有没有接近球、挡住球、保持平衡
动作老师：当前区域的动作像不像对应的人类示范
```

所以整体目标可以理解为：

```text
总目标 =
    挡球任务做得好
  + 动作像对应区域的人类动作
  + 动作平滑
  + 关节不过限
  + 挡球后站稳
```

---

# 三、为什么要把球门划成六个区域

这是论文最核心的设计：Position-Conditioned Task-Motion Constraints。

代码中的区域定义在 [g1_29_config.py](./Humanoid-Goalkeeper/legged_gym/legged_gym/envs/g1/g1_29_config.py:24)。

六个区域本质上是：

```text
             高球区域
       region 2 | region 3
      -----------+-----------
             中球区域
       region 0 | region 1
      -----------+-----------
             低球区域
       region 4 | region 5
```

考虑坐标系左右方向后，代码映射到六份动作数据：

```text
region 0 → lefthand
region 1 → righthand
region 2 → leftjump
region 3 → rightjump
region 4 → leftstep
region 5 → rightstep
```

对应数据文件位于：

[resources/datasets/goalkeeper](./Humanoid-Goalkeeper/legged_gym/resources/datasets/goalkeeper)

包括：

```text
lefthand.pt
righthand.pt
leftjump.pt
rightjump.pt
leftstep.pt
rightstep.pt
```

## 为什么区域划分有效

假设只有一个统一动作判别器。

它看到以下动作都来自专家：

- 左扑；
- 右扑；
- 向上跳；
- 下蹲；
- 左侧步；
- 右侧步。

当球飞向左上时，只要机器人做出了这六种动作中的任意一种，统一判别器都可能认为“像人”。

但任务真正需要的是“左上球对应左跳”。

所以作者同时做两种条件化：

1. 任务奖励按区域选择正确的手、正确的移动方向；
2. AMP 按区域选择正确的专家动作判别器。

也就是：

```text
左上球：
    任务奖励要求左手接近左上落点
    动作判别器要求动作像 leftjump

右下球：
    任务奖励要求右手接近右下落点
    动作判别器要求动作像 rightstep
```

这叫 task-motion alignment：任务约束和动作约束指向同一种行为。

它比简单地给策略一个区域标签更完整，因为区域不仅影响策略输入，还决定了：

- 使用哪只手计算奖励；
- 鼓励哪个方向的身体速度；
- 选择哪个专家动作分布；
- 选择哪个 AMP 判别器。

---

# 四、任务奖励是怎么设计的

## 1. 为什么要先追预测落点，再追真实球位置

论文定义了一个动态目标点：

\[
p_{target} =
\begin{cases}
p_{land}, & \text{球还比较远}\\
p_{ball}, & \text{球已经靠近}
\end{cases}
\]

直觉是：

### 球较远时

机器人不应该傻傻追逐当前球位置。

因为球还在三米外，如果手朝三米外伸没有实际意义。机器人应该提前移动到球未来经过球门线附近的位置。

所以远处使用预测落点：

```text
当前球位置：还在远方
机器人目标：提前占据未来拦截点
```

### 球较近时

预测模型会有误差，而且最终挡球要求手和球真正接触。

所以球靠近后切换成当前真实球位置：

```text
预测只能负责大方向
近距离由实际观测负责最后修正
```

这是经典的 coarse-to-fine 思路：

```text
远处：预测和提前规划
近处：闭环跟踪和精确修正
```

代码在 [legged_robot.py](./Humanoid-Goalkeeper/legged_gym/legged_gym/envs/base/legged_robot.py:167) 中实现。当球进入大约 `x=0.1～0.5m` 的近距离区间时，`end_target` 从预测落点切换为当前球位置。

---

## 2. 手接近目标的连续奖励

论文使用 sigmoid 型距离奖励：

\[
r_t =
\left(
1-\frac{1}{1+\exp[-\sigma(d-d_{keep})]}
\right)(1+\nu^{(R)})
\]

不用纠结公式，它表达的是：

```text
手离目标很远 → 奖励接近 0
手开始靠近   → 奖励逐渐增大
手进入接触区 → 奖励接近 1
```

它比二值奖励好训练。

如果只有：

```python
reward = 1 if 手碰到球 else 0
```

那么训练初期机器人几乎永远碰不到球，PPO 不知道应该往哪里改。

连续距离奖励则能告诉策略：

```text
这一次虽然没碰到，但比上一次近了 20cm，是正确方向。
```

代码里的主要实现是 `_reward_eereach()`，位于 [legged_robot.py](./Humanoid-Goalkeeper/legged_gym/legged_gym/envs/base/legged_robot.py:1361)。

---

## 3. 区域决定使用哪只手

代码为六个区域分别计算距离：

```python
region 0 → 左手到目标的距离
region 1 → 右手到目标的距离
region 2 → 左手到目标的距离
region 3 → 右手到目标的距离
region 4 → 左手到目标的距离
region 5 → 右手到目标的距离
```

这避免了策略钻空子。

例如球在左边，如果用“离球最近的手”计算奖励，策略可能偶尔扭身用右手去碰，只要距离近也能拿奖励。区域绑定以后，动作语义更加稳定。

---

## 4. 动力学调制项：不只伸手，还要动全身

论文中的 \(\nu(R)\) 是一个区域相关的速度奖励。

它的直觉是：

- 左侧球：身体应该向左移动；
- 右侧球：身体应该向右移动；
- 高球：身体应该有向上的速度。

代码在 `_reward_eereach()` 中直接读取 pelvis 的横向或竖直速度：

```text
中球、低球区域 → 奖励正确方向的横向速度
高球区域       → 奖励向上的竖直速度
```

这非常重要。

如果只奖励“手接近球”，RL 很可能找到一种难看的局部最优：

- 脚不动；
- 腰极度扭曲；
- 手臂猛甩；
- 只覆盖身体附近的小范围。

动力学调制告诉机器人：

> 不是只有手到位就行，身体也应该朝正确方向启动。

---

## 5. 成功和挡球奖励

代码有两个容易混淆的奖励。

### `_reward_success`

当指定手距离目标小于严格阈值 `0.15m` 时触发。

它主要表达：

```text
手确实到达了拦截点。
```

### `_reward_stopball`

当球的 x 方向速度发生显著改变，而且球仍在球门前方时，认为成功挡球。

它主要表达：

```text
不是手看起来靠近了，而是球真的被改变了运动状态。
```

这一项权重非常高：

```python
stopball = 100.0
```

因为最终目标仍然是挡球，不是摆动作。

---

# 五、为什么一局要持续 3 秒

球通常 0.4～1 秒就飞完了，但 episode 长度被设置为 3 秒。

原因是：

> 只奖励挡球瞬间，会学出“把身体扔出去碰一下球，然后直接摔倒”的策略。

所以论文专门加入 post-task stability，也就是任务后的稳定性奖励。

挡球完成或球已经飞过机器人后，开始奖励：

- 身体方向恢复正常；
- pelvis 角速度较小；
- 身体线速度较小；
- 手臂、肘和手腕回到合理姿态；
- 腰部回到默认姿态；
- 双脚稳定落地。

代码对应：

```text
_reward_postorientation
_reward_postangvel
_reward_postlinvel
_reward_postupperdofpos
_reward_postwaistdofpos
_reward_successland
```

这些实现集中在 [legged_robot.py](./Humanoid-Goalkeeper/legged_gym/legged_gym/envs/base/legged_robot.py:1480)。

这体现了一个很实用的 RL 原则：

> 任务结束时的状态本身也是任务的一部分。

一个优秀的挡球策略不只是“碰到球”，还要为下一次动作留下可控状态。

---

# 六、从任意姿态继续守门是怎么训练的

论文说，它会把部分环境重置到其他正在执行动作的环境姿态。

代码里的实际实现更加具体：

```python
if continue_keep and random > 0.2:
    新环境的关节位置 = 随机另一个环境当前的关节位置
```

也就是大约 80% 的重置会从另一个环境的关节姿态开始。

但要注意，开源实现并不是把整个物理状态完整复制过去：

- 复制了关节位置；
- 关节速度被清零；
- base/root 状态重新初始化；
- base 速度另外随机化。

所以它更准确的含义是：

> 让机器人从各种“动作中间形态”开始，而不是每次都从标准站姿开始。

这样策略才能学会：

- 手还举着时继续挡下一球；
- 身体偏向一边时重新响应；
- 不必先恢复到标准姿态再启动。

这是论文连续挡球能力的重要来源。

---

# 七、动作先验 AMP 到底是什么

## 1. 专家动作数据怎么得到

论文中的数据管线是：

```text
人类守门动作视频
      ↓
GVHMR 恢复人体三维运动
      ↓
把人体骨架动作 retarget 到 Unitree G1
      ↓
得到 G1 的关节角序列
      ↓
按六个区域保存
```

代码只提供了处理后的六份 `.pt` 数据，没有提供：

- 原始 RGB 视频；
- GVHMR 调用脚本；
- 人体到 G1 的 retarget 脚本；
- 数据清理与人工校准流程。

因此“动作数据如何从视频生产出来”目前不能仅靠主分支完整复现。

数据加载实现在 [g1_utils.py](./Humanoid-Goalkeeper/legged_gym/legged_gym/envs/g1/g1_utils.py:45)。

每份 `.pt` 中包含类似：

```text
base_position
base_pose
joint_position
joint_velocity
link_position
link_orientation
lin_velocity
link_angular_velocity
```

但当前 AMP 实际使用的输入只有关节位置。

---

## 2. 判别器看什么

每个 AMP 判别器的输入是相邻两个时刻的关节位置：

\[
(q_t,q_{t+1})
\]

G1 有 29 个控制关节，因此输入维度是：

```text
29 + 29 = 58
```

为什么不是只看 \(q_t\)？

因为单帧姿态看不出运动方向。

例如同一个抬手姿态可能处于：

- 正在抬手；
- 正在放手；
- 静止保持。

加入下一帧后，判别器能够识别局部动作趋势。

---

## 3. 六个判别器

代码会把同一个 AMP 网络复制六份：

```text
D_lefthand
D_righthand
D_leftjump
D_rightjump
D_leftstep
D_rightstep
```

实现在 [him_ppo.py](./Humanoid-Goalkeeper/rsl_rl/rsl_rl/algorithms/him_ppo.py:89)。

每个判别器只比较对应区域的两种数据：

```text
专家样本：对应区域的人类动作
策略样本：策略在对应落球区域执行的动作
```

它使用 least-squares GAN 风格的目标：

```text
专家动作输出接近 +1
策略动作输出接近 -1
```

判别器损失可以理解为：

\[
L_D =
(D(q^{expert})-1)^2+
(D(q^{policy})+1)^2
\]

并额外加了梯度惩罚，提高训练稳定性。

---

## 4. AMP 奖励为什么不是普通动作跟踪

普通动作跟踪会计算：

\[
\|q_t-q_t^{ref}\|^2
\]

这要求当前动作和参考动作在时间上对齐。

AMP 则只问：

> 你当前这段局部关节变化，像不像专家动作分布中的某一段？

因此它允许：

- 更早伸手；
- 更晚跳起；
- 动作幅度略有变化；
- 根据球速改变动作节奏；
- 在专家动作附近自行寻找更合适的挡球姿态。

对于动态交互任务，这比逐帧模仿更合适。

---

## 5. 论文提出的“软 AMP”是什么

普通 AMP 直接把机器人实际动作送进判别器：

```text
实际动作不像专家 → 低奖励
```

这可能约束太强。

守门时，实际球落点与参考示范不会完全一致。为了碰到球，手的位置可能必须偏离专家动作十几厘米。

作者的做法是：

1. 以实际动作转换 \((q_t,q_{t+1})\) 为中心；
2. 采样 20 个高斯扰动邻居；
3. 把这 20 个邻居全部送进判别器；
4. 取判别器得分最高的一个作为 AMP 奖励。

代码位于 [amp.py](./Humanoid-Goalkeeper/rsl_rl/rsl_rl/modules/amp.py:185)，默认参数是：

```python
num_samples = 20
sigma = 0.3
```

直觉上，它不是问：

> 你的动作有没有精确落在专家流形上？

而是问：

> 你的动作附近，有没有一个合理的人类动作？

这样会把原本很尖锐的动作先验变成一个更宽的“动作走廊”。

优点是任务奖励有更多自由度；风险是最大值操作比较乐观，可能让离专家分布稍远的动作也获得较高奖励。

---

# 八、Actor 到底看到了什么

这是理解代码最重要的部分。

## 1. 单帧 Actor 观测：96 维

配置在 [g1_29_config.py](./Humanoid-Goalkeeper/legged_gym/legged_gym/envs/g1/g1_29_config.py:3)。

每一帧包括：

| 观测 | 维度 | 含义 |
|---|---:|---|
| 球在 torso 局部坐标的位置 | 3 | 球在哪里 |
| base/pelvis 角速度 | 3 | 身体转得多快 |
| 投影重力 | 3 | 身体倾斜方向 |
| 关节位置 | 29 | 当前姿态 |
| 关节速度 | 29 | 当前运动趋势 |
| 上一步动作 | 29 | 控制器刚刚要求什么 |
| 合计 | 96 | |

所以：

\[
3+3+3+29+29+29=96
\]

球位置使用机器人局部坐标，而不是世界坐标。这是为了让仿真和真机统一：

- MoCap 给出世界坐标后转换到机器人坐标；
- 相机天然给出相机/机器人局部坐标；
- 策略不需要知道机器人在足球场的绝对位置。

## 2. Actor 看 10 帧历史

策略输入不是 96 维，而是最近 10 帧拼接：

\[
96 \times 10=960
\]

物理仿真步长是 0.005 秒，策略每 4 个物理步运行一次，因此：

```text
仿真频率：200 Hz
策略频率：50 Hz
10 帧历史：约 0.2 秒
```

0.2 秒历史足以粗略估计：

- 球速度；
- 球下降趋势；
- 未来落点；
- 球属于哪个区域；
- 当前球是否已经被挡回。

这里没有使用 RNN、LSTM 或 Transformer，而是直接把 10 帧展平后送进 MLP。

优点是：

- 简单；
- 推理稳定；
- 容易导出 ONNX；
- 没有 recurrent hidden state 的重置问题。

缺点是历史窗口固定，长期信息能力有限。

---

# 九、非对称 Actor–Critic

训练时 Actor 和 Critic 看到的信息不同。

## Actor 只看可部署信息

Actor 只能看到真机可以获得的：

- 球位置；
- IMU 相关信息；
- 关节位置和速度；
- 上一步动作；
- 历史观测。

## Critic 额外看仿真真值

Critic 是 113 维，包括 Actor 的当前 96 维，加上：

| 特权信息 | 维度 |
|---|---:|
| base 线速度 | 3 |
| 真实目标区域 | 1 |
| 真实拦截目标点 | 3 |
| 真实球速度 | 3 |
| 左右手位置 | 6 |
| 手到目标的距离 | 1 |
| 合计额外信息 | 17 |

因此：

\[
96+17=113
\]

这叫 asymmetric actor–critic。

为什么有效？

Critic 的任务是评价：

```text
当前状态未来能拿多少回报？
```

训练时让 Critic 看到更准确的真值，可以减少价值估计噪声，从而给 Actor 更稳定的梯度。

但部署时只保留 Actor，所以不会依赖特权信息。

---

# 十、策略网络内部还有两个辅助任务

网络定义在 [actor_critic.py](./Humanoid-Goalkeeper/rsl_rl/rsl_rl/modules/actor_critic.py:92)。

## 1. 历史编码器

```text
960 → 128 → 64 → 16
```

它把最近 10 帧压缩成 16 维历史 latent。

可以理解为策略自己学习的一份状态摘要：

```text
球正快速飞向右上
机器人当前向左倾斜
左脚刚离地
```

它并没有显式语义，但理论上会编码这些信息。

## 2. 球轨迹估计器

```text
960 → 128 → 32 → 6
```

论文把它叫 ball estimator。

从实际代码看，这 6 维监督目标是：

```text
3 维拦截目标点
+
3 维真实球速度
```

所以它不只是“预测球位置”，更准确地说是在预测：

```text
未来拦截目标 + 当前球速度
```

使用 MSE：

\[
L_{ball}=\|\hat b-b^{gt}\|^2
\]

## 3. 区域分类器

```text
960 → 128 → 32 → 6
```

输出六个区域 logits，使用交叉熵训练。

然后代码做：

```python
region_id = argmax(region_logits)
```

把预测出来的整数区域编号直接送给 Actor。

一个值得注意的工程细节是：

- `argmax` 不可导；
- 区域分类器主要由交叉熵监督；
- PPO 的任务梯度不能通过这个离散 ID 反传给区域分类器。

这让模块更像一个硬路由器。

更现代的实现可能会把六维 softmax 概率直接送进 Actor，而不是只送一个 `argmax` 整数，这样可以表达不确定性。

---

# 十一、Actor 和 Critic 的完整结构

Actor 最终输入是：

```text
当前一帧观测       96
历史 latent        16
预测目标和球速度    6
预测区域 ID         1
----------------------
总计              119
```

Actor MLP：

```text
119 → 512 → 256 → 256 → 29
```

Critic MLP：

```text
113 → 512 → 256 → 256 → 1
```

训练时 Actor 输出高斯分布的均值，另外学习 29 维标准差：

\[
a \sim \mathcal N(\mu_\theta(s),\sigma)
\]

这样训练时有探索。

部署时直接使用均值：

```python
action = actor_mean
```

不再随机采样。

---

# 十二、动作输出不是电机力矩

论文把它叫端到端策略，但这里要精确定义“端到端”。

它不是：

```text
摄像头像素 → 电机力矩
```

而是：

```text
上游估计好的三维球位置
      +
机器人本体状态
      ↓
RL 策略
      ↓
29 个关节位置偏移
      ↓
PD 控制器
      ↓
电机力矩
```

策略动作经过：

\[
q_{target}=q_{default}+0.25a
\]

然后 PD 控制器：

\[
\tau =
K_p(q_{target}-q)-K_d\dot q
\]

最后按照 URDF 力矩限制裁剪。

实现位于 [legged_robot.py](./Humanoid-Goalkeeper/legged_gym/legged_gym/envs/base/legged_robot.py:627)。

这是一种很常见且实用的 sim-to-real 结构：

- RL 负责高层全身协调；
- PD 负责高频关节稳定；
- 策略只需要 50 Hz；
- 电机底层控制可以更高频。

---

# 十三、完整训练目标

每一步真正给 PPO 的 reward 是：

```python
reward = 0.4 * amp_reward + 0.6 * task_reward
```

代码在 [him_on_policy_runner.py](./Humanoid-Goalkeeper/rsl_rl/rsl_rl/runners/him_on_policy_runner.py:185)。

其中 task reward 又包含：

```text
手接近动态目标
手进入严格接触区
真实改变球速度
正确方向的身体运动
保持在球门线附近
不向后逃
双脚姿态
跳起后成功落地
减少脚底打滑
降低冲击
挡球后恢复身体方向
挡球后降低角速度、线速度
挡球后恢复手臂和腰部
限制角速度
限制关节加速度
限制动作二阶差分
限制力矩和关节速度
限制位置、速度和力矩越界
```

PPO 的优化 loss 可以概括为：

\[
L =
L_{PPO}
+L_{value}
+L_{ball}
+L_{region}
+L_{smooth}
+L_{AMP-discriminator}
-\beta H
\]

要区分两个 AMP 作用路径：

1. AMP 判别器输出 reward，reward 进入 PPO return，从而训练 Actor；
2. 判别器自己的分类 loss 只负责训练六个判别器。

---

# 十四、除了 reward，还有额外平滑约束

代码不仅有动作二阶差分 reward：

\[
a_t-2a_{t-1}+a_{t-2}
\]

还实现了一个观测插值平滑 loss。

它会在相邻状态之间随机插入一个中间状态：

\[
s_{mix}=s_t+\lambda(s_{t+1}-s_t)
\]

然后要求：

```text
相邻状态之间的 Actor 输出不要突变
相邻状态之间的 Critic 输出不要突变
```

这相当于对策略施加局部 Lipschitz 平滑约束。

对于真机很有意义，因为传感器位置轻微抖动时，不希望动作从“左扑”瞬间跳成“右扑”。

---

# 十五、仿真环境是怎么生成球的

每次 reset：

1. 球的起点 x 在 3～5 米；
2. 起点 y、z 随机；
3. 从对应区域采样终点；
4. 终点在机器人身后约 `x=-0.1～-0.6m`；
5. 随机采样飞行时间；
6. 根据抛体公式反推初速度。

竖直初速度满足：

\[
v_z =
\frac{\Delta z+\frac{1}{2}gt^2}{t}
\]

这样球在重力作用下经过指定终点。

代码位于 [legged_robot.py](./Humanoid-Goalkeeper/legged_gym/legged_gym/envs/base/legged_robot.py:771)。

一个细节是，代码不是每次 episode 随机给环境换区域，而是把 6144 个并行环境平均分成六组：

```text
前 1/6 永远负责 region 0
第二个 1/6 负责 region 1
……
最后 1/6 负责 region 5
```

这样每个训练 batch 天然区域均衡，避免某类动作样本太少。

---

# 十六、课程学习

训练不会一开始就把球投到最远的位置。

初始落点范围较窄，例如中区大约：

```text
横向：0.2～1.2m 或 -1.2～-0.2m
高度：0.4～1.2m
```

高球和低球各自有自己的范围。

随着训练推进，代码逐渐扩大范围，最终接近：

```text
横向：±1.8m
高度：最高约 1.8m
```

课程推进时还会：

- 增强接近球、成功和挡球奖励；
- 增强关节位置和力矩越界惩罚；
- 提高困难阶段的安全约束。

课程学习的目的很直观：

```text
先学会近处伸手挡球
再学侧步
再学远距离扑挡
最后学极限区域跳跃
```

如果一开始球就覆盖整个球门，成功奖励过于稀疏，策略很可能什么也学不到。

---

# 十七、Sim-to-Real 是怎么做的

## 1. 两种球感知方案

论文用了两套系统。

### MoCap

- 机器人头部有 marker；
- 球上有 marker；
- 动捕系统给出二者世界坐标；
- 转换为机器人 torso 局部坐标。

优点：

- 范围大；
- 精度高；
- 球远近都能看到。

缺点：

- 依赖外部场地；
- 可能遮挡；
- 不是真正自主。

### RealSense D435i

- 安装在机器人头部；
- 使用红外滤光和高反射球；
- 在红外图中检测球；
- 结合深度得到三维球坐标。

优点：

- 完全机载；
- 更接近真实比赛部署。

缺点：

- 视场角有限；
- 球靠近身体时容易离开画面；
- 论文报告低区域可观测时间甚至不足 0.3 秒。

---

## 2. 球位置噪声

论文写的是最大约 5 cm 的位置扰动。

开源配置实际使用：

```python
noise_scales.ball = 0.08
```

即每个轴加入大约 `[-0.08, 0.08]` 的均匀噪声。

这是论文描述与开源超参数之间的一个小差异。

## 3. 随机丢球观测

代码会把球位置突然设为 0，模拟：

- 球离开相机 FOV；
- MoCap 遮挡；
- 球靠近身体后检测失败；
- 球已被挡走。

因为 Actor 有 10 帧历史和显式球轨迹估计器，即使最后几帧球消失，仍能根据之前轨迹继续执行动作。

这很像人类：

> 球接近身体前虽然暂时看不见了，但人已经根据先前轨迹完成扑救动作。

---

## 4. 物理随机化

代码的 domain randomization 很全面：

- 机器人 base 额外质量：`-5～+10kg`；
- 质心偏移：每轴 ±0.1m；
- 其他 link 质量：0.8～1.2 倍；
- 摩擦系数：0.1～2.0；
- 恢复系数：0～1；
- Kp、Kd：0.8～1.2 倍；
- 初始关节位置随机；
- 电机力矩偏置；
- 随机关节扰动力矩；
- 动作延迟；
- 随机推机器人；
- 球速度扰动；
- 空气阻力和额外随机力。

这覆盖了三类 sim-to-real 误差：

```text
感知不准：球位置噪声、丢帧
执行不准：延迟、力矩偏置、Kp/Kd 偏差
物理不准：质量、质心、摩擦、碰撞恢复、球飞行误差
```

---

# 十八、终止条件

episode 在以下情况终止：

- 膝盖高度低于 0.1m；
- 身体倾斜过大；
- 脚部冲击过大；
- 到达 3 秒超时。

这实际上构成了一套硬安全边界。

Reward 负责告诉机器人“尽量别摔”，termination 则告诉机器人：

> 一旦明显摔倒，这次轨迹直接结束。

---

# 十九、PPO 训练流程

训练配置主要是：

```text
并行环境数：6144
每个环境每轮采样：100 steps
PPO epoch：5
mini-batch 数：4
gamma：0.99
GAE lambda：0.95
clip：0.2
学习率：1e-3，自适应 KL
目标 KL：0.01
entropy：0.01
```

一次 iteration 会采集：

\[
6144\times100=614400
\]

个 environment transitions。

完整循环是：

```text
1. Actor 根据 10 帧历史输出动作
2. PD 控制器执行 4 个物理仿真步
3. 环境计算 task reward
4. 对应区域 AMP 判别器计算 motion reward
5. 按 0.6/0.4 混合奖励
6. 存入 rollout buffer
7. 采满 100 步
8. Critic 计算 bootstrap value
9. GAE 计算 advantage 和 return
10. PPO 更新 Actor/Critic
11. 更新球轨迹估计器和区域分类器
12. 更新六个 AMP 判别器
```

---

# 二十、论文实验说明了什么

## 1. 完整方法的仿真成功率

默认设置：

```text
飞行时间：0.5～1.0s
横向范围：±1.5m
高度：约 1.8m
```

完整方法成功率：

```text
80.92%
```

## 2. 去掉任务约束

成功率只有：

```text
9.44%
```

但它的动作平滑指标最好。

这说明：

> 只要求动作像人，机器人会优先做平滑、自然的动作，却没有足够动力真正去碰球。

“自然”不等于“有用”。

## 3. 去掉动作约束

成功率：

```text
31.69%
```

它可能会探索到能碰球的动作，但动作不稳定、不自然，也更难形成区域明确的全身协调。

## 4. 不划分 AMP

成功率：

```text
76.11%
```

说明区域任务奖励本身已经能引导一部分动作选择，但动作区域匹配率明显降低。

## 5. 完全不按区域划分

成功率：

```text
64.83%
```

会出现动作模式混合和左右偏置。

## 6. 更困难的设置

```text
更快球，0.4～0.7s：64.25%
更宽范围，±2.0m：63.91%
```

这说明方法不是彻底失效，但响应时间和机器人物理覆盖范围仍然是上限。

---

# 二十一、真机结果该怎么看

使用 MoCap 时，完整方法：

```text
21 / 30 成功，约 70%
```

使用机载相机：

```text
14 / 30 成功，约 46.7%
```

主要失败集中在：

- 高球；
- 极限横向区域；
- 相机近距离视野丢失；
- 需要精确对齐的跳跃动作。

值得注意的是，MoCap 结果不是 90% 以上。这说明即使仿真成功率约 81%，真机仍存在明显差距。

相机版本进一步下降，说明当前系统最主要的瓶颈可能已经不是 RL 本身，而是：

```text
观测时间窗口
球的三维定位
视场角
延迟
标定误差
```

---

# 二十二、扩展任务为什么能复用这套方法

论文还展示了两类任务。

## 1. 躲球

按球位置划分：

```text
高球 → 下蹲躲避
低球 → 跳起躲避
```

或者按相应空间语义分配 jump/squat 动作先验。

结果：

```text
Jump Escape：3/5
Squat Escape：5/5
```

## 2. 用软袋抓球

按左右方向划分：

```text
左侧球 → 左移抓球动作
右侧球 → 右移抓球动作
```

结果只有：

```text
1/5
```

主要不是策略完全没到位，而是仿真中把袋口视为固定打开的几何区域，真机软袋会：

- 变形；
- 扭转；
- 袋口闭合；
- 朝向不稳定。

这是典型的 sim-to-real 物体建模失败。

---

# 二十三、项目代码结构

## `legged_gym`

负责：

- Isaac Gym 环境；
- G1 URDF；
- 球物理；
- 观测构造；
- reward；
- reset；
- domain randomization；
- 训练与播放脚本。

最核心文件：

- [g1_29_config.py](./Humanoid-Goalkeeper/legged_gym/legged_gym/envs/g1/g1_29_config.py)
- [legged_robot.py](./Humanoid-Goalkeeper/legged_gym/legged_gym/envs/base/legged_robot.py)
- [g1_utils.py](./Humanoid-Goalkeeper/legged_gym/legged_gym/envs/g1/g1_utils.py)

## `rsl_rl`

是基于 RSL-RL 修改的训练框架，主要加入：

- 历史编码；
- 球轨迹估计；
- 区域分类；
- 六个 AMP 判别器；
- AMP reward；
- 平滑 loss；
- 自定义 rollout storage。

核心文件：

- [actor_critic.py](./Humanoid-Goalkeeper/rsl_rl/rsl_rl/modules/actor_critic.py)
- [amp.py](./Humanoid-Goalkeeper/rsl_rl/rsl_rl/modules/amp.py)
- [him_ppo.py](./Humanoid-Goalkeeper/rsl_rl/rsl_rl/algorithms/him_ppo.py)
- [him_on_policy_runner.py](./Humanoid-Goalkeeper/rsl_rl/rsl_rl/runners/him_on_policy_runner.py)

## `resources`

包含：

- G1 URDF 和 mesh；
- 球 URDF；
- 六份动作数据；
- 官方 goalkeeper checkpoint；
- escape checkpoint。

## `scripts`

- `train.py`：训练；
- `play.py`：仿真评估；
- 支持导出 TorchScript 和 ONNX。

策略导出时只保留：

```text
history_encoder
ball_estimator
region_estimator
actor
```

不会导出 Critic 和 AMP 判别器，这是正确的，因为部署时不需要它们。

---

# 二十四、开源项目没有提供什么

这是工程上必须明确的地方。

当前主分支没有提供：

1. 真机控制循环；
2. Unitree G1 SDK 接口；
3. MoCap 接收与坐标转换代码；
4. RealSense 红外球检测代码；
5. 相机标定代码；
6. 真机安全状态机；
7. 从视频到机器人动作的完整数据生产脚本；
8. 论文评估指标的完整复现脚本；
9. 真机实验数据；
10. 连续比赛级高层规划器。

因此这个仓库更准确的定位是：

> 训练环境、核心 RL/AMP 算法、动作先验数据和仿真推理代码的开放实现。

它不是一个 clone 下来就能在 G1 上运行的完整产品。

逃球任务在单独的 `escape` 分支，不在 `main`。

---

# 二十五、代码中值得注意的工程问题

## 1. AMP 判别器没有被保存进 checkpoint

`save()` 只保存：

```python
self.alg.actor_critic.state_dict()
self.alg.optimizer.state_dict()
```

六个 AMP 判别器不属于 `actor_critic`，因此它们的模型权重没有直接保存。

虽然 optimizer state 保存了动量等状态，但它不等于保存模型参数。

结果是：

- 推理 checkpoint 没问题，因为推理不需要 AMP；
- 从 checkpoint 恢复训练时，六个判别器可能重新随机初始化；
- resume 训练不完全等价于从原训练状态继续。

这是开源实现里一个比较明显的 checkpoint 完整性问题。

## 2. 区域 ID 使用硬 `argmax`

策略看到的是一个整数 `0～5`，而不是六维区域概率。

这样容易出现边界跳变：

```text
上一帧：region 2，准备左跳
下一帧：region 0，突然变成左手扑
```

代码用历史窗口和平滑 loss 缓解，但 soft probability 或 mixture-of-experts gating 可能更自然。

## 3. Actor 没有直接看到估计器置信度

相机快丢球时，估计器可能很不确定，但 Actor 只拿到：

```text
一个确定的六维估计
一个硬区域编号
```

没有：

- 方差；
- 置信度；
- 当前是否丢帧；
- 最近一次有效观测距今多久。

这些信息加入后可能提升真机鲁棒性。

## 4. README 的训练量描述比较含糊

README 写 goalkeeper 大约“20k episodes”收敛，但代码是：

```text
6144 个并行环境
每 iteration 采 100 步
最多 200000 iterations
```

这里的“20k episodes”很可能是泛指训练进度或 iteration，而不是严格意义上单个 episode 数量，否则和并行规模不太匹配。

## 5. 开源版本依赖较老

它依赖：

- Python 3.8；
- NVIDIA Isaac Gym；
- `numpy<1.24`；
- CUDA/NVIDIA GPU。

所以在当前 macOS 上不能原生跑完整 Isaac Gym 训练。通常需要 Linux + NVIDIA GPU 环境。

---

# 二十六、这项工作的真正优点

从算法工程师角度，我认为它最值得学习的不是某个公式，而是以下设计组合。

## 1. 把多模态问题显式拆成区域

不是让一个判别器自己理解六种动作，而是用任务语义帮它做条件化。

## 2. 任务约束和模仿约束共享同一条件

区域同时控制：

- 哪只手追目标；
- 身体向哪运动；
- 使用哪份专家动作；
- 使用哪个判别器。

这避免任务奖励与动作先验互相打架。

## 3. 模仿动作风格而不是逐帧轨迹

保留了根据真实球轨迹调整时机和幅度的自由度。

## 4. 显式学习感知摘要

历史 encoder、球轨迹预测和区域分类都在策略网络内部联合训练，减少了手写抛物线预测器与策略之间的接口误差。

## 5. 把任务后稳定性纳入目标

避免“碰到球就摔”的奖励漏洞。

## 6. 通过任意姿态 reset 训练连续技能

让策略不依赖标准初始姿态，增强动作之间的连接能力。

---

# 二十七、它的本质局限

## 1. 不是完整比赛级守门员

它是局部反应式技能：

```text
检测到球 → 判断区域 → 执行拦截动作
```

它没有：

- 判断球是否真的射向球门；
- 赛场全局定位；
- 根据射门者姿态提前预测；
- 守门员站位规划；
- 挡球后回到最佳站位；
- 多次连续射门的高层决策；
- 与队友协作。

论文自己也明确指出，它可能对视野内任何球都反应，即使那个球根本不会进门。

## 2. 区域是人工离散的

六区域简单有效，但边界处不连续。

更理想的方案可能是连续条件：

```text
落点 y、z
到达时间 t
速度方向
置信度
```

再用连续条件动作生成模型，而不是硬分六类。

## 3. AMP 只看两帧关节位置

58 维局部 transition 能识别基本动作风格，但没有直接看：

- base 位移；
- 手的位置；
- 脚接触；
- 身体方向；
- 长时间动作结构。

因此它学习的是较局部的关节运动先验，不是完整的全身时序动作分布。

## 4. 感知仍是最大瓶颈

MoCap 21/30，相机 14/30，说明训练噪声不能完全解决：

- 相机 FOV；
- 深度误差；
- 遮挡；
- 延迟；
- 极短观测时间；
- 球检测错误。

## 5. 抓球需要更真实的接触建模

“挡球”只需改变球速度，对手部接触精度要求相对低；“抓球”涉及柔性袋、接触时序和袋口方向，当前刚性仿真假设明显不够。

---

# 最后用一句容易记住的话总结

这篇论文不是在教机器人背六套守门动作，而是在做下面这件事：

> 让机器人根据球的历史轨迹，自己判断该使用哪类人类动作；任务奖励负责告诉它“球必须挡住”，区域专属 AMP 负责告诉它“应该以哪种方式挡”，而 PPO 在二者之间寻找既有效、又自然、还能真机执行的全身动作。

如果用机器学习术语概括，它是一套：

```text
非对称 Actor–Critic
+ 历史状态估计
+ 辅助轨迹预测
+ 区域分类
+ 条件化多判别器 AMP
+ PPO
+ 课程学习
+ Domain Randomization
+ 任务后稳定控制
```

最终部署策略本身则相对简洁：

```text
960 维历史观测
      ↓
历史编码 + 轨迹估计 + 区域分类
      ↓
119 维 Actor 输入
      ↓
29 维关节目标
      ↓
50 Hz 策略 + 200 Hz 仿真/底层 PD
```

> ssh -p 7980 fjc@111.2.199.31
> jushenfjc
> 看看服务器里最新的checkpont到多少了

<details><summary>3 previous messages</summary>

> 我现在登录服务器，只做只读检查：先定位训练目录和日志，再按 checkpoint 编号与修改时间确认最新进度，不会改动训练任务。
>
> <details><summary>Ran a command</summary>
>
> - Ran `ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=15 -p 7980 fjc@111.2.199.31`
>
> </details>
>
> 已经定位到了训练目录：当前有 7 个训练进程仍在运行（基线 `711-1`，以及 seed2、seed4～seed8；没有看到 seed3 进程）。我再按每个实验目录取“编号最大”的文件，并核对更新时间，避免把“最新写入”和“checkpoint 编号最高”混为一谈。
</details>

服务器上当前最高 checkpoint 是 **10800**。

| 实验 | 最新 checkpoint | 北京时间 |
|---|---:|---|
| `711-1` | **10800** | 07-14 09:38 |
| `711-seed4` | **10800** | 07-14 09:02 |
| `711-seed2` | 10600 | 07-14 09:46 |
| `711-seed5` | 10600 | 07-14 09:24 |
| `711-seed6` | 10600 | 07-14 09:30 |
| `711-seed7` | 10600 | 07-14 08:55 |
| `711-seed8` | 10600 | 07-14 09:16 |
| `711-seed3` | 7400 | 07-13 16:07 |

目前 `711-1`、seed2、seed4～seed8 共 **7 个训练进程仍正常运行**，每个占用约 21 GB 显存。

`seed3` 没有在运行，停在 **7400**。最新写入的是 seed2 的 `model_10600.pt`，说明训练仍在持续推进。

> 怎么把seed3停了，我指的是整个进程都kill

<details><summary>2 previous messages</summary>

> 我会按 `711-seed3` 的完整命令行匹配，先查看 PID、PPID、进程组和会话，再终止整个进程组，避免只杀 `python` 后被外层 shell 拉起。上次检查时 seed3 已经没有训练进程；这次会再次确认，也会检查是否有残留启动器。
>
> <details><summary>Ran a command</summary>
>
> - Ran `ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=15 -p 7980 fjc@111.2.199.31`
>
> </details>
</details>

`seed3` 目前已经彻底停了，不需要再 kill：

- 没有匹配 `711-seed3` 的 Python 或父 shell 进程
- 没有 `gk_seed3` 的 tmux session
- GPU 上也没有 seed3 进程
- 最后停在 `model_7400.pt`

我没有动其他训练；`711-1` 和 seed2、seed4～seed8 都还正常运行。

以后要彻底停止某个 seed，直接杀对应 tmux session：

```bash
tmux kill-session -t gk_seed3
```

这会连同 session 内的父 shell、Python 训练进程及其子进程一起结束。

> 这些现阶段效果最差的是哪两个

<details><summary>6 previous messages</summary>

> 我会用各实验相同训练阶段的指标来比较，优先看守门成功相关指标和最近一段平均回报，避免仅凭 checkpoint 编号判断。先读取 TensorBoard/W&B 日志里的实际 metric 名称，再给出最差两个及依据。
>
> <details><summary>Ran a command</summary>
>
> - Ran `ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=15 -p 7980 fjc@111.2.199.31`
>
> </details>
>
> 日志没有直接记录 `success_rate`，但有两个可靠代理指标：`rew_stopball`（球是否被真正挡停/改变速度）和 `rew_success`（指定手是否到达严格拦截距离），再结合 `mean_raw_reward`。我会同时看最近 200 次迭代均值和所有实验共同的 7000～7400 阶段，防止 seed3 因训练更早停止而被不公平比较。
>
> 你直接查看tmux里的输出
>
> 好，我直接读取各个训练 tmux pane 的最近输出，按它们打印的 `Mean reward`、各 reward 项和当前 iteration 横向比较，不再只依赖 TensorBoard 聚合值。
>
> <details><summary>Ran a command</summary>
>
> - Ran `ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=15 -p 7980 fjc@111.2.199.31`
>
> </details>
</details>

直接看各 tmux 最新两轮输出，当前效果最差的是：

1. `seed7`

   - Mean reward：`5.32 → 6.92`
   - stopball：`0.4959 → 0.4870`
   - success：`1.9459 → 2.1262`
   - eereach：`14.4100 → 14.7137`

2. `seed8`

   - Mean reward：`10.95 → 9.86`
   - stopball：`0.5181 → 0.4865`
   - success：`2.1338 → 2.1701`
   - eereach：`14.6204 → 14.8308`

其中 `seed7` 整体最差；`seed8` 最新的挡球指标甚至略低于 seed7。第三差是 `seed4`，但它的 `stopball≈0.54`、`success≈2.51`，明显好于这两个。

我没有终止任何进程。

> gk_tensorboard 这个tmux是啥玩意

`gk_tensorboard` 是专门运行 TensorBoard 可视化服务的 tmux，不是训练进程。

它执行的是：

```bash
tensorboard \
  --logdir=/home/fjc/shoumen/Humanoid-Goalkeeper/legged_gym/logs/g1 \
  --host=127.0.0.1 \
  --port=16006
```

作用是读取所有 seed 的训练日志，在网页里展示 reward、loss 等曲线。它不会更新模型，也基本不占 GPU。

本地可以通过 SSH 隧道访问：

```bash
ssh -p 7980 -L 16006:127.0.0.1:16006 fjc@111.2.199.31
```

然后浏览器打开：

```text
http://127.0.0.1:16006
```

如果杀掉它，只会关闭 TensorBoard 网页，不影响任何训练：

```bash
tmux kill-session -t gk_tensorboard
```

> 怎么退出当前tmux

如果想退出 tmux、但让里面的程序继续运行，按：

```text
Ctrl+B
```

松开后再按：

```text
D
```

也就是 tmux 的 detach。

如果要直接关闭当前 tmux，并终止里面的程序，输入：

```bash
exit
```

训练任务一般使用 `Ctrl+B`、`D`，避免误杀。
