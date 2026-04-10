# DeepLearning 课程实践总结

这个目录收集了几份循序渐进的深度学习练习脚本，目标不是只“跑通模型”，而是通过可执行代码把训练流程、模型结构和实验分析串起来。

## 目录结构

- `Class_1.py`：前馈神经网络（MLP）与二分类训练流程。
- `Class_2.py`：序列建模入门，从字符级 RNN 到简化 Transformer。
- `Class_3.py`：小型 CNN 目标检测器，包含分类头、边框回归头与实时推理。
- `Class_4.py`：卷积自编码器与表示学习，包含重建、线性探针、PCA 与 t-SNE 可视化。
- `Class_5.py`：深度强化学习，包含离散动作的 DQN 与连续动作的策略梯度示例。
- `artifacts/`：运行脚本后生成的模型权重、重建图、可视化结果等产物。

---

## 第 1 课：MLP 与监督学习训练流程

`Class_1.py` 重点演示一个完整的监督学习工程链路，适合用来理解“数据、模型、损失、优化器、验证集”是如何串起来工作的。

### 你会看到什么

- 合成二分类数据集的构造方式。
- 小型 MLP 的定义与前向传播。
- `BCEWithLogitsLoss`、`AdamW`、`Dropout`、Early Stopping 等常见训练组件。
- checkpoint 保存与训练曲线绘制。

### 这一课的核心收获

- 模型不只是要拟合训练集，还要尽量提升泛化能力。
- 正则化、验证集监控和早停机制是避免过拟合的重要工具。

---

## 第 2 课：序列建模，从 RNN 到 Transformer

`Class_2.py` 用字符级语言建模任务，把序列模型最核心的两条路线放在一起对比：RNN 和 Transformer。

### 你会看到什么

- 字符表构建、文本编码和解码。
- RNN 如何用隐藏状态处理时序输入。
- Self-Attention 的基本思想与简化实现。
- 一个极简 Transformer Block 和文本生成流程。

### 这一课的核心收获

- RNN 擅长从“时间步”角度理解序列，但存在长程依赖和串行计算瓶颈。
- Transformer 通过注意力机制直接建模全局依赖，更适合并行训练。

---

## 第 3 课：CNN 目标检测基础

`Class_3.py` 演示一个教学型的小目标检测例子：在合成图形图像上同时预测类别和边框位置，并支持实时摄像头推理。

### 你会看到什么

- 合成检测数据集：图像、类别标签、边框标签。
- 一个共享 backbone、双头输出的 Tiny CNN Detector。
- 分类损失与边框回归损失的联合训练。
- IoU、MAE、分类准确率等评估指标。
- checkpoint 保存、加载以及实时检测展示。

### 这一课的核心收获

- 检测任务本质上是“分类 + 定位”的组合问题。
- 多任务损失设计和评估指标的选择，会直接影响模型训练效果。

---

## 第 4 课：自编码器、VAE 与 GAN

`Class_4.py` 现在覆盖了第四节课的三部分内容：普通 Autoencoder、Variational Autoencoder（VAE）和最简 GAN。它既保留了表示学习实验，也补上了概率生成与对抗生成的教学代码。

### 你会看到什么

- 合成灰度图形数据集：`square`、`h_bar`、`v_bar`、`x_shape`。
- 普通 Autoencoder：编码器如何把 `16x16` 图像压缩到 `16` 维潜在向量，解码器如何重建原图。
- AE 表示分析：逐像素 `MSE`、线性探针、PCA 与 t-SNE 可视化。
- VAE：`mu / logvar`、重参数化技巧、`reconstruction + KL` 联合损失。
- VAE 可视化：重建图、随机采样图、latent traversal 图。
- GAN：最简 `Generator / Discriminator` 架构与对抗训练流程。
- GAN 输出：从随机噪声生成新样本。

### 运行后会生成什么

运行 `Class_4.py` 后，通常会在 `artifacts/` 目录下生成：

- `autoencoder_shapes_v2.pt`：自编码器权重文件。
- `autoencoder_reconstructions.png`：原图与重建图对比图。
- `latent_pca.png`：latent space 的 PCA 可视化。
- `latent_tsne.png`：latent space 的 t-SNE 可视化。
- `vae_shapes.pt`：VAE 权重文件。
- `vae_reconstructions.png`：VAE 原图与重建图对比。
- `vae_samples.png`：VAE 从标准正态采样后的生成结果。
- `vae_latent_traversal.png`：固定其余维度、扰动部分 latent 后的生成图网格。
- `gan_shapes.pt`：GAN 生成器与判别器权重。
- `gan_samples.png`：GAN 从随机噪声生成的样本图。

### 这一课的核心收获

- bottleneck 会强迫模型学习压缩后的有效表示，而不是直接复制输入。
- reconstruction loss 会约束潜在表示保留还原输入所需的关键信息。
- 在当前这组简化图形数据上，`v2` 版本的纯 `MSE` 训练最稳定，实验中可得到约 `0.016` 的逐像素 MSE 和约 `0.65` 的线性探针准确率。
- PCA 与 t-SNE 可视化表明：同类样本在 latent space 中已经出现一定聚类，但不同类别之间仍有部分重叠，说明模型已经学到有区分度的表示，但仍有进一步提升空间。
- VAE 相比普通 AE，重建通常会更模糊一些，但 latent space 更适合采样、插值和扰动分析；本次实验里 VAE 验证集逐像素重建 MSE 约为 `0.048`。
- VAE 中最关键的思想是重参数化技巧：把随机性放到 `epsilon ~ N(0, 1)`，再令 `z = mu + sigma * epsilon`，从而让采样过程仍可反向传播。
- GAN 的核心结构是 `Generator + Discriminator + adversarial training`；本次最简 GAN 已能生成样本，但损失波动明显，说明对抗训练比 AE / VAE 更不稳定。

### 本次实验结果速记

- AE：重建最好，表示最稳定，适合理解 bottleneck 与 latent representation。
- VAE：重建略差，但更适合理解“可采样 latent space”和重参数化。
- GAN：直接从噪声生成样本，概念清晰，但训练稳定性最差。

---

## 第 5 课：深度强化学习（DQN 与 Policy Gradient）

`Class_5.py` 把 MIT 6.S191 第 5 课里最核心的两条强化学习路线拆成两个最小可运行实验：一个是离散动作空间上的 DQN，另一个是连续动作空间上的策略梯度。它对应的是“监督学习和无监督学习之外，深度学习如何在动态环境中做序列决策”这个主题。

### 这一课在讲什么

和前面几课最大的区别在于：`Class_5.py` 里的模型不是只看一份静态数据集，而是要不断和环境交互。

- 智能体会先观察当前状态 `state`。
- 然后根据策略或价值函数选择动作 `action`。
- 环境返回下一个状态 `next_state` 和即时奖励 `reward`。
- 模型的目标不再是预测单个标签，而是最大化长期累计回报 `return`。

这正是强化学习和普通监督学习最本质的区别：监督学习优化的是“当前样本预测得准不准”，而强化学习优化的是“这一连串动作最后值不值得”。

### 你会看到什么

- `LineWorldEnv`：一个离散状态、离散动作的 1D 环境，用来演示 DQN 的完整训练闭环。
- DQN 核心组件：经验回放、目标网络、epsilon-greedy 探索、Bellman target。
- `ContinuousDriveEnv`：一个连续控制小环境，用来演示策略梯度如何输出连续动作。
- Gaussian Policy：策略网络输出高斯分布参数，再从分布中采样动作。
- REINFORCE 训练流程：采样轨迹、计算 discounted return、按 `-log pi(a|s) * R` 更新策略。

### 脚本里的两条主线分别对应什么

#### 1. DQN：价值学习

这一部分回答的问题是：“在当前状态下，哪个动作的长期价值最高？”

- 网络输出的是每个动作的 `Q(s, a)`。
- 选动作时采用 `argmax_a Q(s, a)`。
- 训练目标来自 Bellman 方程：
  `target = r + gamma * max_a' Q(s', a')`
- 训练损失是当前 Q 预测与 target 之间的均方误差。

在这份脚本里，DQN 用的是 `LineWorldEnv`。环境很小，但该有的关键机制一个不少：

- `ReplayBuffer`：打散样本相关性。
- `target_net`：降低目标值快速震荡。
- `epsilon-greedy`：平衡探索和利用。

#### 2. Policy Gradient：策略学习

这一部分回答的问题是：“我能不能直接学一个策略，而不是先学 Q 函数？”

- 网络直接输出动作分布，而不是离散动作表上的价值。
- 连续动作通过高斯分布建模：输出均值 `mean` 和标准差 `std`。
- 训练时从分布中采样动作，评估时直接用均值动作。
- 核心更新形式是 REINFORCE：
  `loss = -log pi(a|s) * G`

在这份脚本里，策略梯度用的是 `ContinuousDriveEnv`，它更接近机器人控制的设定：动作是连续值，策略需要学会“朝目标加速，同时在接近目标时减速停稳”。

### 为什么第 5 课特意做成两个环境

这是为了把“离散动作”和“连续动作”这两类问题彻底分开看：

- `LineWorldEnv` 更适合理解 DQN，因为动作是有限个，网络一次前向传播就能枚举全部动作的 Q 值。
- `ContinuousDriveEnv` 更适合理解策略梯度，因为动作是连续值，没法像 DQN 那样穷举所有动作。

这也是工业里常见的算法分工：

- 游戏、离散决策、调度问题，通常更容易从 DQN / value learning 入门。
- 机器人控制、机械臂、自动驾驶这类连续控制任务，更自然地落在 Policy Gradient / Actor-Critic 路线上。

### 运行后会生成什么

运行 `Class_5.py` 后，通常会在 `artifacts/` 目录下生成：

- `dqn_lineworld.pt`：DQN 模型权重。
- `dqn_returns.png`：DQN 训练回报曲线。
- `dqn_success_rate.png`：DQN 训练成功率变化曲线。
- `dqn_training_history.txt`：DQN 逐回合训练记录，包含回报、成功标记和滑动平均。
- `dqn_policy_report.txt`：每个离散状态下的 greedy action 与对应 Q 值。
- `dqn_rollout.txt`：DQN 一次贪心 rollout 的详细轨迹。
- `policy_gradient_drive.pt`：策略梯度模型权重。
- `policy_gradient_returns.png`：策略梯度训练回报曲线。
- `policy_gradient_success_rate.png`：策略梯度训练成功率变化曲线。
- `policy_gradient_training_history.txt`：策略梯度逐回合训练记录。
- `policy_gradient_report.txt`：连续状态探针下的策略均值与标准差。
- `policy_gradient_rollout.txt`：连续控制任务中的一次确定性 rollout 轨迹。
- `policy_gradient_position_curve.png`：rollout 过程中位置变化曲线。

### 如何读这些输出结果

- `dqn_policy_report.txt`：适合检查 DQN 是否真的学会了“在大部分状态下都往目标方向走”。
- `dqn_rollout.txt`：适合逐步查看 DQN 一次完整决策序列。
- `dqn_training_history.txt`：适合回看每个 episode 的回报、是否成功，以及滑动平均是怎么变好的。
- `policy_gradient_report.txt`：适合看连续状态下策略输出的动作均值是否合理，例如离目标远时动作更大，接近目标时动作更平缓。
- `policy_gradient_rollout.txt`：适合看策略是不是只会“靠近目标”，还是已经学会“接近后减速并停下”。
- `policy_gradient_position_curve.png`：适合直观看到位置随时间的变化轨迹，判断是否存在明显过冲或停不稳。
- `policy_gradient_training_history.txt`：适合回看策略梯度训练中成功率从 0 逐步提升的过程。

### 这一课的核心收获

- DQN 属于价值学习：网络先近似 `Q(s, a)`，再通过 `argmax_a Q(s, a)` 得到策略。
- 经验回放和目标网络是 DQN 训练更稳定的关键工程技巧。
- 策略梯度属于直接策略学习：不再显式估计每个动作的 Q 值，而是直接优化动作分布。
- 连续动作空间无法像 DQN 那样枚举动作，因此策略梯度天然更适合机器人控制、自动驾驶、机械臂等问题。
- 这个脚本里的两个环境都很小，但已经能完整体现“状态 -> 动作 -> 奖励 -> 更新策略/价值函数”的强化学习闭环。

### 当前这份实现的教学取舍

- 它刻意保持“最小可读实现”，没有引入 `gymnasium`、`stable-baselines3` 等更完整的 RL 框架。
- DQN 保留了最重要的两个稳定训练技巧：经验回放和目标网络。
- 策略梯度使用的是最基础的 REINFORCE，而不是 PPO / SAC / TD3 这类工业级算法。
- 脚本里还加入了 checkpoint 体检逻辑：如果策略梯度旧权重质量太差，会自动重训，避免“加载到一份能跑但不成功的旧模型”。

### 如果继续往下学，下一步最自然的方向是什么

- 从 DQN 继续往后，可以学习 Double DQN、Dueling DQN、Prioritized Replay。
- 从 Policy Gradient 继续往后，可以学习 Advantage、Baseline、Actor-Critic、PPO、SAC。
- 如果你对机器人更感兴趣，那么这节课最值得延伸的方向通常是连续控制和 Sim2Real。`ContinuousDriveEnv` 就是在为这条路打基础。

---

## 建议的学习顺序

1. 先看 `Class_1.py`，把训练循环、损失函数和优化器的基本逻辑吃透。
2. 再看 `Class_2.py`，理解“为什么序列任务需要不同于 MLP 的建模方式”。
3. 然后看 `Class_3.py`，把分类问题扩展到“分类 + 定位”的检测问题。
4. 最后看 `Class_4.py`，理解无监督表示学习如何在没有显式标签的情况下提取有效特征。
5. 再看 `Class_5.py`，把深度学习从“静态数据学习”推进到“在环境中交互并最大化长期回报”的序列决策问题。

---

## 一句话总结

这组脚本覆盖了从监督学习、序列建模、目标检测、表示学习到强化学习的一条入门实践路线，适合作为边学边跑的教学型代码仓库。
