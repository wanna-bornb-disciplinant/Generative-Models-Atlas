# WGAN和WGAN-GP

## 发表情况
- Arjovsky, M., Chintala, S., & Bottou, L. (2017, July). Wasserstein generative adversarial networks. In International conference on machine learning (pp. 214-223). Pmlr.
  - ICML 2017
- 主要作者：Martin Arjovsky，Soumith Chintala，Leon Bottou
  - FAIR工作
- 截止2026.10 cite：23127+
- 用 Wasserstein 距离替代 JS 散度，从理论上解释 GAN 梯度消失问题，奠定 “距离度量视角” 研究 GAN 的主线，重塑 GAN 损失函数设计思路
- [原文链接](https://proceedings.mlr.press/v70/arjovsky17a/arjovsky17a.pdf)

- Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., & Courville, A. (2017). Improved training of wasserstein gans. Advances in neural information processing systems, 30.
  - NIPS 2017
- 主要作者：Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, Aaron Courville
- 截止2026.10 cite：16249+
- 改进WGAN的Lipschitz约束
- [原文链接](https://proceedings.neurips.cc/paper_files/paper/2017/file/892c3b1c6dccd52936e27cbd0ff683d6-Paper.pdf)

## 摘要解读：
- 极简短的摘要内容，原始的GAN采用JS散度，这个理论上合理的目标，在实际神经网络训练时，不一定具有良好的优化性质
- 因此，WGAN通过 Wasserstein-1 距离重新定义了生成模型的优化目标，并为这一目标提供了可实现的对抗训练方法
- 文章用了一整个章节的篇幅论述了两种距离衡量方式的优劣，这是不同的写作方式

## 背景说明：
- 对于一张高维图像，数据可能集中在低维流形附近，而不是均匀分布于整个高维空间

- 这个问题同样会发生在GAN类的生成网络中，生成数据只覆盖生成器映射所能到达的区域，在通常的光滑性与非退化条件下，该分布集中在至多 100 维的低维集合上，这就引出一个问题：
  - 真实数据分布和生成数据分布可能分别集中在不同的低维集合上，它们的支撑集没有重叠，或者只有测度为零的交集
  - 此时LS的优化目标可能会失效，JS 散度只能告诉我们两个分布是否重合，却不能在这个例子中体现它们距离重合还差多少
  - Wasserstein 距离则能够连续地描述这种差异，原文的图1也描述了这种情况下的差异
 
- 原文的Example1想说明即使两个低维数据流形几何上非常接近，在 JS 散度意义下也可能没有任何变化


## 具体分析：
- Wasserstein 距离最早并不是为深度学习而提出的，而是来自最优传输理论（Optimal Transport）。它也被称为 Earth Mover's Distance（EM Distance）

- 考虑两个分布，最优传输理论希望解决的问题是：

$$
W_1(P_r,P_g)=
\inf_{\gamma\in\Pi(P_r,P_g)}
\mathbb E_{(x,y)\sim\gamma}
[\|x-y\|]
$$
- 其中 $\Pi(P_r,P_g)$ 表示边缘分布分别是 $P_r$ 和 $P_g$ 的所有联合分布。这里的 $\gamma(x,y)$ 可以理解为：选择将多少位于 $x$ 的质量与位于 $y$ 的质量配对；最佳运输尝试在所有合法的搬运方案中寻找总成本最低的方案

- 为什么Wasserstein 距离对应更好的连续性，这需要对应原文的理论1和理论2部分
- 理论1希望说明的是WGAN为生成器构造了一个在温和条件下连续且几乎处处可微的分布距离目标，主要推导的方式是通过最优传输成本、Lipschitz条件和三角不等式说明，当参数变化很小时，生成分布在 Wasserstein 意义下也只会发生有限且受控的变化
- 理论2希望说明的是在紧致数据空间中KL散度趋近于0，可以推出JS散度趋近于0，可以推出W1距离趋近于0，作者认为更弱的收敛概念反而适合训练：一个分布在参数稍微改变后，更容易被 Wasserstein 距离判断为只发生了小变化

- 但原始的W1距离在高维图像中几乎无法计算，文章通过Kantorovich–Rubinstein 对偶理论解决这个问题：

$$
W_1(P_r,P_g)=
\sup_{\|f\|_L\le1}
\left(
\mathbb E_{x\sim P_r}[f(x)]-
\mathbb E_{x\sim P_g}[f(x)]
\right)
$$
- 对偶形式中，将优化传输计划改为了优化函数；将最小化总搬运成本改为最大化真实与生成数据的平均函数值之差；将需要考虑两个分布的联合配对改为只需分别采样两个分布；同时函数还可以通过网络近似

- 原始 GAN 使用神经网络 $D$ 学习样本的真假分类，而WGAN 使用神经网络 $f_w$ 近似对偶函数（这是判别器的训练目标）：

$$
\max_w
\left[
\mathbb E_{x\sim P_r}[f_w(x)]-
\mathbb E_{z\sim P_z}[f_w(G_\theta(z))]
\right]
$$
- 上述公式中强制要求判别器是1-Lipschitz，实现方式是后续工作的改进点

- 从最优 Critic 到生成器梯度：
- 先固定生成器参数 \(\theta\)，优化 Critic：

$$
w^*(\theta)=\arg\max_w F(\theta,w)
$$
- 计算生成器梯度时，将最优Critic视为固定函数，对生成器参数求偏导：

$$
\nabla_\theta W_1(P_r,P_\theta)=-
\mathbb E_{z\sim P_z}
\left[
\nabla_\theta f^*(G_\theta(z))
\right]
$$
- 按照链式法则展开：

$$
\nabla_\theta W_1=-
\mathbb E_z
\left[
J_{G_\theta}(z)^\top
\nabla_x f^*(x)
\big|_{x=G_\theta(z)}
\right]
$$
- 与原始 GAN 相比，这里的关键优势是：最优 Critic 不需要退化成概率为 0 或 1 的分类结果；它可以继续对生成样本提供一个具有空间变化信息的函数值和梯度

## 实现细节：
- 从实现角度，WGAN的核心改变是：
  - 将判别器变成 Critic，不再进行概率分类
  - 去掉判别器最后的 Sigmoid
  - 替换对抗损失函数
  - 对 Critic 施加 Lipschitz 约束
  - 在每次更新 Generator 之前，多次更新 Critic（5次）
 
- 最开始版本的论文通过Weight Clipping（权重裁剪）实现Lipschitz 约束，即通过将参数限制在有界集合内，使网络具有统一的有限 Lipschitz 上界(-0.01和0.01)

- 实验部分：
  - 实验一：八个高斯混合分布，这种分布特别适合观察 Mode Collapse(图2)
  - 实验二：最优判别器与 Critic 梯度 Figure 3
  - 实验三：Loss 能否反映生成质量 Figure 4、5
  - 实验四：不同网络架构下的稳定性

## 站在现在的角度，回看这篇文章
- 原始 WGAN 用参数裁剪实现 Lipschitz 约束，虽然方法简单，但存在比较明显的问题

- WGAN-GP 的基本思想是：既然我们希望 Critic 满足 1-Lipschitz 条件，那么是否能够直接约束其输入梯度？

- WGAN-GP 提出了如下损失：

$$
\begin{aligned}
\mathcal L_C^{GP}
={}&
\mathbb E_{x\sim P_g}[f_w(x)]-
\mathbb E_{x\sim P_r}[f_w(x)]\\
&+
\lambda
\mathbb E_{\hat x\sim P_{\hat x}}
\left[
\left(
\|\nabla_{\hat x}f_w(\hat x)\|_2-1
\right)^2
\right]
\end{aligned}
$$
- 其中插值样本：

$$
\hat x=
\epsilon x_r+(1-\epsilon)x_g
$$

$$
\epsilon\sim U[0,1]
$$
- WGAN-GP 并不是严格施加全局 1-Lipschitz 约束：首先，原始的 1-Lipschitz 条件要求梯度范数不超过 1，而 WGAN-GP 惩罚其偏离 1；其次，它只在采样得到的插值点上施加软约束

## 参考文章和博客，感谢他们的分析解读


