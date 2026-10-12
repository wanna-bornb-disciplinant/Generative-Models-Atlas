# VAE

## 发表情况
- Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114.
  - ICLR 2014
- 主要作者：Diederik P. Kingma，Max Welling
- 截止2026.10 cite：60000+
- 2024年 首届 ICLR 时间检验奖
- [原文链接](https://arxiv.org/pdf/1312.6114)

## 摘要解读：
- 首先作者想讨论的是具有明确概率生成过程的模型，即建模概率分布，这区别于GAN等方式
- VAE中作者提出两个问题：
  - 针对建模高维数据x的分布，更好的路径是建模更低维度的隐变量z，但真实后验 $p_\theta(z\mid x)$ 可能涉及难以计算的积分，因而无法直接获得
  - 即使某种近似推断算法可以处理单个样本，如果每个样本都需要执行昂贵的独立迭代，整体训练仍然可能很慢
- VAE中作者强调了两个贡献：
  - 对变分下界进行重参数化，得到可以使用标准随机梯度方法优化的估计器
  - 对具有独立同分布观测、每个样本有连续潜变量的数据集，训练一个近似推断模型，使后验推断更高效

## 背景说明：
- 假设每个观测样本x都由一个没有被直接观察到的变量z产生：
        
$$
z\sim p(z),\qquad x\sim p_\theta(x\mid z)
$$
- 模型对观测数据的边缘似然为：

$$
p_\theta(x)=\int p_\theta(x\mid z)p(z)\,dz
$$    
- 训练生成模型时，一个自然的目标是最大化训练数据的对数似然：
        
$$
\max_\theta \sum_{i=1}^{N}\log p_\theta(x_i)
$$
- 上述逻辑的问题在于：连续潜变量的积分通常难以精确计算。因此，即使生成过程定义得很清楚，我们也可能无法直接计算训练目标及其梯度
- 训练目标中，实际上定义了先验分布p(z)和似然分布 $p_\theta(x\mid z)$ ，这是**生成方向**，即先采样z，再根据似然分布生成x；但在训练时需要解决的是**推断的问题**，即我们手里有数据x，推断它可能对应什么样的z，这时就需要后验分布 $p_\theta(z\mid x)$
- 似然分布和后验分布是如何建立起联系的：
  - 生成模型训练完成后，只需要先采样z，然后通过似然分布生成新数据，这个过程不需要Encoder，也不需要后验分布，因此后验分布并不是生成所需的必要环节
  - 但生成模型的训练依赖于后验分布，上述生成过程需要$p_\theta(x\mid z)$，其中 $\theta$ 需要通过训练得到的，训练目标就可以定为最大化训练数据的对数似然
  - 回顾上面的对数似然，因为decoder network非线性的缘故，这个关于z的积分几乎是无法计算的，原因在于如果假定z是一个满足多维高斯分布的潜变量，如果decoder是一个线性网络，仍然可以通过高斯分布的良好性质求得最终的 $p_\theta(x)$ ，但decoder非线性就导致无法直接计算
  - 基于此，**Fisher's identity**构建起后验分布和数据的边缘分布之间的关系，起因是即使 $log p_\theta(x)$ 无法计算，我们只要能计算这个loss的梯度进行梯度回传，仍然可以训练网络，则有：

$$
  \begin{aligned}
  \nabla_\theta\log p_\theta(x)
  &=\frac{1}{p_\theta(x)}
  \nabla_\theta p_\theta(x)\\
  &=\frac{1}{p_\theta(x)}
  \nabla_\theta\int p_\theta(x,z)dz\\
  &=\int
  \frac{p_\theta(x,z)}{p_\theta(x)}
  \nabla_\theta\log p_\theta(x,z)dz\\
  \nabla_\theta\log p_\theta(x)
  &=
  \mathbb E_{p_\theta(z\mid x)}
  \left[
  \nabla_\theta\log p_\theta(x,z)
  \right]
  \end{aligned}
$$ 
  - 上述公式的含义在于：我们没必要去遍历整个z空间做积分，只需要关注真正和x相关的z做蒙特卡洛采样一样可以获得较好的结果，但相应的又会引入一个新的问题，即新的优化目标中仍然有积分，仍然有不可计算的后验分布，到这一步为止，我们基本理清了后验分布和边缘分布的关系
- 变分贝叶斯（Variational Bayes, VB）试图使用一个可处理的分布 q(z)，去近似难以计算的后验分布 $p_\theta(z\mid x)$：

$$
q^*=\arg\min_q
D_{\mathrm{KL}}\big(q(z)\|p_\theta(z\mid x)\big)
$$
- 具体来说，VB尝试构建了一个ELBO的优化目标，他和真实的优化目标的差距只有一个KL散度，并且这个散度建立在真实后验分布和引入的分布q(z)之间：

$$
\begin{aligned}
\log p_\theta(X)
&= \int_z q_\phi(z\mid X)\log p_\theta(X)dz\\
&= \int_z q_\phi(z\mid X)
\log\frac{p_\theta(X,z)}{p_\theta(z\mid X)}dz\\
&= \int_z q_\phi(z\mid X)
\log\left(
\frac{p_\theta(X,z)}{q_\phi(z\mid X)}
\cdot
\frac{q_\phi(z\mid X)}{p_\theta(z\mid X)}
\right)dz\\
&= \int_z q_\phi(z\mid X)
\log\frac{p_\theta(X,z)}{q_\phi(z\mid X)}dz
+\int_z q_\phi(z\mid X)
\log\frac{q_\phi(z\mid X)}{p_\theta(z\mid X)}dz\\
&= \ell(p_\theta,q_\phi)
+D_{\mathrm{KL}}(q_\phi,p_\theta)\\
&\geq \ell(p_\theta,q_\phi)
&& \text{也可以用Jensen不等式解释}.
\end{aligned}
$$
- 上面两个公式，一个利用VB构建了一个ELBO的优化目标，一个用Fisher's identity构建了后验分布的期望，如果在后面的KL散度为0的情况下，即引入分布和后验分布完全相同，前面的部分求导也会完全相同，这说明了：
  - 我们从VB的角度分析了优化目标，最优情况很自然地和拟合后验分布的目标重合了
  - 其实本身从ELBO的角度出发，我们也应该在Encoder不变的情况下，寻找让KL散度最小的情况，这自然地让引入分布和后验分布相同
  - 同时从Fisher's identity出发，拟合的比较好的引入分布，也可以完成替代完整积分z空间的任务
- 上述结论都是之前的理论，VAE的真正贡献在于通过AEVB(Auto Encoder VB)的方式让**已有的变分推断目标能够在复杂神经网络生成模型中得到高效、可扩展的优化**

## 具体分析：
- 上述理论分析最终落脚在：
  - **VAE希望通过 $\theta$ 学习数据的概率生成过程**
  - **VAE希望通过 $p_\theta(z\mid x)$ 对观测数据的潜在成因进行推断**
  - **VAE希望通过数据分布建模支持去噪、补全、超分辨率等任务，这是后续很多分析VAE的文章经常会忽略的原文目标**
- 将ELBO视为网络训练目标，自然就会得到：

$$
\mathcal L(\theta,\phi;x)=
\mathbb E_{q_\phi(z\mid x)}
\left[
\log p_\theta(x,z)-\log q_\phi(z\mid x)
\right]
$$
- 进一步简化训练目标，有：

$$
\mathcal L
=
\mathbb E_{q_\phi(z\mid x)}
[\log p_\theta(x\mid z)]
-
D_{\mathrm{KL}}
\left(q_\phi(z\mid x)\|p(z)\right)
$$
- 上述优化目标，一部分是要求潜变量经过解码器后能重建输入，一部分是要求潜空间服从某个指定的先验分布；这类似于一个重建项+先验正则项，截止目前，理论部分就全部搭建完毕，剩下的就是具体实现可训练的部分
- VAE希望联合优化两部分参数，其实也就是联合优化Encoder和Decoder，这一关系可以参考原文的图1：
![fig1](./images/vae/1.png)

## 实现细节：
- 
  - 重参数化解决反向传播中需要采样z的问题
  - 对于一个x采样几个z的问题，实际上一般都只会采样一次
  - 最终输出的x的概率分布，但一般图简单好实现，最终就拿输出的均值作为最后的x输出
  - 优化loss继续化简，将decoder输出的指定z下的方差全设置为1/2，这样对期望部分的loss计算进一步化简

- VAE如果硬要从原理上总结一下，我认为是：**本身对x的建模跟变分和MAP都无关，变分和MAP都是体现在对z的后验概率建模的，z的后验概率是如何参与到x的概率分布建模的呢，是因为我们不想大量的去采样z，我们尽量去寻找和特定x关联性比较强的z**

## 参考文章和博客，感谢他们的分析解读
- [机器学习方法—优雅的模型（一）：变分自编码器（VAE）](https://zhuanlan.zhihu.com/p/348498294)
