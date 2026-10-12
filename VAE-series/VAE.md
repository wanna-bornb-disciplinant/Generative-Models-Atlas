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
\mathcal L =
\mathbb E_{q_\phi(z\mid x)}
[\log p_\theta(x\mid z)] - D_{\mathrm{KL}}
\left(q_\phi(z\mid x)\|p(z)\right)
$$
- 上述优化目标，一部分是要求潜变量经过解码器后能重建输入，一部分是要求潜空间服从某个指定的先验分布；这类似于一个重建项+先验正则项，截止目前，理论部分就全部搭建完毕，剩下的就是具体实现可训练的部分
- VAE希望联合优化两部分参数，其实也就是联合优化Encoder和Decoder，这一关系可以参考原文的图1：
![fig1](./images/vae/1.png)

## 实现细节：
- ELBO中的重建项依赖参数 $\phi$ 。我们不仅要优化生成模型参数，还要优化推断模型参数；因此一个很关键的问题是如何对 $\phi$ 进行求导计算梯度
- VAE最开始尝试用score function的思想解决采样无法梯度回传的问题，大致思路就是本身是对f(z)的z求导，但z是采样得到的，但z的分布又和Encoder的参数有关，梯度回传就断了，score function的解决方式是将求导转移到分布本身的log概率上：

$$
\begin{aligned}
\nabla_\phi \mathbb{E}[f(z)]
&= \nabla_\phi \int f(z) q_\phi(z) dz \\
&= \int f(z)\cdot \nabla_\phi q_\phi(z) dz \\
&= \int f(z)\cdot q_\phi(z)\cdot \frac{\nabla_\phi q_\phi(z)}{q_\phi(z)} dz \\
&= \int f(z)\, q_\phi(z)\cdot \nabla_\phi \log q_\phi(z) dz \\
&= \mathbb{E}_{q_\phi(z)}\Big[ f(z)\cdot \nabla_\phi \log q_\phi(z) \Big]
\end{aligned}\)
$$
- 上述公式最大的问题在于方差不可控，两项相乘会放大随机波动，导致训练震荡很难收敛
- 因此，VAE考虑**重参数化**解决这一关键问题，很多解读会将重参数化一笔带过，但其实这才是VAE的关键贡献
- 重参数化的关键公式是：

$$
z=g_\phi(\epsilon,x),\quad \epsilon\sim p(\epsilon)
$$
- 其中辅助随机变量 $\epsilon$ 的分布不依赖于 $\phi$ ，而函数 $g_\phi$ 才包含学习函数，核心贡献是：将随机期望改写成关于参数可微的期望，从而能够用蒙特卡洛样本估计目标及其梯度
- 进一步，VAE提出 **SGVB(Stochastic Gradient Variational Bayes)**，这是为了在ELBO的目标下方便计算梯度和随机估计的具体方法，VAE or AEVB是利用这些方法训练模型的算法框架
- SGVB的第一种形式就是回归最简单的ELBO形式，利用蒙特卡洛和重参数化改写优化目标，即：

$$
\widetilde{\mathcal L}^{A}(\theta,\phi;x) =
\frac{1}{L}
\sum_{l=1}^{L}
\left[
\log p_\theta(x,z^{(l)}) -
\log q_\phi(z^{(l)}\mid x)
\right]
$$
- 进一步的，如果KL散度可以单独计算，可以将SGVB写成第二种形式：

$$
\widetilde{\mathcal L}^{B}(\theta,\phi;x) =
-D_{\mathrm{KL}}
\left(
q_\phi(z\mid x)\|p(z)
\right)+
\frac{1}{L}
\sum_{l=1}^{L}
\log p_\theta(x\mid z^{(l)})
$$
- 上述的关键就是KL散度要方便计算，这就牵涉到VAE的实际设计，即z的先验分布定义
- 作者选择标准多元高斯先验，这个选择有三个好处：采样容易；数学表达简单；当我们假定近似后验也是一个高斯分布时，上述KL散度可以解析计算，具体的解析计算结果就可以参照其他关于VAE的解读文章，这里不再展开(高斯分布之间KL散度的公式推导)
- 到这里，VAE就彻底搭建完毕，但跳脱出数学推导，VAE的计算非常简单，**向Encoder输入x，Encoder输出预测后验分布的均值和方差，通过重参数化采样，计算得到采样的z，输入Decoder，希望他预测原始x，通过loss监督网络同时学习两部分的参数**
- 实际训练过程中，单个样本只采样一次潜变量

## 站在现在的角度，回看这篇文章
- 最大化 ELBO，不等于最大化真实数据似然
- 为什么近似后验一定要是对角高斯
- **KL 正则项可能让潜变量失去作用**,当解码器足够强时，它可能仅凭自身参数就很好地拟合数据，而不需要使用z。此时，Encoder可能学成一个完全的标准高斯模型，这被称为**后验坍塌**
- 标准 VAE 的图像经常显得模糊，这是一个开放问题，后续有很多工作尝试探讨和解决，这本身也和VAE的工程实现有关系

## 参考文章和博客，感谢他们的分析解读
- [机器学习方法—优雅的模型（一）：变分自编码器（VAE）](https://zhuanlan.zhihu.com/p/348498294)
