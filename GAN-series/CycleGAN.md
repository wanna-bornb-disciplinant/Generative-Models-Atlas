# GAN

## 发表情况
- Goodfellow, I. J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ... & Bengio, Y. (2014). Generative adversarial nets. Advances in neural information processing systems, 27.
  - NIPS 2014
- 主要作者：Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio
- 截止2026.10 cite：99363+
- 提出**对抗训练极小极大博弈框架**，催生了庞大的GAN变体谱系
- NeurIPS 2024 Test of Time Award（时间检验奖）
- [原文链接](https://proceedings.neurips.cc/paper_files/paper/2014/file/f033ed80deb0234979a61f95710dbe25-Paper.pdf)

## 摘要解读：
- GAN的摘要简明扼要地说明了文章和算法的内容
  - 用两个网络共同学习真实数据分布
  - 从概率建模转变为极小极大博弈
  - 摘要给出了理论上的理想平衡点

## 背景说明：
- 在GAN之前，很多生成模型的核心问题是：如何构造并学习一个能够描述真实数据的概率分布？
- 但GAN提出了另外一种思路：
  - 我们不一定需要知道生成数据的显式概率密度，也不一定需要计算似然；只要能通过对抗过程，让模型生成的数据与真实数据在统计上无法区分，就有可能完成生成分布的学习
- 上述思路被称为**隐式生成模型（Implicit Generative Model）**

- GAN 的核心问题是：
  - 判别器能够提供什么样的分布差异信息
  - 这种信息能否变成生成器的有效梯度
  - 两个网络交替优化最终能否达到平衡

## 具体分析：
- 文章的introduction以概率建模和判别式模型入手，对于一个二分类模型，设标签为：

$$
y=
\begin{cases}
1,& x\text{ 来自真实数据}\\
0,& x\text{ 来自生成器}
\end{cases}
$$
- 对于一个二分类模型，其伯努利似然为：

$$
p(y\mid x)=
D(x)^y(1-D(x))^{1-y}
$$
- 上述似然去对数，结果为：

$$
\log p(y\mid x)=
y\log D(x)+(1-y)\log(1-D(x))
$$
- 当真实样本与生成样本以相等权重参与训练时，最大化二分类对数似然（忽略共同常数因子）就得到：

$$
\mathbb E_{p_{\mathrm{data}}}[\log D(x)]+
\mathbb E_{p_g}[\log(1-D(x))]
$$

- 接下来需要解决的问题是：当判别器已经最优时，它究竟学到了什么？固定生成器 $G$，只优化判别器 $D$，将原始损失合并可以写成：

$$
V(D,G)=
\int\left[p_{\mathrm{data}}(x)\log D(x)+
p_g(x)\log(1-D(x))
\right]dx
$$

- 定义：

$$
a=p_{\mathrm{data}}(x),\qquad b=p_g(x)
$$
- 于是问题变成：

$$
\max_{0\leq d\leq 1}
a\log d+b\log(1-d)
$$

- 令 $d$的相关导数为0，最终可得到：

$$
D_G^*(x)=
\frac{p_{\mathrm{data}}(x)}
{p_{\mathrm{data}}(x)+p_g(x)}
$$
- 上述结论最终说明：**最优判别器实际上估计的是真实分布与生成分布之间的密度比信息**，GAN 判别器的本质不是简单地判断真假，而是在学习两个概率分布之间的相对密度关系

- 进一步我们分析生成器的优化目标，定义在最优判别器下的生成器目标：

$$
C(G)=\max_D V(D,G)=V(D_G^*,G)
$$

$$
\begin{aligned}
C(G)=&
\int p_{\mathrm{data}}(x)
\log\frac{p_{\mathrm{data}}(x)}
{p_{\mathrm{data}}(x)+p_g(x)}dx\\
&+
\int p_g(x)
\log\frac{p_g(x)}
{p_{\mathrm{data}}(x)+p_g(x)}dx
\end{aligned}
$$
- 构造一个混合分布：

$$
m(x)=
\frac{p_{\mathrm{data}}(x)+p_g(x)}2
$$
- 将优化目标可写为：

$$
\begin{aligned}
C(G)=&
D_{\mathrm{KL}}(p_{\mathrm{data}}\|m)\\
&+D_{\mathrm{KL}}(p_g\|m)-2\log2
\end{aligned}
$$
- 上述优化目标和Jensen–Shannon 散度形式相同，这意味着达到优化目标的最小值的条件是生成概率和原始数据概率相同，**理想情况下，原始 minimax GAN 等价于最小化真实分布和生成分布之间的 JS 散度**

## 实现细节：
- 原论文第 3 页已经指出了一个很重要的实际问题：当生成器较差时，判别器很容易识别生成样本，这可能使原始生成器目标的梯度过弱
- 为了解决这个问题，Goodfellow 在原论文中提出另一种生成器目标，不再最小化：

$$
\log(1-D(G(z)))
$$
- 而是最大化：

$$
\log D(G(z))
$$
- Non-saturating Loss 解决的是一类梯度饱和问题，并不保证 GAN 不会出现梯度消失(BCEloss实现的时候是最小化负的Non-saturating Loss)

- GAN 用可学习的判别器作为分布差异的间接优化机制，在训练中，我们并不计算显式概率，而是直接使用两个分布的样本训练一个二分类器

- 训练过程需要交替优化： $D$ 需要不断适应当前生成器所产生的假数据；而 $G$ 需要不断利用当前判别器反馈的信息来改善生成分布；如果只持续更新生成器而不更新判别器，生成器可能学到的是如何利用固定判别器的漏洞，而非真正匹配数据分布；如果判别器过强，生成器又可能接收不到有用的训练信号

- 整体的交替学习过程可以由原文的图1表示：
![https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/gan/1.png](https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/gan/1.png)

- 整体上，可以用一个简单的pytorch代码展示GAN是如何训练的：
```python
import torch
import torch.nn as nn

latent_dim = 100
image_dim = 28 * 28

G = nn.Sequential(
    nn.Linear(latent_dim, 256),
    nn.ReLU(),
    nn.Linear(256, 512),
    nn.ReLU(),
    nn.Linear(512, image_dim),
    nn.Sigmoid()
)

D = nn.Sequential(
    nn.Linear(image_dim, 512),
    nn.LeakyReLU(0.2),
    nn.Linear(512, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 1)
)

criterion = nn.BCEWithLogitsLoss()
opt_G = torch.optim.Adam(G.parameters(), lr=2e-4)
opt_D = torch.optim.Adam(D.parameters(), lr=2e-4)

def train_step(real):
    real = real.reshape(real.size(0), -1)
    batch_size = real.size(0)
    device = real.device
    G.to(device)
    D.to(device)

    real_labels = torch.ones(batch_size, 1, device=device)
    fake_labels = torch.zeros(batch_size, 1, device=device)

    # 1. Train discriminator
    z = torch.randn(batch_size, latent_dim, device=device)
    fake = G(z).detach()

    d_real = D(real)
    d_fake = D(fake)

    loss_D = (
        criterion(d_real, real_labels)
        + criterion(d_fake, fake_labels)
    )

    opt_D.zero_grad()
    loss_D.backward()
    opt_D.step()

    # 2. Train generator (non-saturating)
    z = torch.randn(batch_size, latent_dim, device=device)
    fake = G(z)

    loss_G = criterion(D(fake), real_labels)

    opt_G.zero_grad()
    loss_G.backward()
    opt_G.step()

    return loss_D.item(), loss_G.item()
```

- 作者在三个数据集上进行了实验：MNIST：手写数字；TFD：Toronto Face Database；CIFAR-10：自然图像

## 站在现在的角度，回看这篇文章
- JS散度本身具有局限性：JS 散度可以判断两个分布是否一致，但当两个分布的支撑集不重叠时，它可能无法提供关于如何移动生成分布的有效距离信息
  - 对于确定性且足够光滑的生成器，它所诱导的分布可能集中在低维流形上；真实数据和生成数据各自所在的流形在训练初期可能不重叠；在这种情况下，基于最优判别器的 JS 散度目标可能饱和

- 原文提到过这一问题：因为 GAN 的每一步训练并不是直接对 $p_g$ 做全局分布优化，而是利用有限样本上的判别器梯度改变模式，这会导致生成器可能发现某些输出特别容易欺骗当前判别器，于是越来越多不同的潜变量被映射到这些输出附近，这就是典型的模式崩塌（Mode Collapse）
- 崩塌并不是由 JS 散度饱和单独导致的：它还与非凸非凹博弈、有限容量判别器、训练动力学等因素有关

- 简单的同步梯度下降—上升仍然可能出现向外螺旋发散，而不是收敛，这是GAN训练困难的一个很大障碍，交替优化是必然选择，但也是次优选择，至少从训练的层面来说是这样

## 参考文章和博客，感谢他们的分析解读

- [GAN论文逐段精读【论文精读】-李沐-Bilibili](https://www.bilibili.com/video/BV1rb4y187vD)

