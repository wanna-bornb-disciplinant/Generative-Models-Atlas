# PixelCNN++

## 发表情况
- Salimans, T., Karpathy, A., Chen, X., & Kingma, D. P. (2017). Pixelcnn++: Improving the pixelcnn with discretized logistic mixture likelihood and other modifications. arXiv preprint arXiv:1701.05517.
  - ICLR 2017
- 主要作者：Tim Salimans, Andrej Karpathy, Xi Chen, Diederik P. Kingma
- 截止2026.10 cite: 1477+
- OpenAI, VAE团队follow PixelCNN的范式的改进之作
- [原文链接](https://arxiv.org/pdf/1701.05517)

## 摘要解读
- 摘要简明扼要，说明本文就是针对PixelCNN的改进，主要集中在五点：
  - 离散 Logistic 混合似然
  - 按完整像素建模
  - 下采样
  - 长距离跳跃连接
  - Dropout

- 论文在 CIFAR-10 上将测试集负对数似然降至 2.92 bits per sub-pixel，优于论文表 1 所列的原始 PixelCNN、Gated PixelCNN 和 PixelRNN 等模型

## 背景说明
- [PixelCNN or PixelRNN 说明](https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/AutoRegressive-series/PixelCNN.md)
- [Gated PixelCNN 说明](https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/AutoRegressive-series/Gated_pixelCNN.md)
- 文章分析旧版PixelCNN的三个瓶颈：
  - 像素的概率分布建模成本过高：传统实现为每个颜色通道输出 256 个类别的 Softmax 概率。对一个 RGB 像素，三个通道就需要 768 个类别概率；如果还要建模通道间的依赖，输出参数会进一步增加。问题不只是输出维度大，还在于这种离散分类没有直接利用像素值的有序关系。例如，127 和 128 的数值很接近，但在 Softmax 中，它们是两个独立的类别。模型需要从训练数据中自行学习它们之间的统计联系。
  - 局部卷积不容易高效建模全局结构：小卷积核适合捕获邻近像素的纹理，但图像中的物体形状、背景布局和远距离空间关系，需要更大的感受野。原始 PixelCNN 可以通过增加网络深度扩大感受野，但这会增加计算成本；使用膨胀卷积也能扩大感受野，却不一定是最节省计算的方案。
  - 更大的网络不一定带来更好的生成结果：过拟合的 PixelCNN 不仅泛化性能差，生成图像的感知质量也可能不好。因此，PixelCNN++ 的任务不是单纯扩大网络，而是同时改进概率分布、特征传递和正则化

## 具体分析
- PixelCN++的核心创新就是**离散 Logistic 混合似然**
  - 传统 Softmax 输出具有很强的表达能力，可以对每个像素值单独分配概率。但是，它有两个问题：每个位置都需要预测大量输出参数；相邻像素值之间没有被显式赋予连续或平滑的关系
  - PixelCNN++ 希望引入一种更有结构的概率分布：既能够表示多峰的像素分布，又能够利用数值的邻近关系

- 论文的建模思路是，先定义一个连续的潜在颜色强度 $\nu$，再将它四舍五入到 8-bit 像素值 $x$:

$$
\nu\sim Logistic(\mu,s)
$$
  - $\mu$：分布的位置参数，决定分布中心； $s>0$：尺度参数，决定分布的宽窄。
  - Logistic 分布的累积分布函数为：

$$
F(\nu;\mu,s)=
\sigma\left(\frac{\nu-\mu}{s}\right)
$$
  - 其中 $\sigma(t)=1/(1+e^{-t})$
  - 注意，最终观测到的是离散整数，而不是连续的 $\nu$。所以不能直接把连续概率密度 $f(x)$ 当成离散像素值的概率。正确的做法是计算能够舍入到 $x$ 的整个区间内的概率质量。
  - 对普通的内部像素值 $1\le x\le254$，有：

$$
\begin{aligned}
P(x\mid\mu,s)
={}&F(x+0.5;\mu,s)\\
&-F(x-0.5;\mu,s)
\end{aligned}
$$
- 进一步，单个 Logistic 分布通常只有一个主要峰值，而给定图像上下文后，某个像素的条件分布可能具有多个峰。例如，物体边界上的某个位置可能对应不同的颜色；或者背景中存在多种合理的像素值。

- 因此，PixelCNN++ 使用 $K$ 个 Logistic 分布的混合：

$$
p(\nu\mid c)=
\sum_{k=1}^{K}
\pi_k(c)
Logistic
\big(\nu;\mu_k(c),s_k(c)\big)
$$
- 论文指出，实验中使用约 5 个混合分量就能够较好地建模条件像素分布。

- PixelCNN++的第二个主要改进：将空间上的自回归顺序定义在完整像素上，但并没有忽略 RGB 内部的依赖关系。对于空间位置 \((i,j)\)，令：

$$
\mathbf{x}_{i,j}=(r_{i,j},g_{i,j},b_{i,j})
$$

- 其联合条件概率可以分解为：

$$
\begin{aligned}
&p(r,g,b\mid C)\\
&=p(r\mid C)\,
p(g\mid r,C)\,
p(b\mid r,g,C)
\end{aligned}
$$
- 这意味着：先预测红色通道；再根据红色通道预测绿色通道；最后根据红色和绿色通道预测蓝色通道。每个通道都使用离散 Logistic 混合分布，但后续通道的混合分量位置参数会受到前面通道的影响。

$$
\begin{aligned}
\mu_g=\mu_g^{(0)}+\alpha r\\
\mu_b=\mu_b^{(0)}+\beta r+\gamma g
\end{aligned}
$$
- 这种设计的关键是：简化空间上的自回归结构，同时保留像素内部的颜色依赖

- 下采样、长距离跳跃连接和Dropout不展开

## 实现细节
- 本文做了比较全面的消融实验，本身也是一个类似技术报告一样的改进工作

## 站在现在的角度，回看这篇文章
- PixelCNN++ 说明，输出层的概率分布设计本身就是生成模型的重要组成部分
- PixelCNN++ 的实验显示，**训练似然、测试似然和生成图像的感知质量并不总是一致**

## 参考文章和博客，感谢他们的分析解读
- [详解自回归模型 PixelCNN 大家族](https://zhuanlan.zhihu.com/p/632209862)

