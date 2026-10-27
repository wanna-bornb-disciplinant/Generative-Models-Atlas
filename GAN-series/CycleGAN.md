# CycleGAN

## 发表情况
- Zhu, J. Y., Park, T., Isola, P., & Efros, A. A. (2017, October). Unpaired image-to-image translation using cycle-consistent adversarial networks. In 2017 IEEE international conference on computer vision (ICCV) (pp. 2242-2251). Ieee.
  - ICCV 2017
- 主要作者：Jun-Yan Zhu，Taesung Park，Phillip Isola，Alexei A. Efros
- 截止2026.10 cite：34278+
- 无配对图像翻译开山之作，循环一致性损失成为无监督跨域转换的经典思路，影响力溢出 GAN 领域，广泛用于医学、遥感等跨域任务
- [原文链接](https://www.researchgate.net/profile/Jun-Yan-Zhu-2/publication/315710637_Unpaired_Image-to-Image_Translation_using_Cycle-Consistent_Adversarial_Networks/links/5c18207a299bf139c7606041/Unpaired-Image-to-Image-Translation-using-Cycle-Consistent-Adversarial-Networks.pdf?__cf_chl_tk=Pme5weEVwTz8hxYrw8TlQQJUK.M24Fi.ed49PTkZv9w-1793088568-1.0.1.1-vNVhtaCEferKK8wsaRH0kzXlkodIsBZMfNYqhs49UPk)

## 摘要解读：
- CycleGAN希望解决的问题是Image-to-Image Translation（图像到图像转换）

- 在 CycleGAN 之前，pix2pix(也是junyan zhu的工作) 已经能够利用条件 GAN 实现图像转换，但CycleGAN 的研究目标，就是利用这两个彼此独立的数据集合，学习具有意义的映射 $G:X\to Y$

## 背景说明：
- 原始GAN的优化目标只约束输出分布，不约束输入输出之间的关系
- 将GAN最开始输入的随机噪声z替换为源域图像x之后，原始GAN的优化目标是：

$$
\min_G\max_D V(D,G)=
\mathbb E_{y\sim p_Y}[\log D(y)]+
\mathbb E_{x\sim p_X}
[\log(1-D(G(x)))]
$$
- 但在上述优化目标中，对抗训练得到的映射并不一定符合语义
  - 一个网络即使能够输出与真实斑马分布非常接近的图像，也不能保证：输入一匹站立的马，输出站立的斑马；输入一匹奔跑的马，输出奔跑的斑马；输入图像中的背景、姿势和空间结构能够保留
  - 因为判别器只关心生成的图像是否属于目标域，而不一定关心它是否忠实于输入图像

- 这引出了图像转换中的一个关键问题：分布匹配（Distribution Matching）与语义对应（Semantic Correspondence）是两回事

- CycleGAN通过Cycle Consistency 改善这个问题
  - CycleGAN 引入另一个映射：

$$
F:Y\rightarrow X
$$

- 具体来说，对于某张输入图像 $x$，先执行前向转换，再执行反向转换，要求最后转换的结果与输入相似
- 这个要求意味着，前向转换 $G$ 不能随意丢弃大量输入信息，否则反向映射 $F$ 就难以重建原始图像
- 因此，循环一致性可以在一定程度上抑制多对一塌缩；但同时，循环一致性只能约束映射的可恢复性，并不能保证映射具有正确的语义

## 具体分析：
- 具体分析可以参照原文的第三部分

- CycleGAN 的模型包含四个独立网络：
- 生成器 $G:X\to Y$
- 生成器 $F:Y\to X$
- 判别器 $D_Y$：判断图像是否属于真实目标域 $Y$
- 判别器 $D_X$：判断图像是否属于真实源域 $X$

- 具体的训练方式可以参照原文的图3
![https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/gan/4.png](https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/gan/4.png)
- 上述图中的优化目标可以分为三部分：
  - 第一部分：Adversarial Loss
    - 对前向映射： $G:X\rightarrow Y$，使用判别器 $D_Y$，对抗损失为（反向映射也是类似的，反向映射不再展开）：

$$
\begin{aligned}
\mathcal L_{\mathrm{GAN}}(G,D_Y)
={}&
\mathbb E_{y\sim p_Y}
[\log D_Y(y)]\\
&+
\mathbb E_{x\sim p_X}
[\log(1-D_Y(G(x)))]
\end{aligned}
$$
  - 上述的第一部分loss实际上完成的就是数据分布匹配
  - 第二部分：Cycle Consistency Loss
    - CycleGAN 最核心的设计是（和前面的对抗损失一样，也是分为前向和反向两部分）：

$$
\begin{aligned}
\mathcal L_{\mathrm{cyc}}(G,F)
={}&
\mathbb E_{x\sim p_X}
\left[\|F(G(x))-x\|_1\right]\\
&+
\mathbb E_{y\sim p_Y}
\left[\|G(F(y))-y\|_1\right]
\end{aligned}
$$
  - 第二部分的loss使得以下结论可以成立：**CycleGAN 是一种以目标图像域作为中间表示空间、并通过对抗学习约束中间表示分布的双向自编码结构**
  - 第三部分：Full Objective
    - 作者将两项对抗损失和循环一致性损失合并，使用循环一致性损失权重调节两部分loss的比例

- 实际训练时，作者将原始 GAN 的对数对抗损失替换为 Least-Squares GAN（LSGAN）形式，以改善训练稳定性
- 对于生成器 $G$，实际使用：

$$
\mathcal L_G^{\mathrm{adv}}=
\mathbb E_{x\sim p_X}
[(D_Y(G(x))-1)^2]
$$
- 对于判别器：

$$
\begin{aligned}
\mathcal L_{D_Y}
={}&
\mathbb E_{y\sim p_Y}[(D_Y(y)-1)^2]\\
&+ \mathbb E_{x\sim p_X}[D_Y(G(x))^2]
\end{aligned}
$$
- 这里的核心变化是，将二元交叉熵形式换成平方误差形式，这样一来，即使生成样本已经被判别器识别为假样本，生成器仍能通过平方误差获得用于调整输出的梯度

- 原文的Section5.2部分，作者为部分任务引入了额外的 Identity Mapping Loss：

$$
\begin{aligned}
\mathcal L_{\mathrm{id}}(G,F)
={}&
\mathbb E_{y\sim p_Y}
[\|G(y)-y\|_1]\\
&+
\mathbb E_{x\sim p_X}
[\|F(x)-x\|_1]
\end{aligned}
$$
- 含义是：如果输入已经属于目标域，就不应该进行没有必要的修改；Identity Loss 就提供了额外的归纳偏置，鼓励模型减少不必要的改变

## 实现细节：
- CycleGAN 使用基于 Johnson 等人图像转换网络的 ResNet Generator
  - 这里可以和StyleGAN的实现做一个对比，CycleGAN 直接使用了适合保留输入空间结构的编码—残差处理—解码结构，原因是CycleGAN解决的是Image-to-Image Translation，本身输入的维度和原始的GAN就已经不同了

- CycleGAN 没有使用仅输出一个整图真假分数的判别器，而是采用 PatchGAN。PatchGAN 为多个局部感受野区域输出真假判断，再聚合这些判断形成对抗损失

- 文章进行了消融实验验证了几个loss的不可或缺性，最终说明：GAN + Two Cycles	通常获得更完整的跨域映射约束
- 但关于这部分，更合理更严谨的说法是：完整的双向循环目标在整体实验中表现出更强的稳定性与适用性，但它并不是能够解决全部语义对应问题的理论保证。

## 站在现在的角度，回看这篇文章
- 如果从生成模型发展的角度看，CycleGAN 的价值不主要在于它使用了 GAN，而在于它提供了这样一种思想：当我们无法获得样本级监督时，是否可以通过分布约束和结构一致性构造训练信号

- 但CycleGAN的问题也很明显，即Cycle Loss 衡量的是信息能否被逆映射恢复，而不是信息是否以合理的语义形式存在

- GAN 可以不仅仅作为生成器，而是作为**分布约束工具**。只要我们能够从目标分布采样，就可以使用判别器提供学习信号。

- 实际上，CycleGAN完成的任务和WGAN、DCGAN、StyleGAN都不相同，后面的几种侧重于研究GAN类生成模型的训练目标和稳定性、提升图像生成质量，同时一些工作还会将工作重点放在如何解耦和分级潜变量上，这也类似是图像反演（Image Inversion）领域的早期工作，但CycleGAN侧重于条件跨域映射

## 参考文章和博客，感谢他们的分析解读
