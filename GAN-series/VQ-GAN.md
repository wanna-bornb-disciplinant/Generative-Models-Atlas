# VQ-GAN

## 发表情况
- Esser, P., Rombach, R., & Ommer, B. (2021). Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 12873-12883).
  - CVPR 2021 oral
- 主要作者：Patrick Esser, Robin Rombach, Björn Ommer
  - 海德堡大学
- 截止2026.10 cite：5715+
- 在 VQ-VAE 基础上，**增加 patch 判别器 + 感知损失**，解决高压缩率下重建模糊问题
  - **从像素级或传统潜变量生成，走向高质量压缩表征驱动生成的重要桥梁**
- [原文链接](https://proceedings.neurips.cc/paper_files/paper/2017/file/7a98af17e63a0ac09ce2e96d03992fbc-Paper.pdf)

## 摘要解读：
- 从标题上看，虽然VQ-GAN延续了VQ-VAE的思路，但更重要的点在于如何使用Transformer作为自回归码本生成工具

- CVPR的格式下，文章的图1开门见山地说明了VQ-GAN希望解决的场景，即高分辨率的图像生成
- 这个任务场景可以从摘要部分解读得到：
  - 图像中的低层局部结构适合通过 CNN 建模，而高层视觉组成关系需要更灵活的全局建模机制
  - Transformer 旨在学习序列数据中的长程交互，与卷积神经网络（CNN）不同，Transformer 不包含偏向局部交互的归纳偏置（inductive bias）；这赋予了它强大的表达能力，但也导致其在处理长序列（如高分辨率图像）时，计算成本过高，难以实际应用
  - 因此，VQ-GAN不再要求一个生成模型同时承担像素级细节恢复与全局语义建模，而是让两种网络各自完成擅长的工作(通过CNN 编码 + 向量量化压缩形成离散视觉 Token，通过Transformer 学习 Token 分布，思路其实和VQ-VAE是一样的，但思考的角度不同)

## 背景说明：
- 此前的 PixelRNN、PixelCNN 到 PixelCNN++，自回归模型通过概率链式分解建模图像：

$$
p(x)=\prod_{i=1}^{N}p(x_i\mid x_{<i})
$$
- 这种设计提供了明确的似然建模目标，但图像的维度非常高，在更高维度的图像中，这种建模方式不仅让计算量规模呈指数性增长，同时还需要学习相邻像素之间大量冗余的低层统计关系
- 因此，把像素直接作为 Token，实际上浪费了 Transformer 在全局关系建模上的表达能力

- 回顾VQ-VAE，VQ-VAE提出了一个重要思想：不直接建模连续的高维图像，而是先学习离散潜变量
  - 推理生成时，可以使用 PixelCNN 学习离散潜变量的概率分布
  - VQ-VAE 早已提出了“离散表示 + 自回归先验”的两阶段框架
  - 但VQ-VAE的问题在于：大幅压缩图像之后，能否仍然保留足够丰富的视觉信息
 
- VQ-VAE的问题集中在VAE为主的训练方式上，VAE通过像素重建误差进行训练，即：

$$
\mathcal L_{\mathrm{rec}}=\|x-\hat x\|_2^2
$$
- 平方误差并非必然导致模糊，但当细节存在多种合理重建、模型又无法准确保留所有信息时，它容易鼓励平均化的预测，这直接引出了感知损失与 GAN 的引入

- 2019 年的 VQ-VAE-2：它利用层次化离散表示改善了高质量图像生成，但其多尺度潜表示也会增加建模复杂度。VQ-GAN 则重点探索强大的第一阶段表征是否能够降低后续 Transformer 的负担

## 具体分析：
- 原文的图2说明了整体的网络和训练方式：
![https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/gan/5.png](https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/gan/5.png)

- VQ-GAN 保留了 VQ-VAE 最重要的三部分：

$$
x\xrightarrow{E}z_e\xrightarrow{Q}z_q\xrightarrow{G}\hat{x}
$$
- 其中：
  - $E$：CNN 编码器
  - $Q$：向量量化操作
  - $G$：CNN 解码器
  - $\mathcal Z=\{e_k\}_{k=1}^{K}$：可学习的离散码本
- 真正的变化不是量化操作，而是 VQ-GAN 对第一阶段的训练目标进行了重新设计
- VQ-GAN 使用感知重建损失 $\mathcal L_{\mathrm{perc}}$ 替代普通像素重建损失
- 作者引入基于图像局部区域的判别器 $D$，区分真实图像和解码器重建的图像
- 完整训练目标采用最小最大博弈形式：

$$
\min_{E,G,\mathcal Z}\max_D
\mathbb E_{x\sim p(x)}
\left[
\mathcal L_{\mathrm{VQ}}+
\lambda\mathcal L_{\mathrm{GAN}}
\right]
$$
- 其中 VQ 部分为：

$$
\begin{aligned}
\mathcal L_{\mathrm{VQ}}
={}&\mathcal L_{\mathrm{perc}}(x,\hat x)\\
&+\|\operatorname{sg}[E(x)]-z_q\|_2^2\\
&+\beta\|\operatorname{sg}[z_q]-E(x)\|_2^2
\end{aligned}
$$
- 作者选择 PatchGAN，这与论文的整体分工高度一致：作者希望CNN 编码器与解码器负责局部视觉结构
  - PatchGAN 重点检查局部区域的真实感，与“让 Transformer 专注于全局关系”的设计思想契合
 
- 为什么 VQ-GAN 能实现更强压缩
  - 在给定相同 Token 数量的情况下，普通 VQ-VAE 与 VQ-GAN 的区别可以概括为：
    - VQVAE倾向于优化像素级重建准确性；强压缩下可能损失高频视觉细节
    - VQGAN利用感知与对抗目标，在强压缩下尽量保持视觉上可信的局部结构与纹理
    - VQ-GAN 允许解码器根据学习到的图像先验，重建那些无法通过有限 Token 完整保存的细节，这和最终的损失目标有关
   
- 为什么需要Tranformer：
  - Transformer 通过自回归方式建模，训练损失是标准的负对数似然：

$$
p(s)=\prod_{i=1}^{N}p(s_i\mid s_{<i})
$$
  - 因此，二阶段单独训练建模Tranformer的时候没有引入特殊的 GAN 损失，也不需要同时与图像判别器对抗
  - 论文还提出通过条件序列实现：

$$
p(s\mid c)=\prod_i p(s_i\mid s_{<i},c)
$$
  - 当条件具有空间结构时，论文可以使用另一个 VQ-GAN 将其转化为条件 Token 序列 $r$，将 $r$ 放在目标序列之前，Transformer 在预测 $s_i$ 时能够访问条件 Token 和先前生成的图像 Token

- 但即便压缩之后，百万像素图像仍可能产生较长的 Token 序列，VQ-GAN通过Sliding attention window解决这个问题，具体可以查看原文图3：
![https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/gan/6.png](https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/gan/6.png)
- 具体来说，编码部分可以对序列进行维度缩小(二维图像阶段用卷积达到)，但实验观察发现缩小到一定程度时生成质量会发生衰减
- 因此，考虑使用Sliding attention window，它的做法其实就是训练阶段使用固定大小的图像裁剪块，使 Token 序列限制在可承受的长度；生成阶段让有限上下文窗口在更大画布上移动，以逐步产生完整图像
  - 上述做法来自于一篇COCOGAN的文章
 
## 实现细节：
- 论文的实验并不是仅仅展示生成图像，而是分别验证了三件事：
  - 在同一离散潜空间上，Transformer 是否比卷积自回归模型更有效？
  - 强压缩、高感知质量的 Codebook 是否有助于全局生成？
  - 整个两阶段框架能否达到有竞争力的图像生成质量？
 
- 实验中的重要结论包括：
  - Transformer 的上下文能力不仅取决于 Token 数量，还取决于每个 Token 表示了多少有意义的视觉上下文
  - 论文也没有证明 VQ-GAN 全面超越专门设计的 GAN，打不过BigGAN (ImageNet256*256的生成实验)
  - VQ-GAN 的竞争力并不完全在于最好的 FID，而在于它同时提供了离散表征、可重建性、概率建模与统一的条件生成接口
 
- 两阶段训练的好处在于：Transformer 不需要处理复杂的连续图像重建问题，只需要对离散类别分布建模；VQ-GAN 可以复用于不同的生成任务，而无需每次都重新训练图像编解码器

## 站在现在的角度，回看这篇文章
- VQ-GAN 并不是 GAN 路线的直接延续，VQ-GAN 并非主要在研究新的生成对抗博弈，而是将 GAN 的感知建模优势用于训练视觉 Tokenizer
- **在 VQ-GAN 中，GAN 已经从整个生成系统的核心生成机制，转变为学习高质量潜表示的一种训练工具**

- 感知真实不代表细节忠实。在科学图像、医学图像等任务中，解码器补充的纹理可能并不对应真实测量信息

- VQ-GAN的局限：
  - 离散量化引入信息瓶颈
  - 自回归采样依然存在串行瓶颈
  - VQ-GAN 第一阶段主要优化感知重建质量，而不是直接优化第二阶段 Transformer 的最终生成性能
    - **因此，最容易重建图像的 Token 表示，不一定就是最容易进行概率建模的 Token 表示，这导致Transformer的最优建模和编码解码器的最优建模有mismatch**
  - 滑动窗口无法彻底解决全局一致性
    
## 参考文章和博客，感谢他们的分析解读

