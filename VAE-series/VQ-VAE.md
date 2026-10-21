# VQ-VAE

## 发表情况
- van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). Neural Discrete Representation Learning. Advances in Neural Information Processing Systems, 30
  - NIPS 2017
- 主要作者：Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu
- 截止2026.10 cite：10348+
- 开创离散潜空间生成的主流分支；后续DALL.E1使用VQ-VAE，埋下大模型的种子
- [原文链接](https://proceedings.neurips.cc/paper_files/paper/2017/file/7a98af17e63a0ac09ce2e96d03992fbc-Paper.pdf)

## 摘要解读：
- 从标题上来看，这篇文章不从生成的角度分析，而是站在标准学习的视角将VQ-VAE看作是离散表征学习在生成任务上的一个具体应用
- 摘要中直接点名和VAE的区别：
  - 编码器不再输出用于采样的连续高斯潜变量，而是输出连续特征，再通过最近邻量化得到离散编码--离散化编码输出
  - 潜变量的先验不再局限于简单的固定分布，而是在表示学习完成后，学习一个可以捕获空间或时间依赖关系的自回归先验--可学习的先验分布

## 背景说明：
- 普通 VAE 通常让编码器输出一个高斯分布，模型从中采样连续潜变量 $z$ ，再由解码器重建输入 $x$

- 但连续空间并不一定是所有任务最合适的表示形式。例如，语音中的音素、文本中的符号、图像中的局部模式，都可能更适合通过有限个离散符号的组合来表达

- VQ-VAE 于是提出另一种思路：**编码器产生的连续特征->从有限码本中选出离散量化向量->解码器根据量化向量生成的重建结果**

- 摘要部分还提到了在当时的生成模型中经常会发生的一个问题：强大的自回归解码器能够直接学习数据中的复杂依赖关系，获得很好的似然表现；但如果解码器本身已经足够强，它就可能不再需要潜变量，导致 VAE 中的 posterior collapse（后验坍塌）；如果潜变量不再承载有效信息，那么即使模型有潜变量结构，也很难获得真正有用的压缩表示，关于这里可以参照[https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/VAE-series/VAE.md](https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/VAE-series/VAE.md)

- 换言之，离散编码的引入本质上是为了解决后验坍塌，使前半部分的Encoder的编码输出能真正包含高层语义、长程结构等信息

- VAE 试图用潜变量解释数据；PixelCNN 直接建模像素级联合分布；VQ-VAE 则把两者结合起来，让 PixelCNN 不再直接预测每一个像素，而是预测压缩后的离散编码

## 实现细节：
- 给定输入 $x$，编码器输出：

$$
z_e(x)=f_\phi(x)
$$
- 假设码本包含 $K$ 个向量，每个向量的维度为 $D$：

$$
\mathcal E=\{e_1,e_2,\ldots,e_K\},
\qquad e_k\in\mathbb R^D
$$

## 站在现在的角度，回看这篇文章


## 参考文章和博客，感谢他们的分析解读

