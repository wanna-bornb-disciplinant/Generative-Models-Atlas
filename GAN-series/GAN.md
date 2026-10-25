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

## 实现细节：


## 站在现在的角度，回看这篇文章


## 参考文章和博客，感谢他们的分析解读


