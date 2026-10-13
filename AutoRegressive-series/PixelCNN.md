# PixelCNN

## 发表情况
- Van Den Oord, A., Kalchbrenner, N., & Kavukcuoglu, K. (2016, June). Pixel recurrent neural networks. In International conference on machine learning (pp. 1747-1756). PMLR.
  - ICML 2016
- 主要作者：Aaron van den Oord, Nal Kalchbrenner, Koray Kavukcuoglu
- 截止2026.10 cite: 3743+
- Google DeepMind
- [原文链接](https://proceedings.mlr.press/v48/oord16.pdf)

## 摘要解读
- PixelCNN仍然希望建模自然图像的数据分布
- 文章对生成任务的模型需求定义为具备强大的表达能力、可计算性和可扩展，三点分别对应于生成模型需要足够强大、对于似然概率可以精确求解计算、能在大数据集上，而不是MNIST和CIFAR之类的数据集上玩泥巴
- 原文的摘要部分更多的强调了PixelRNN的结构，包括文章标题也是，本身PixelCNN在文中是作为方法的一种变种存在的
- 文章在摘要部分强调了自回归建模的理论可行性，为每个像素单独建模概率分布，但同时利用自回归的特性建模像素之间的复杂依赖关系，但这是理论上的说明

## 背景说明
- PixelCNN仍然像VAE一样强调了概率建模本身不只是可以用于生成，也可以用在图像压缩、修复等领域
- PixelCNN的核心在于：

$$
p_\theta(\mathbf{x})=
\prod_i p_\theta(x_i\mid\mathbf{x}_{<i})
$$
- 例如在VAE中，很难解决的一个问题是如何定义高维的分布，因为归一化常数永远是一个绕不开的话题，但在PixelCNN这类自回归模型中，我只需要归一化单个像素的条件分布，联合分布自然也是归一化的，这意味着：
  - 不需要额外的推断网络，类似VAE的Encoder，需要去学习概率
  - 不需要估计潜变量后验，ELBO希望解决的问题
  - 不需要变分下界
  - 可以直接优化精确的模型对数似然
- 但自回归的模式解决的是概率密度可计算的问题，并没有解决生成采样的计算效率问题，自回归分解反而带来了严格的顺序生成约束
- 上述公式依赖于概率论的链式法则，从概率表达能力看，如果所有条件分布都能够被完美建模，那么这个分解可以表示任意离散图像联合分布。不过，这不代表任意有限规模的 PixelRNN 都具有这种表达能力。链式分解在理论上无损，具体神经网络的参数化仍然会限制其表达能力。
- 还有一个点是，PixelCNN和PixelRNN直接预测原始 8-bit 像素值的离散概率，作者认为离散分布预测使模型能够表达更复杂的局部条件分布；但256 类 Softmax 也存在代价：输出维度较大，而且没有显式利用相邻强度值之间的有序关系

## 具体分析
- 论文实际上提出了几种相关结构，包括按行递归的Row LSTM、对角线递归的Diagonal BiLSTM和掩码卷积堆叠的PixelCNN
- 

## 实现细节

## 站在现在的角度，回看这篇文章

## 参考文章和博客，感谢他们的分析解读
- [详解自回归模型 PixelCNN 大家族](https://zhuanlan.zhihu.com/p/632209862)
- 关于RNN、LSTM和GRU的解读：
  - [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
  - [Generative-Models-Atlas/RNN.md](https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/AutoRegressive-series/RNN.md)
