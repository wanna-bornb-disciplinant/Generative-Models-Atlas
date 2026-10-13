# RNN、GRU 与 LSTM：设计原理、前向传播、反向传播与代码实现

> 从循环状态的设计出发，推导 Vanilla RNN、LSTM、GRU 的前向计算和时间反向传播（BPTT），并通过 NumPy 手写梯度与 PyTorch 自动求导实现建立联系。
>
> **约定**：本文采用列向量公式；NumPy 示例按单样本时间序列计算，PyTorch 示例使用 `(batch, time, feature)` 布局。GRU 更新门 `z` 表示**旧状态的保留比例**；不同资料可能采用互补约定。

## 目录

1. [为什么需要循环神经网络](#1-为什么需要循环神经网络)
2. [Vanilla RNN 的前向传播](#2-vanilla-rnn-的前向传播)
3. [RNN 的时间反向传播 BPTT](#3-rnn-的时间反向传播-bptt)
4. [梯度消失与爆炸](#4-梯度消失与爆炸)
5. [LSTM 的设计与前向传播](#5-lstm-的设计与前向传播)
6. [LSTM 的反向传播](#6-lstm-的反向传播)
7. [GRU 的设计与前向传播](#7-gru-的设计与前向传播)
8. [GRU 的反向传播](#8-gru-的反向传播)
9. [数值梯度检查](#9-数值梯度检查)
10. [PyTorch 从零实现与训练](#10-pytorch-从零实现与训练)
11. [三种网络的本质比较](#11-三种网络的本质比较)
12. [计算复杂度与工程实践](#12-计算复杂度与工程实践)
13. [与自回归生成模型的联系](#13-与自回归生成模型的联系)
14. [核心公式与参考文献](#14-核心公式与参考文献)

---

## 1. 为什么需要循环神经网络

普通 MLP 对单个时间步进行独立映射：

$$
\mathbf h_t=\phi(W\mathbf x_t+\mathbf b).
$$

如果不显式地输入历史信息，$\mathbf h_t$ 并不知道 $\mathbf x_{t-1}$ 乃至更早时刻发生过什么。虽然可以把固定长度的整段序列展平输入 MLP，但这通常要求固定输入长度，且无法自然地通过递推处理新时间步。

RNN 的关键改动是把**上一时间步的隐藏状态**作为当前时间步的另一项输入：

$$
\boxed{\mathbf h_t=\phi(W_x\mathbf x_t+W_h\mathbf h_{t-1}+\mathbf b)}.
$$

因此：

$$
\begin{aligned}
\mathbf h_1&=f(\mathbf x_1,\mathbf h_0),\\
\mathbf h_2&=f(\mathbf x_2,f(\mathbf x_1,\mathbf h_0)),\\
\mathbf h_3&=f(\mathbf x_3,f(\mathbf x_2,f(\mathbf x_1,\mathbf h_0))).
\end{aligned}
$$

**两项核心性质**：

- **递推状态**：当前状态依赖于此前的输入；但不意味着它一定能无损保留全部历史信息。
- **参数共享**：各时间步重复使用同一组 $W_x,W_h,\mathbf b$，参数量不随序列长度线性增长。

---

## 2. Vanilla RNN 的前向传播

令输入维度为 $D$、隐藏维度为 $H$：

$$
\mathbf x_t\in\mathbb R^D,\quad \mathbf h_t\in\mathbb R^H,\quad
W_x\in\mathbb R^{H\times D},\quad W_h\in\mathbb R^{H\times H}.
$$

标准 tanh RNN：

$$
\boxed{\begin{aligned}
\mathbf a_t&=W_x\mathbf x_t+W_h\mathbf h_{t-1}+\mathbf b,\\
\mathbf h_t&=\tanh(\mathbf a_t).
\end{aligned}}
$$

通常可令 $\mathbf h_0=\mathbf0$，也可以把初始状态作为输入或学习参数。tanh 取值在 $(-1,1)$，并满足 $\tanh'(a)=1-\tanh^2(a)$。

有时每步再映射成输出：

$$\widehat{\mathbf y}_t=W_y\mathbf h_t+\mathbf b_y.$$

也可以只把 $\mathbf h_T$ 送往分类或回归头。

### 2.1 NumPy 前向实现

```python
import numpy as np

def rnn_forward(X, Wx, Wh, b, h0):
    """X:(T,D), Wx:(H,D), Wh:(H,H), b/h0:(H,)"""
    h = h0.copy()
    hidden = [h.copy()]
    pre_activations = []
    for x_t in X:
        a = Wx @ x_t + Wh @ h + b
        h = np.tanh(a)
        pre_activations.append(a)
        hidden.append(h.copy())
    return np.stack(hidden[1:]), {
        "hidden": hidden,
        "pre_activation": pre_activations,
    }
```

---

## 3. RNN 的时间反向传播 BPTT

BPTT（Backpropagation Through Time）就是将循环结构按时间展开为计算图，按普通链式法则从 $T$ 逐步回传到 $1$。

为突出循环部分，暂用隐藏状态直接计算每步平方误差：

$$
L=\sum_{t=1}^T L_t,\qquad
L_t=\tfrac12\|\mathbf h_t-\mathbf y_t\|_2^2.
$$

如果有输出层 $W_y$，只需先将其反向梯度汇入 $\mathbf h_t$，循环层的推导不变。

### 3.1 为什么每步有两种梯度来源

$\mathbf h_t$ 一方面影响本步损失 $L_t$，另一方面影响下一状态 $\mathbf h_{t+1}$，从而影响未来的损失。因此总梯度为：

$$
\boxed{\frac{\partial L}{\partial \mathbf h_t}
=\frac{\partial L_t}{\partial\mathbf h_t}
+\left(\frac{\partial\mathbf h_{t+1}}{\partial\mathbf h_t}\right)^\top
\frac{\partial L}{\partial\mathbf h_{t+1}}}.
$$

定义 $\mathbf q_t=\partial L_t/\partial\mathbf h_t$，$\boldsymbol\delta_t=\partial L/\partial\mathbf a_t$。由于 $\mathbf h_t=\tanh(\mathbf a_t)$：

$$
\boxed{\begin{aligned}
\mathbf g_t&=\mathbf q_t+W_h^\top\boldsymbol\delta_{t+1},\\
\boldsymbol\delta_t&=\mathbf g_t\odot(1-\mathbf h_t^2).
\end{aligned}}
$$

边界为 $\boldsymbol\delta_{T+1}=\mathbf0$；$\odot$ 表示逐元素乘法。

### 3.2 参数梯度与输入梯度

一个时间步的贡献：

$$
\begin{aligned}
\left.\frac{\partial L}{\partial W_x}\right|_t&=\boldsymbol\delta_t\mathbf x_t^\top,\\
\left.\frac{\partial L}{\partial W_h}\right|_t&=\boldsymbol\delta_t\mathbf h_{t-1}^\top,\\
\left.\frac{\partial L}{\partial\mathbf b}\right|_t&=\boldsymbol\delta_t,\\
\frac{\partial L}{\partial\mathbf x_t}&=W_x^\top\boldsymbol\delta_t.
\end{aligned}
$$

**因为时间步之间共享权重，参数梯度要在全部时间步求和**，例如：

$$\boxed{\frac{\partial L}{\partial W_h}=\sum_{t=1}^T\boldsymbol\delta_t\mathbf h_{t-1}^\top}.$$

### 3.3 NumPy 手写 BPTT

```python
def rnn_backward(X, Y, Wx, Wh, cache):
    """L = 0.5 * sum_t ||h_t - Y_t||^2"""
    hidden = cache["hidden"]
    T, D = X.shape
    H = Wh.shape[0]
    dWx = np.zeros_like(Wx)
    dWh = np.zeros_like(Wh)
    db = np.zeros(H)
    dX = np.zeros_like(X)
    dh_next = np.zeros(H)

    for t in reversed(range(T)):
        h_t = hidden[t + 1]
        h_prev = hidden[t]
        dh = (h_t - Y[t]) + dh_next
        da = dh * (1 - h_t**2)
        dWx += np.outer(da, X[t])
        dWh += np.outer(da, h_prev)
        db += da
        dX[t] = Wx.T @ da
        dh_next = Wh.T @ da

    return {"Wx": dWx, "Wh": dWh, "b": db,
            "X": dX, "h0": dh_next}
```

两行尤其重要：

```python
dh = (h_t - Y[t]) + dh_next  # 本步损失 + 未来回传
dh_next = Wh.T @ da          # 再传向上一个时间步
```

---

## 4. 梯度消失与爆炸

令单步 Jacobian：

$$
J_t=\frac{\partial\mathbf h_t}{\partial\mathbf h_{t-1}}
=\operatorname{diag}(1-\mathbf h_t^2)W_h.
$$

最终状态对早期状态的导数为：

$$
\frac{\partial\mathbf h_T}{\partial\mathbf h_k}
=J_T J_{T-1}\cdots J_{k+1}.
$$

长期梯度依赖**连续 Jacobian 矩阵乘积**：反复收缩可能造成梯度消失，反复放大可能造成梯度爆炸。

一维近线性例子：

$$h_t=\tanh(wh_{t-1}+ux_t),\qquad
\frac{\partial h_T}{\partial h_k}\approx w^{T-k}.$$

- 若 $|w|<1$，梯度倾向指数衰减。
- 若 $|w|>1$，梯度可能指数增大。
- tanh 饱和时 $1-\tanh^2(a)\to0$，会进一步加剧衰减。

**准确性提醒**：多维非线性情形不能仅靠单个 $W_h$ 的特征值断言最终梯度，必须考虑变化的 Jacobian 乘积及各时间步实际激活状态。

常见缓解办法包括正交初始化、梯度裁剪、合理激活与门控结构。**梯度裁剪主要限制爆炸，不会把已经消失的梯度恢复回来。**

---

## 5. LSTM 的设计与前向传播

LSTM（Long Short-Term Memory）引入两个状态：

$$\mathbf h_t\quad\text{隐藏状态},\qquad
\mathbf c_t\quad\text{细胞状态（记忆）}.$$

设计重点是：**让长期记忆通过受门控制的加性通道更新，而不是每步整体经过一次新的隐藏矩阵与 tanh。**

### 5.1 三个门与一个候选记忆

使用 $\sigma(a)=1/(1+e^{-a})$：

$$
\boxed{\begin{aligned}
\mathbf f_t&=\sigma(W_f\mathbf x_t+U_f\mathbf h_{t-1}+\mathbf b_f),&&\text{遗忘门},\\
\mathbf i_t&=\sigma(W_i\mathbf x_t+U_i\mathbf h_{t-1}+\mathbf b_i),&&\text{输入门},\\
\mathbf g_t&=\tanh(W_g\mathbf x_t+U_g\mathbf h_{t-1}+\mathbf b_g),&&\text{候选记忆},\\
\mathbf o_t&=\sigma(W_o\mathbf x_t+U_o\mathbf h_{t-1}+\mathbf b_o),&&\text{输出门},\\
\mathbf c_t&=\mathbf f_t\odot\mathbf c_{t-1}+\mathbf i_t\odot\mathbf g_t,\\
\mathbf h_t&=\mathbf o_t\odot\tanh(\mathbf c_t).
\end{aligned}}
$$

它们的分工：

- $\mathbf f_t$：旧记忆保留多少；接近 1 时旧记忆沿直接通路保留。
- $\mathbf g_t$：新的候选内容是什么。
- $\mathbf i_t$：候选内容写入多少。
- $\mathbf o_t$：当前从记忆中向外暴露哪些信息。

尤其是：

$$\boxed{\mathbf c_t=\mathbf f_t\odot\mathbf c_{t-1}+\mathbf i_t\odot\mathbf g_t}.$$

当 $\mathbf f_t\approx\mathbf1$、$\mathbf i_t\approx\mathbf0$ 时，有 $\mathbf c_t\approx\mathbf c_{t-1}$。长时记忆不必每步整体经过 tanh 重算。

> 历史说明：今天框架中常见、包含遗忘门的 LSTM 是后来逐步发展的标准形式，并非与 1997 年原始 LSTM 的细节完全一致。

### 5.2 合并四个仿射变换

为了减少矩阵乘法调用，令 $\mathbf z_t=[\mathbf x_t;\mathbf h_{t-1}]$：

$$
\begin{bmatrix}\mathbf a_{f,t}\\\mathbf a_{i,t}\\\mathbf a_{g,t}\\\mathbf a_{o,t}\end{bmatrix}
=W\mathbf z_t+\mathbf b,
\quad W\in\mathbb R^{4H\times(D+H)}.
$$

注意下方 NumPy 和 PyTorch 实现的门顺序固定为 `f, i, g, o`。

### 5.3 NumPy 前向实现

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def lstm_forward(X, W, b, h0, c0):
    """X:(T,D), W:(4H,D+H), b:(4H,), h0/c0:(H,)"""
    h, c = h0.copy(), c0.copy()
    outputs, cache = [], []
    for x in X:
        concat = np.concatenate([x, h])
        af, ai, ag, ao = np.split(W @ concat + b, 4)
        f, i = sigmoid(af), sigmoid(ai)
        g, o = np.tanh(ag), sigmoid(ao)
        c_new = f * c + i * g
        h_new = o * np.tanh(c_new)
        cache.append({
            "concat": concat, "c_prev": c.copy(),
            "f": f, "i": i, "g": g, "o": o,
            "c": c_new.copy(), "h": h_new.copy()
        })
        h, c = h_new, c_new
        outputs.append(h.copy())
    return np.stack(outputs), (h, c), cache
```

---

## 6. LSTM 的反向传播

与 RNN 不同，LSTM 必须同时回传 $\mathbf h_t$ 和 $\mathbf c_t$ 的梯度。

记 $\bar{\mathbf v}=\partial L/\partial\mathbf v$。以下 $\bar{\mathbf h}_t$ 是已汇合本步损失和未来隐藏状态分支的总梯度，而 $\bar{\mathbf c}_t$ 还需要合并来自未来细胞状态的梯度。

### 6.1 输出门和细胞状态

由 $\mathbf h_t=\mathbf o_t\odot\tanh(\mathbf c_t)$：

$$
\begin{aligned}
\bar{\mathbf o}_t&=\bar{\mathbf h}_t\odot\tanh(\mathbf c_t),\\
\bar{\mathbf a}_{o,t}&=\bar{\mathbf o}_t\odot\mathbf o_t\odot(1-\mathbf o_t),\\
\boxed{\bar{\mathbf c}_t}&=\bar{\mathbf c}_t^{\mathrm{future}}+
\bar{\mathbf h}_t\odot\mathbf o_t\odot(1-\tanh^2(\mathbf c_t)).
\end{aligned}
$$

这说明细胞梯度有两条来源：未来时刻的细胞通路、当前隐藏状态的输出通路。

### 6.2 记忆更新公式求导

从 $\mathbf c_t=\mathbf f_t\odot\mathbf c_{t-1}+\mathbf i_t\odot\mathbf g_t$ 可得：

$$
\begin{aligned}
\bar{\mathbf f}_t&=\bar{\mathbf c}_t\odot\mathbf c_{t-1},\\
\bar{\mathbf i}_t&=\bar{\mathbf c}_t\odot\mathbf g_t,\\
\bar{\mathbf g}_t&=\bar{\mathbf c}_t\odot\mathbf i_t,\\
\boxed{\bar{\mathbf c}_{t-1}^{\mathrm{direct}}}&=\bar{\mathbf c}_t\odot\mathbf f_t.
\end{aligned}
$$

再考虑 sigmoid/tanh 导数：

$$
\begin{aligned}
\bar{\mathbf a}_{f,t}&=\bar{\mathbf f}_t\odot\mathbf f_t\odot(1-\mathbf f_t),\\
\bar{\mathbf a}_{i,t}&=\bar{\mathbf i}_t\odot\mathbf i_t\odot(1-\mathbf i_t),\\
\bar{\mathbf a}_{g,t}&=\bar{\mathbf g}_t\odot(1-\mathbf g_t^2),\\
\bar{\mathbf a}_{o,t}&=\bar{\mathbf o}_t\odot\mathbf o_t\odot(1-\mathbf o_t).
\end{aligned}
$$

拼接 $\bar{\mathbf a}_t=[\bar{\mathbf a}_{f,t};\bar{\mathbf a}_{i,t};\bar{\mathbf a}_{g,t};\bar{\mathbf a}_{o,t}]$，则：

$$
\left.\frac{\partial L}{\partial W}\right|_t=\bar{\mathbf a}_t\mathbf z_t^\top,\qquad
\left.\frac{\partial L}{\partial\mathbf b}\right|_t=\bar{\mathbf a}_t,\qquad
\bar{\mathbf z}_t=W^\top\bar{\mathbf a}_t.
$$

把 $\bar{\mathbf z}_t$ 按特征维拆开，就得到 $\bar{\mathbf x}_t$ 与向 $\mathbf h_{t-1}$ 传播的梯度。

### 6.3 NumPy 手写 BPTT

```python
def lstm_backward(X, Y, W, cache):
    """L = 0.5 * sum_t ||h_t - Y_t||^2"""
    T, D = X.shape
    H = Y.shape[1]
    dW, db = np.zeros_like(W), np.zeros(4 * H)
    dX = np.zeros_like(X)
    dh_next, dc_next = np.zeros(H), np.zeros(H)

    for t in reversed(range(T)):
        s = cache[t]
        f, i, g, o = s["f"], s["i"], s["g"], s["o"]
        c, c_prev, h, z = s["c"], s["c_prev"], s["h"], s["concat"]

        dh = (h - Y[t]) + dh_next
        tanh_c = np.tanh(c)
        dao = (dh * tanh_c) * o * (1 - o)
        dc = dc_next + dh * o * (1 - tanh_c**2)
        daf = (dc * c_prev) * f * (1 - f)
        dai = (dc * g) * i * (1 - i)
        dag = (dc * i) * (1 - g**2)
        dc_next = dc * f

        da = np.concatenate([daf, dai, dag, dao])
        dW += np.outer(da, z)
        db += da
        dz = W.T @ da
        dX[t] = dz[:D]
        dh_next = dz[D:]

    return {"W": dW, "b": db, "X": dX,
            "h0": dh_next, "c0": dc_next}
```

### 6.4 为什么有助于长期梯度传播

沿**直接的细胞记忆通路**，当把其他门控反馈路径分开看时：

$$
\bar{\mathbf c}_{t-1}^{\mathrm{direct}}=\bar{\mathbf c}_t\odot\mathbf f_t.
$$

因此从 $T$ 回到 $k$ 的这条路径具有：

$$
\boxed{\bar{\mathbf c}_k^{\mathrm{direct\ path}}
=\left(\bigodot_{t=k+1}^{T}\mathbf f_t\right)
\odot\bar{\mathbf c}_T}.
$$

当遗忘门接近 1，直接通路的梯度可以较好地保留。但**这不是完整 Jacobian 的全部梯度，也不保证 LSTM 永不出现梯度消失**。门本身依赖 $\mathbf h_{t-1}$，完整反向传播还包含门控反馈路径。

---

## 7. GRU 的设计与前向传播

GRU（Gated Recurrent Unit）将门控记忆整合到单一的 $\mathbf h_t$，不再显式维护独立细胞状态。它主要包含两个门：

- **更新门** $\mathbf z_t$：最终状态保留多少旧隐藏状态。
- **重置门** $\mathbf r_t$：生成候选状态时使用多少历史状态。

### 7.1 前向公式

本文采用以下记号：

$$
\boxed{\begin{aligned}
\mathbf z_t&=\sigma(W_z\mathbf x_t+U_z\mathbf h_{t-1}+\mathbf b_z),\\
\mathbf r_t&=\sigma(W_r\mathbf x_t+U_r\mathbf h_{t-1}+\mathbf b_r),\\
\widetilde{\mathbf h}_t&=\tanh\!\left(W_h\mathbf x_t+U_h(\mathbf r_t\odot\mathbf h_{t-1})+\mathbf b_h\right),\\
\mathbf h_t&=\mathbf z_t\odot\mathbf h_{t-1}+(1-\mathbf z_t)\odot\widetilde{\mathbf h}_t.
\end{aligned}}
$$

当 $\mathbf z_t\approx1$ 时保留旧状态；当 $\mathbf z_t\approx0$ 时使用候选状态。若 $\mathbf r_t\approx0$，生成候选状态时几乎不使用旧隐藏状态。

**不同文献的记号提示**：有的 GRU 公式写成 $(1-z_t)h_{t-1}+z_t\tilde h_t$，只是将 $z$ 定义为“采用新信息的比例”。阅读论文和移植权重时必须核实这一点。

**PyTorch 内置 GRU 的另一项差异**：它通常采用

$$
\widetilde{\mathbf h}_t=\tanh\left(
W_h\mathbf x_t+\mathbf r_t\odot(U_h\mathbf h_{t-1}+\mathbf b_{hh})+\mathbf b_{ih}\right).
$$

它与本文的 $U_h(\mathbf r_t\odot\mathbf h_{t-1})$ **通常不等价**，因为矩阵乘法与逐元素门控不能任意交换。下文手写实现遵循本文公式，不应直接拿未经转换的参数与 `nn.GRU` 做严格数值比较。

### 7.2 NumPy 前向实现

```python
def gru_forward(X, params, h0):
    """params: {'z':(Wz,Uz,bz), 'r':(...), 'h':(...)}"""
    Wz, Uz, bz = params["z"]
    Wr, Ur, br = params["r"]
    Wh, Uh, bh = params["h"]
    h = h0.copy()
    outputs, cache = [], []

    for x in X:
        h_prev = h.copy()
        z = sigmoid(Wz @ x + Uz @ h_prev + bz)
        r = sigmoid(Wr @ x + Ur @ h_prev + br)
        v = r * h_prev
        candidate = np.tanh(Wh @ x + Uh @ v + bh)
        h = z * h_prev + (1 - z) * candidate
        cache.append({"x": x, "h_prev": h_prev,
                      "z": z, "r": r, "v": v,
                      "candidate": candidate, "h": h.copy()})
        outputs.append(h.copy())

    return np.stack(outputs), cache
```

---

## 8. GRU 的反向传播

令 $\bar{\mathbf h}_t$ 表示已汇合本步和未来损失的总隐藏状态梯度。

### 8.1 更新公式求导

由

$$\mathbf h_t=\mathbf z_t\odot\mathbf h_{t-1}+(1-\mathbf z_t)\odot\widetilde{\mathbf h}_t$$

可得：

$$
\boxed{\begin{aligned}
\bar{\mathbf z}_t&=\bar{\mathbf h}_t\odot(\mathbf h_{t-1}-\widetilde{\mathbf h}_t),\\
\bar{\widetilde{\mathbf h}}_t&=\bar{\mathbf h}_t\odot(1-\mathbf z_t),\\
\bar{\mathbf h}_{t-1}^{\mathrm{direct}}&=\bar{\mathbf h}_t\odot\mathbf z_t.
\end{aligned}}
$$

最后一项是 GRU 缓解长期梯度消失的重要**直接保留路径**。

### 8.2 候选状态与重置门求导

令 $\mathbf v_t=\mathbf r_t\odot\mathbf h_{t-1}$，则：

$$
\begin{aligned}
\bar{\mathbf a}_{h,t}&=\bar{\widetilde{\mathbf h}}_t\odot(1-\widetilde{\mathbf h}_t^2),\\
\bar{\mathbf v}_t&=U_h^\top\bar{\mathbf a}_{h,t},\\
\bar{\mathbf r}_t&=\bar{\mathbf v}_t\odot\mathbf h_{t-1},\\
\bar{\mathbf h}_{t-1}^{(r)}&=\bar{\mathbf v}_t\odot\mathbf r_t.
\end{aligned}
$$

两个 sigmoid 门的激活前梯度：

$$
\begin{aligned}
\bar{\mathbf a}_{z,t}&=\bar{\mathbf z}_t\odot\mathbf z_t\odot(1-\mathbf z_t),\\
\bar{\mathbf a}_{r,t}&=\bar{\mathbf r}_t\odot\mathbf r_t\odot(1-\mathbf r_t).
\end{aligned}
$$

例如 $\partial L/\partial W_z|_t=\bar{\mathbf a}_{z,t}\mathbf x_t^\top$，对其余线性层同理。把四条来源汇合：

$$
\boxed{\bar{\mathbf h}_{t-1}=
\underbrace{\bar{\mathbf h}_t\odot\mathbf z_t}_{\text{直接路径}}+
\underbrace{\bar{\mathbf v}_t\odot\mathbf r_t}_{\text{候选路径}}+
U_z^\top\bar{\mathbf a}_{z,t}+
U_r^\top\bar{\mathbf a}_{r,t}.}
$$

### 8.3 NumPy 手写 BPTT

```python
def gru_backward(X, Y, params, cache):
    """L = 0.5 * sum_t ||h_t - Y_t||^2"""
    T, D = X.shape
    H = Y.shape[1]
    Wz, Uz, bz = params["z"]
    Wr, Ur, br = params["r"]
    Wh, Uh, bh = params["h"]

    grads = {
        key: [np.zeros_like(p) for p in values]
        for key, values in params.items()
    }
    dX = np.zeros_like(X)
    dh_next = np.zeros(H)

    for t in reversed(range(T)):
        s = cache[t]
        x, h_prev, h = s["x"], s["h_prev"], s["h"]
        z, r, v, candidate = s["z"], s["r"], s["v"], s["candidate"]
        dh = (h - Y[t]) + dh_next

        dz = dh * (h_prev - candidate)
        dcandidate = dh * (1 - z)
        dh_prev = dh * z

        dah = dcandidate * (1 - candidate**2)
        grads["h"][0] += np.outer(dah, x)
        grads["h"][1] += np.outer(dah, v)
        grads["h"][2] += dah

        dv = Uh.T @ dah
        dr = dv * h_prev
        dh_prev += dv * r

        dar = dr * r * (1 - r)
        grads["r"][0] += np.outer(dar, x)
        grads["r"][1] += np.outer(dar, h_prev)
        grads["r"][2] += dar
        dh_prev += Ur.T @ dar

        daz = dz * z * (1 - z)
        grads["z"][0] += np.outer(daz, x)
        grads["z"][1] += np.outer(daz, h_prev)
        grads["z"][2] += daz
        dh_prev += Uz.T @ daz

        dX[t] = Wh.T @ dah + Wr.T @ dar + Wz.T @ daz
        dh_next = dh_prev

    return {"params": grads, "X": dX, "h0": dh_next}
```

---

## 9. 数值梯度检查

手写 BPTT 容易因门顺序、转置或漏掉某条梯度路径而出错。可用中心差分近似参数的数值导数：

$$
\boxed{g_{\mathrm{num}}\approx\frac{L(\theta+\epsilon)-L(\theta-\epsilon)}{2\epsilon}}.
$$

与手写解析梯度 $g_{\mathrm{ana}}$ 比较，例如相对误差：

$$
E=\frac{|g_{\mathrm{ana}}-g_{\mathrm{num}}|}
{\max(1,|g_{\mathrm{ana}}|,|g_{\mathrm{num}}|)}.
$$

### RNN 最小检查示例

```python
np.random.seed(42)
T, D, H = 5, 3, 4
X, Y = np.random.randn(T, D), np.random.randn(T, H)
Wx = np.random.randn(H, D) * 0.2
Wh = np.random.randn(H, H) * 0.2
b, h0 = np.zeros(H), np.zeros(H)

def compute_loss():
    output, _ = rnn_forward(X, Wx, Wh, b, h0)
    return 0.5 * np.sum((output - Y)**2)

_, cache = rnn_forward(X, Wx, Wh, b, h0)
grads = rnn_backward(X, Y, Wx, Wh, cache)
eps, original = 1e-5, Wh[0, 0]
Wh[0, 0] = original + eps
loss_plus = compute_loss()
Wh[0, 0] = original - eps
loss_minus = compute_loss()
Wh[0, 0] = original
numerical = (loss_plus - loss_minus) / (2 * eps)
analytical = grads["Wh"][0, 0]
print("Numerical:", numerical)
print("Analytical:", analytical)
print("Absolute error:", abs(numerical - analytical))
```

同理，可以使用 `lstm_forward/lstm_backward` 或 `gru_forward/gru_backward` 对每个参数逐元素执行中心差分。建议用 `float64`、小规模权重和合理的 $\epsilon$，数值梯度检查仅用于测试，不适合正式训练。

---

## 10. PyTorch 从零实现与训练

以下类不调用 `nn.RNN`、`nn.LSTM`、`nn.GRU`。PyTorch 的 Autograd 根据每步运算构建计算图，因此 `loss.backward()` 自动执行同一套 BPTT 链式法则。

### 10.1 Vanilla RNN

```python
import torch
import torch.nn as nn

class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.Wx = nn.Linear(input_size, hidden_size)
        self.Wh = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, h0=None):
        B, T, D = x.shape
        h = x.new_zeros(B, self.hidden_size) if h0 is None else h0
        outputs = []
        for t in range(T):
            h = torch.tanh(self.Wx(x[:, t, :]) + self.Wh(h))
            outputs.append(h)
        return torch.stack(outputs, dim=1), h
```

### 10.2 LSTM

```python
class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.gates = nn.Linear(input_size + hidden_size, 4 * hidden_size)

    def forward(self, x, h0=None, c0=None):
        B, T, D = x.shape
        H = self.hidden_size
        h = x.new_zeros(B, H) if h0 is None else h0
        c = x.new_zeros(B, H) if c0 is None else c0
        outputs = []
        for t in range(T):
            a = self.gates(torch.cat([x[:, t, :], h], dim=-1))
            af, ai, ag, ao = a.chunk(4, dim=-1)
            f, i, g, o = torch.sigmoid(af), torch.sigmoid(ai), torch.tanh(ag), torch.sigmoid(ao)
            c = f * c + i * g
            h = o * torch.tanh(c)
            outputs.append(h)
        return torch.stack(outputs, dim=1), (h, c)
```

### 10.3 GRU

```python
class MyGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        H = hidden_size
        self.hidden_size = H
        self.Wz = nn.Linear(input_size + H, H)
        self.Wr = nn.Linear(input_size + H, H)
        self.Wn = nn.Linear(input_size, H)
        self.Un = nn.Linear(H, H, bias=False)

    def forward(self, x, h0=None):
        B, T, D = x.shape
        H = self.hidden_size
        h = x.new_zeros(B, H) if h0 is None else h0
        outputs = []
        for t in range(T):
            x_t = x[:, t, :]
            combined = torch.cat([x_t, h], dim=-1)
            z = torch.sigmoid(self.Wz(combined))
            r = torch.sigmoid(self.Wr(combined))
            candidate = torch.tanh(self.Wn(x_t) + self.Un(r * h))
            h = z * h + (1 - z) * candidate
            outputs.append(h)
        return torch.stack(outputs, dim=1), h
```

### 10.4 用序列求和任务训练

下面任务让网络从长度为 $T$ 的标量序列预测其和：

$$y=\sum_{t=1}^{T}x_t.$$

```python
torch.manual_seed(42)
B, T, D, H = 64, 20, 1, 32

class SequenceModel(nn.Module):
    def __init__(self, rnn_type="lstm"):
        super().__init__()
        encoders = {"rnn": MyRNN, "gru": MyGRU, "lstm": MyLSTM}
        self.encoder = encoders[rnn_type](D, H)
        self.head = nn.Linear(H, 1)

    def forward(self, x):
        outputs, _ = self.encoder(x)
        return self.head(outputs[:, -1, :])

model = SequenceModel("lstm")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for step in range(1000):
    X = torch.randn(B, T, D)
    Y = X.sum(dim=1)
    prediction = model(X)
    loss = criterion(prediction, Y)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    if (step + 1) % 100 == 0:
        print(f"Step {step+1}: Loss = {loss.item():.6f}")
```

注意这只是展示训练流程的教学实验，并不能单凭一次结果证明某类网络更优。若要严格比较长期依赖，应改变序列长度、随机种子并使用延迟复制、Adding Problem 等任务。

---

## 11. 三种网络的本质比较

| 比较项 | Vanilla RNN | LSTM | GRU |
|---|---|---|---|
| 递推状态 | $h_t$ | $(h_t,c_t)$ | $h_t$ |
| 门控 | 无 | 遗忘、输入、输出 | 更新、重置 |
| 旧状态传播 | 非线性重新计算 | 细胞状态加性更新 | 隐藏状态门控插值 |
| 主要直接梯度路径 | 无独立复制通道 | $\bar c_t\odot f_t$ | $\bar h_t\odot z_t$ |
| 结构 | 最简单 | 最复杂 | 介于二者之间 |
| 长期依赖 | 通常较难训练 | 通常更容易 | 通常更容易 |

从最核心的状态更新式对比：

**RNN**：

$$\mathbf h_t=\tanh(W_x\mathbf x_t+W_h\mathbf h_{t-1}+\mathbf b).$$

**LSTM**：

$$\mathbf c_t=\mathbf f_t\odot\mathbf c_{t-1}+\mathbf i_t\odot\mathbf g_t.$$

**GRU**：

$$\mathbf h_t=\mathbf z_t\odot\mathbf h_{t-1}+(1-\mathbf z_t)\odot\widetilde{\mathbf h}_t.$$

三者的关键区别不只是门的数量，而是**旧状态从 $t-1$ 到 $t$ 的传播方式**。RNN 每步重新做矩阵与非线性变换；LSTM 为细胞记忆提供直接加性路径；GRU 直接对旧、新隐藏状态进行门控混合。

### 11.1 固定保留系数的直观例子

若只看直接复制路径，且不考虑新信息与其他门控反馈，设 $s_t=\alpha s_{t-1}$，则：

$$s_T=\alpha^T s_0,\qquad \frac{\partial s_T}{\partial s_0}=\alpha^T.$$

例如连续 30 步、$\alpha=0.95$ 时，直接通路的比例约为 $0.95^{30}\approx0.215$。这说明即使保留门每一步都比较接近 1，时间足够长时仍可能逐步衰减；网络必须学会对重要信息使用足够高的保留系数。

---

## 12. 计算复杂度与工程实践

### 12.1 参数量

单层、单方向，且每组变换只用**一组偏置**时：

| 模型 | 参数量 |
|---|---:|
| RNN | $H(D+H+1)$ |
| GRU | $3H(D+H+1)$ |
| LSTM | $4H(D+H+1)$ |

当 $D=64,H=128$ 时，参数量分别为 **24,704 / 74,112 / 98,816**。实际框架可能分别维护输入和循环部分的偏置；尤其 PyTorch 内置层的参数布局与手写合并层并不完全一致。

GRU 在相同隐藏宽度下通常具有比 LSTM 更少的参数和理论算术量，但**实际运行时间**还取决于融合 kernel、GPU、batch size 和序列长度。

### 12.2 时间串行依赖

由于 $\mathbf h_t=F(\mathbf x_t,\mathbf h_{t-1})$，单方向标准循环网络的时间步通常需要依次执行。一个时间步内部的矩阵运算及不同 batch 样本仍可并行。这是 Transformer 在训练时更适合并行处理整个序列的原因之一；但自回归推理的 token 生成仍有因果串行依赖。

### 12.3 Truncated BPTT（截断时间反向传播）

长序列完整展开会占用较多内存。截断 BPTT 保留隐藏状态的**值**，但周期性切断计算图，避免当前损失继续反传到更早时间步：

```python
# 伪代码：model 必须支持传入并返回隐藏状态
h = None
for x_chunk, y_chunk in data_loader:
    if h is not None:
        h = h.detach()  # 保留数值，切断历史计算图
    outputs, h = model(x_chunk, h)
    loss = criterion(outputs, y_chunk)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

LSTM 则需要将 `(h, c)` 两个状态同时 `detach()`。此方法降低时间反传跨度，但不等于重置记忆。

### 12.4 梯度裁剪

对整体参数梯度 $\mathbf g$，按阈值 $c$ 裁剪其范数：

$$\widetilde{\mathbf g}=\mathbf g\min\left(1,\frac{c}{\|\mathbf g\|_2}\right).$$

PyTorch 用法：

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

梯度裁剪主要防止更新幅度过大，不能直接修复梯度消失。

---

## 13. 与自回归生成模型的联系

序列自回归分解：

$$p(\mathbf x)=\prod_{t=1}^T p(x_t\mid x_1,\ldots,x_{t-1}).$$

循环模型可以用状态 $\mathbf h_t=F_\theta(\mathbf h_{t-1},x_t)$ 压缩历史，再用它参数化后续条件分布。经典 PixelRNN 即将循环机制用于图像像素的条件依赖建模。

不过，**自回归分解是一种概率模型的因子分解方式，RNN 是实现条件建模的一类架构**：Transformer 等模型也能够采用相同的自回归概率分解。

可以将三种循环网络统一理解为非线性状态空间模型：

$$\boxed{\mathbf s_t=F_\theta(\mathbf s_{t-1},\mathbf x_t)}.$$

其中 RNN/GRU 的状态是 $\mathbf h_t$，LSTM 的完整递推状态是 $(\mathbf h_t,\mathbf c_t)$。

---

## 14. 核心公式与参考文献

### 14.1 最值得记住的六个公式

1. **RNN 前向**：
   $$\mathbf h_t=\tanh(W_x\mathbf x_t+W_h\mathbf h_{t-1}+\mathbf b).$$
2. **RNN 的 BPTT**：
   $$\boldsymbol\delta_t=(\mathbf q_t+W_h^\top\boldsymbol\delta_{t+1})\odot(1-\mathbf h_t^2).$$
3. **LSTM 记忆更新**：
   $$\mathbf c_t=\mathbf f_t\odot\mathbf c_{t-1}+\mathbf i_t\odot\mathbf g_t.$$
4. **LSTM 输出**：
   $$\mathbf h_t=\mathbf o_t\odot\tanh(\mathbf c_t).$$
5. **GRU 候选状态**：
   $$\widetilde{\mathbf h}_t=\tanh\left(W_h\mathbf x_t+U_h(\mathbf r_t\odot\mathbf h_{t-1})+\mathbf b_h\right).$$
6. **GRU 更新**：
   $$\mathbf h_t=\mathbf z_t\odot\mathbf h_{t-1}+(1-\mathbf z_t)\odot\widetilde{\mathbf h}_t.$$

### 14.2 推荐原始文献

- Rumelhart, Hinton, & Williams (1986). *Learning representations by back-propagating errors*. Nature. https://doi.org/10.1038/323533a0
- Bengio, Simard, & Frasconi (1994). *Learning long-term dependencies with gradient descent is difficult*. IEEE Transactions on Neural Networks. https://doi.org/10.1109/72.279181
- Hochreiter & Schmidhuber (1997). *Long Short-Term Memory*. Neural Computation. https://doi.org/10.1162/neco.1997.9.8.1735
- Gers, Schmidhuber, & Cummins (2000). *Learning to Forget: Continual Prediction with LSTM*. Neural Computation. https://doi.org/10.1162/089976600300015015
- Cho et al. (2014). *Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation*. https://arxiv.org/abs/1406.1078
- van den Oord et al. (2016). *Pixel Recurrent Neural Networks*. https://arxiv.org/abs/1601.06759

---

**核心结论**：Vanilla RNN 引入递推历史状态，LSTM 通过独立细胞记忆和门控加性更新改进长期信息传递，GRU 通过对隐藏状态进行门控插值实现更紧凑的结构。学习这三类模型时，最重要的是理解状态的**前向保留路径**如何对应到**反向梯度路径**。
