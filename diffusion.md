# Diffusion 模型讲解

## 核心思想

Diffusion 模型是一类生成模型，核心思路非常直觉：**先把图像"破坏"成噪声，再学会"还原"它**。

就像把一张照片慢慢打上马赛克直到完全看不出来，然后训练神经网络学会把马赛克一步步去掉。

---

## 两个过程

### 🔴 前向过程（Forward Process）— 加噪

从真实图像 $x_0$ 出发，每一步加一点点高斯噪声，经过 $T$ 步后变成纯噪声 $x_T$：

$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t;\ \sqrt{1-\beta_t}\, x_{t-1},\ \beta_t \mathbf{I})
$$

一个很重要的性质是**可以跳步直达任意时刻**，不需要一步步算：

$$q(x_t | x_0) = \mathcal{N}(x_t;\ \sqrt{\bar\alpha_t}\, x_0,\ (1-\bar\alpha_t)\mathbf{I})$$

所以 $x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\, \epsilon$，其中 $\epsilon \sim \mathcal{N}(0, I)$。

### 🟢 逆向过程（Reverse Process）— 去噪

训练一个 U-Net，输入「当前噪声图 $x_t$ + 时间步 $t$」，**预测其中的噪声 $\epsilon$**，然后逐步还原出 $x_0$。

训练目标非常简单——最小化预测噪声与真实噪声的 MSE：

$$\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$

---

## 四个递进的技术

```
DDPM → DDIM → Latent DDPM → Latent DDPM + CFG
 基础    加速     更高效         可控生成
```

### 1️⃣ DDPM（基础版）
- 直接在像素空间做扩散
- 推理需要 **~1000步**，很慢
- 用 U-Net 预测每步噪声

### 2️⃣ DDIM（加速推理）
- 把马尔可夫链改为**非马尔可夫过程**
- 推理只需 **50~100步**，速度提升 10-20x
- 可以做到**确定性采样**（同样输入 → 同样输出）
- 核心公式：
$$x_{t-1} = \sqrt{\bar\alpha_{t-1}} \underbrace{\left(\frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta}{\sqrt{\bar\alpha_t}}\right)}_{\text{预测的}\ x_0} + \underbrace{\sqrt{1-\bar\alpha_{t-1}}\,\epsilon_\theta}_{\text{指向噪声方向}}$$

### 3️⃣ Latent Diffusion（LDM）
- 直接在像素空间训练**极慢**（作业说可能要 100万步才出效果）
- 解决方案：先用 **VAE** 把图像压缩到低维隐空间，再在隐空间跑扩散
- 压缩比例通常 4x~8x，计算量大幅降低
- Stable Diffusion 就是这个架构！

```
图像 → [VAE Encoder] → 隐变量 z → [DDPM/DDIM] → z' → [VAE Decoder] → 生成图像
```

### 4️⃣ Classifier-Free Guidance（CFG）
- 目标：让模型**按类别可控生成**（比如"生成一只猫"）
- 训练时：随机以 10% 概率**丢弃类别标签**，让模型同时学会有条件/无条件生成
- 推理时：做两次前向传播，然后加权组合：

$$\hat\epsilon = \epsilon_\theta(x_t, t) + w \cdot \underbrace{(\epsilon_\theta(x_t, t, y) - \epsilon_\theta(x_t, t))}_{\text{条件引导信号}}$$

$w$ 越大 → 生成图像越符合类别，但多样性降低。

---

## 评估指标

| 指标 | 含义 | 越好 |
|------|------|------|
| **FID** | 生成图像分布 vs 真实图像分布的距离 | 越低越好 |
| **IS** | 生成图像的质量 × 多样性 | 越高越好 |

---

## 一句话总结

> Diffusion = **用噪声破坏数据，训练神经网络学会反向去噪**，加上 DDIM 加速、VAE 压缩、CFG 控制，就构成了现代图像生成（如 Stable Diffusion）的核心技术栈。

---
## VAE（变分自编码器）

VAE（变分自编码器）是 Latent Diffusion 的核心基础，它解决了一个关键问题：**如何把高维图像压缩成低维"精华"，又能还原回来？**

先看整体结构：## 普通 AE 和 VAE 的本质区别

普通自编码器（AE）把图像压缩成一个**固定的点**，VAE 则压缩成一个**高斯分布**（由均值 μ 和方差 σ 描述）。这个区别至关重要：VAE 的隐空间是**连续且规整**的，任意一点都能解码出有意义的图像，这正是 Diffusion 模型能在隐空间里"漫游"的原因。

## 三大核心机制

**1. 重参数化技巧（Reparameterization Trick）**

采样操作 $z \sim \mathcal{N}(\mu, \sigma^2)$ 不可微，无法反向传播。VAE 的解决方式是：

$$z = \mu + \sigma \cdot \underbrace{\epsilon}_{\epsilon \sim \mathcal{N}(0,I)}$$

把随机性"外包"给 ε，梯度只流过 μ 和 σ，训练得以正常进行。

**2. 损失函数：两项对抗**

$$\mathcal{L}_{VAE} = \underbrace{\| x - \hat{x} \|^2}_{\text{重建损失}} + \underbrace{D_{KL}(q(z|x) \| \mathcal{N}(0,I))}_{\text{正则化损失}}$$

两项之间存在张力：重建损失希望编码尽量精确保留信息；KL 散度则强迫隐空间分布向标准正态靠拢，让空间结构规整。正是这个张力产生了有用的压缩表示。

**3. KL 散度的解析形式**

对高斯分布，KL 项可以直接算，不需要采样估计：

$$D_{KL} = -\frac{1}{2} \sum_j \left(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

## 在 Diffusion 里怎么用

作业里的流程是：

```
训练阶段：图像 → VAE Encoder → z × 0.1845 → DDPM 加噪/去噪
推理阶段：随机噪声 → DDPM 去噪 → z ÷ 0.1845 → VAE Decoder → 图像
```

乘以 0.1845 是为了让隐变量的方差约等于 1，匹配 DDPM 噪声调度的假设。VAE 权重是预训练好的，你只需要实现 `encode()` 和 `decode()` 两个函数即可。

---

### DDIM 

部分你需要做的事情很具体，分三块：

## 要实现什么

**核心任务：在 `schedulers/scheduling_ddim.py` 里实现 DDIM 的反向采样步骤。**

具体就是实现这个公式：

$$x_{t-1} = \sqrt{\bar\alpha_{t-1}} \underbrace{\left(\frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta}{\sqrt{\bar\alpha_t}}\right)}_{\text{第一步：预测 }x_0} + \underbrace{\sqrt{1-\bar\alpha_{t-1} - \sigma_t^2}\,\cdot\epsilon_\theta}_{\text{第二步：加方向}} + \underbrace{\sigma_t \cdot z}_{\text{第三步：加噪声（可选）}}$$

代码逻辑拆开就是：

```python
# 1. 用模型预测的噪声，估计出 x0
pred_x0 = (x_t - sqrt(1 - alpha_bar_t) * eps) / sqrt(alpha_bar_t)

# 2. 指向噪声方向的分量
direction = sqrt(1 - alpha_bar_prev - sigma_t**2) * eps

# 3. 可选的随机噪声（sigma_t=0 时完全确定性）
noise = sigma_t * torch.randn_like(x_t)

# 4. 组合得到 x_{t-1}
x_prev = sqrt(alpha_bar_prev) * pred_x0 + direction + noise
```

---

## 三件具体的事

**1. 修改噪声 schedule**
把 DDPM 的 $\bar\alpha_t$ 序列适配到 DDIM——DDIM 只需要从 1000 步里均匀挑选 50 个时间步（比如），而不是全跑。

**2. 实现 `step()` 函数**
就是上面那个公式。其中 $\sigma_t$ 的设置决定了是否确定性：
- `σ_t = 0` → 完全确定性，同样输入每次输出相同
- `σ_t > 0` → 随机采样，类似 DDPM

**3. 接入推理 pipeline**
让 `pipelines/ddpm.py` 在推理时能切换使用 DDIM scheduler，并且支持只跑 50 步而不是 1000 步。

---

## 和 DDPM 的区别在哪

DDPM 的 scheduler 你需要实现 `add_noise()`（训练用，前向加噪）。DDIM 的 scheduler 只用于**推理**，不涉及训练，因为它复用 DDPM 训练好的模型，只是换了一种反向采样的方式。

所以 DDIM 部分**不需要重新训练**，在已有的 DDPM checkpoint 上直接换 scheduler 就能跑，然后对比两者的生成速度和质量（FID/IS）写进报告。

---

## 验证方法

文档建议：用同样的随机种子，分别跑 DDPM（1000步）和 DDIM（50步），对比：
- 生成速度（DDIM 应该快 10-20×）
- 图像质量（FID/IS 应该接近）

这个对比结果需要写进 Midterm Report，因为 DDPM + DDIM 是 3 月 13 日就要交的内容。