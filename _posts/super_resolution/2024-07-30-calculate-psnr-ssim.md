---
layout: post
title: '峰值信噪比 (PSNR) 与结构相似性 (SSIM) 的计算'
date: 2024-07-30
author: 洪茬铭
cover: 'https://pic.imgdb.cn/item/66a8e435d9c307b7e9370a88.png'
tags: 图像超分
---

>**本文目录：**
>
>1. 峰值信噪比 (PSNR)
>2. 结构相似性 (SSIM)



# 1. 峰值信噪比 (PSNR)

## 1.1 计算公式

峰值信噪比 (Peak Signal-to-Noise Ratio, PSNR)，用于衡量两张图像之间差异，例如压缩图像与原始图像，评估压缩图像质量；复原图像与 ground truth，评估复原算法性能等。

公式：

$$
\text{PSNR} = 10 \times \log_{10} \left( \frac{\text{MaxValue}^2}{\text{MSE}} \right) \tag{1}
$$

其中，$MSE$ 为两张图像逐像素的均方误差；$MaxValue$ 为图像像素可取到的最大值，例如 8 位图像为  $2^8-1=255$。在代码实现中，直接将次方提到外面：
$$
\text{PSNR}=20\times\log_{10}(\frac{\text{MaxValue}}{\sqrt{\text{MSE}}}) \tag{2}
$$

## 1.2 代码实现

### 1.2.1 BGR空间

使用  opencv 读取图像时，默认的通道顺序 `HWC`，颜色顺序 `BGR`，数据类型 `np.ndarray`，数据范围为 `np.uint8` 型的 $[0, 255]$。如果是在 `BGR` 空间计算 PSNR，那么只需要分别在 `BGR` 三个通道上分别计算 $MSE$，取平均然后带入公式2即可。主要代码如下：

```python
import cv2
import numpy as np

# 读取需要比较的图像
img1 = cv2.imread('./image1.png')
img2 = cv2.imread('./image2.png')

# 数据类型转换，结果更精确
img1 = img1.astype(np.float64)
img2 = img2.astype(np.float64)

# 计算3个通道的平均MSE，然后计算psnr
mse = np.mean((img1 - img2) ** 2)
if mse == 0:
    psnr = float('inf')
else:
    psnr = 20 * np.log10(255. / np.sqrt(mse))
 
print(f'PSNR value: {psnr:.6f}dB.')
```

### 1.2.2 YCbCr空间

想要计算 `YCbCr` 空间的 PSNR，首先需要将 opencv 读取的 `BGR` 颜色空间的图像转换到 `YCbCr`颜色空间。opencv 自带有颜色空间转换函数 `cv2.COLOR_BGR2YCrCb()`，但是，opencv 的转换遵循的是 **[JPEG 转换](https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion)**，转换后的各通道具有完整的 8 bit 范围 $[0, 255]$。但是在超分的 PSNR 计算时，往往使用的是 **[ITU-R BT.601 转换](https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion)**，这种转换的结果是 Y 分量的范围从 $[0, 255]$ 变为 $[16, 235]$，而 $C_B$ 和 $C_R$ 分量从 $[0, 255]$ 变为 $[16, 240]$。ITU-R BT.601 转换公式：

$$
Y'=16+(65.481\cdot R'+128.553\cdot G'+24.966\cdot B') \\
C_B=128+(-37.797\cdot R'-74.203\cdot G'+112.0\cdot B') \tag{3} \\ 
C_R=128+(112.0\cdot R'-93.786\cdot G'+18.214\cdot B')
$$

其中，$B’G'R’$  的数值范围是  $[0,  1]$ ` type(float)`。

_**仔细看 ITU-R BT.601 转换时可以发现，计算出来的并不是 Y，而是 Y'，这两者有什么区别呢？**_

> **[伽马校正](https://blog.csdn.net/p312011150/article/details/82664844?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-1-82664844-blog-127968779.235^v43^pc_blog_bottom_relevance_base9&spm=1001.2101.3001.4242.2&utm_relevant_index=2)**——人眼特性
>
> 由于人眼对于暗光的变化较为敏感，所以当亮度由 0 变到 0.01 时，人眼感觉到的亮度变化会比亮度由 0.99 变到 1.0 更明显。也就是说，人眼认为的中灰其实不在亮度为 0.5 的地方，而是在大约亮度为 0.18 的地方（18 度灰）。
>
> ![](https://pic.imgdb.cn/item/66a8e258d9c307b7e9343743.png)
>
> 针对于人眼的特性，在拍照时需要将更多的资源用于存储暗光部分，应该把人眼认为的中灰亮度放在像素值为 0.5 的地方，也就是说，0.18 亮度应该编码成 0.5 像素值。这样存储空间就可以充分利用起来了。所以，摄影设备如果使用了 8 位空间存储照片的话，会用大约为 2.2 或 2.4 的 **Encoding Gamma** 来对输入的亮度编码，得到一张图像。这一步使得图像数据更加紧凑，可以在相同的比特深度下保存更多的细节信息。2.2 这个值完全是由于人眼的特性测量得到的。
>
> $$
> I_{\mathrm{encoded}}=I_{\mathrm{linear}}^{1/\gamma} \tag{4}
> $$
>
> 由于这个原因，我们保存的图像一般都是经过了伽马编码的，在使用 opencv 读取时，得到的信息也都是经过了伽马编码的，所以在进行颜色空间转换时得到的是经过了伽马校正的 Y‘ 而不是原始的 Y。
>
> 那么我们使用 opencv 的 `cv2.imshow()` 函数来显示图像时，显示的是校正后的还是原始的呢？答案是校正后的。OpenCV 的 `cv2.imshow()` 函数只是将图像数据直接显示到窗口中，不会应用任何额外的伽马校正。显示伽马校正（**Display Gamma**）通常是显示设备（如监视器、电视等）的责任，而不是 OpenCV 的功能。
>
> 在显示图像之前，显示设备（如监视器、电视等）会应用 **Display Gamma**，以确保图像在显示设备上看起来是正确的。这里的 gamma 通常也是 2.2 或 2.4 ，与编码伽马相匹配，但也会根据显示设备的特性有所调整
> $$
> I_{\mathrm{displayed}}=I_{\mathrm{encoded}}^{\gamma} \tag{5}
> $$
> ***注：虽然计算出来的确实是 Y'，但是在超分论文中并没有明确指出来是 Y'，而是直接使用 Y 来表示***

`BGR` ------> `YCbCr` 的主要步骤如下：[0, 255] type(int)  ---> [0, 255] type(float) ---> [0, 1] type(float) (公式3中要求为此范围) ---> [0, 255] type(float)

**主要代码如下：**

```python
"""计算结果和官方代码稍有区别"""
import cv2
import numpy as np

# 读取需要比较的图像
img1 = cv2.imread('./images/bird.png')
img2 = cv2.imread('./images/bird_SwinIR.png')

img1 = img1.astype(np.float32) / 255.
img2 = img2.astype(np.float32) / 255.

img1_y = np.dot(img1, [24.996, 128.553, 65.481]) + 16.0
img2_y = np.dot(img2, [24.996, 128.553, 65.481]) + 16.0

img1_y = img1_y[..., None]
img2_y = img2_y[..., None]

# 计算Y通道的MSE，然后计算psnr
mse = np.mean((img1_y - img2_y) ** 2)
if mse == 0:
    psnr = float('inf')
else:
    psnr = 20 * np.log10(255. / np.sqrt(mse))
 
print(f'PSNR value: {psnr:.6f}dB.')
```



# 2. 结构相似性 (SSIM)

## 2.1 计算公式

结构相似性 (Structural Similarity, SSIM) 基于人眼会提取图像中结构化信息的假设，比传统方式更符合人眼视觉感知。

公式：

$$
\mathrm{SSIM}(\mathbf{x},\mathbf{y})=[l(\mathbf{x},\mathbf{y})]^\alpha\cdot[c(\mathbf{x},\mathbf{y})]^\beta\cdot[s(\mathbf{x},\mathbf{y})]^\gamma \tag{6}
$$

SSIM 由三个部分组成，$α,β,γ>0$，用于调整三个部分的比重：

$$
l(\mathbf{x},\mathbf{y})=\frac{2\mu_x\mu_y+C_1}{\mu_x^2+\mu_y^2+C_1} \\
c(\mathbf{x},\mathbf{y})=\frac{2\sigma_x \sigma_y+C_2}{\sigma_x^2+\sigma_y^2+C_2} \tag{7} \\
s(\mathbf{x},\mathbf{y})=\frac{\sigma_{xy}+C_3}{\sigma_x\sigma_y+C_3}
$$

其中，$C_1=(K_1L)^2$ , $C_2=(K_2L)^2$, 用于规避分母为0的情况，$L$ 等价于 PSNR 中的 $MaxValue$，$K_1, K_2 << 1$是很小的常数，默认 $K_1=0.01, K_2=0.03$。论文中设置 $α=β=γ=1$ 和 $C_3 = C_2/2$ 来简化公式：

$$
\mathrm{SSIM}(\mathbf{x},\mathbf{y})=\frac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)}{\left(\mu_x^2+\mu_y^2+C_1\right)\left(\sigma_x^2+\sigma_y^2+C_2\right)} \tag{8}
$$

其中，均值：
$$
\mu_x=\frac1N\sum_{i=1}^Nx_i \tag{9}
$$
标准差：
$$
\sigma_x = \left(\frac{1}{N-1}\sum_{i=1}^N\left(x_i - \mu_x\right)^2\right)^{1/2} \tag{10}
$$
协方差：
$$
\sigma_{xy}=\frac{1}{N-1}\sum_{i=1}^{N}\left(x_i-\mu_x\right)\left(y_i-\mu_y\right) \tag{11}
$$

## 2.2 代码实现

具体代码见github仓库。



> [Github地址](https://github.com/ChamingHong/calculate_psnr_ssim)