---
layout: post
title: 'OpenCV读取图像的坐标系问题'
date: 2024-08-05
author: 洪茬铭
cover: ''
tags: python
---

> **本文目录：**
>  1. OpenCV 读取后的图片的属性
>  2. 实验验证
>  3. 总结

# 1. OpenCV 读取后的图片的属性

使用 `cv2.imread()` 函数读取彩色图像后，图像的存储的数据范围是 `[0, 255]` 的 `<class 'numpy.uint8'>`，通道顺序为 `BGR`，当使用 `matplotlib.pyplot` 绘制时需要使用 `cv2.cvtColor()` 函数来转换为支持的 `RGB` 顺序。

读取的图片 **左上角点为原点，水平向右为 X 轴正方向，垂直向下为 Y 轴正方向**。

![](https://pic.imgdb.cn/item/66b0d6a7d9c307b7e985d65a.jpg)

如果图片的高度是 H，宽度为 W，那么存储的数据尺寸为 `(H,W,3)`，底层存储方式与我们肉眼的直观感觉是一致的。H 对应着 y 轴，W 对应着 x 轴，也就是说，使用 OpenCV 访问某一位置的像素值时，不是先确定 `x` 坐标再确定 `y` 坐标，而是相反。也就是说，对于**坐标系下**的 `(x,y)` 这一点，我们是使用 `(y,x)` 来访问的。

# 2. 实验验证
使用的上述图片高度为 328，宽度为 400， 

- **shape() —— 先高再宽**

```python
img = cv2.imread('dog.jpg')
print(img.shape)

>>> (328, 400, 3)
```

- **resize() —— 先 x 再 y**

```python
img = cv2.imread('dog.jpg')
resized_img = cv2.resize(img, (0, 0), fx=0.5, fy=1.0)  # 这里的fx对应着x轴，fy对应着y轴
```

![](https://pic.imgdb.cn/item/66b0d9cfd9c307b7e988dfdd.jpg)

```python
img = cv2.imread('dog.jpg')
resized_img = cv2.resize(img, (100, 200))  # 先x轴，再y轴
```

![](https://pic.imgdb.cn/item/66b0da6ed9c307b7e9897573.jpg)

**在电脑中图片的属性中显示的分辨率也是先 x 再 y**

![](https://pic.imgdb.cn/item/66b0db2dd9c307b7e98aa0e4.jpg)

- **访问像素值 —— 先高后宽**

```python
img = cv2.imread('dog.jpg')
img[:200, :300, :] = 0
```

![](https://pic.imgdb.cn/item/66b0db6fd9c307b7e98ae421.jpg)



# 3. 总结

除了在文件属性和 `resize` 时是以坐标轴为顺序（先宽后高），其他操作可以将其看作是普通的三维矩阵即可，第一维为行（高），第二维为列（宽），第三维为通道。















