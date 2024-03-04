# 泰山派嵌入式AI开发实战教程

## 简介

    随着AI技术在各行业快速发展，将AI技术与嵌入式系统结合，构建边缘计算成为技术热点之一。嵌入式神经网络处理器（NPU）采用了针对神经网络计算优化的计算架构，能够更快速地执行矩阵乘法等神经网络运算，同时低功耗、高并行等特点，能够有效支持各种人工智能应用的运行。
    泰山派上的RK3566搭载了0.8TOPS的NPU, ,具备一定的AI算力，同时，瑞芯微官方提供了RKNN组件支持主流TensorFlow、TF-lite、Pytorch、Caffe、ONNX等深度学习框架，能很方便进行算法的端侧部署。
    瑞芯微提供了RKNPU2， RKNN Toolkit2等组件。RKNPU2提供了运行库和+编程接口，用来部署推理一种根据NPU硬件架构定义的一套模型格式模型

## MobileNetV3

    MobileNetV3是由G

### 模型训练

#### 基础环境安装

设备：一台高性能PC，Linux或Windows都可以，需要安装conda环境

[Miniconda &#8212; Anaconda documentation](https://docs.anaconda.com/free/miniconda/)

```bash
source .bashrc #这个命令不能在windows系统上运行，请打开Anaconda Prompt (Miniconda3)
```

![](C:\Users\13890\AppData\Roaming\marktext\images\2024-03-03-16-37-39-image.png)

conda在国内需要换源获得更高的下载速度[anaconda | 镜像站使用帮助 | 清华大学开源软件镜像站 | Tsinghua Open Source Mirror](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)

[pypi | 镜像站使用帮助 | 清华大学开源软件镜像站 | Tsinghua Open Source Mirror](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)

安装完成后我们需要创建一个Python环境

```bash
conda create -n moblienet python=3.10
```

然后激活环境

```bash
conda activate moblienet
```

![](C:\Users\13890\AppData\Roaming\marktext\images\2024-03-03-16-38-05-image.png)

### 安装torch

https://pytorch.org/get-started/locally/

```bash
pip3 install torch torchvision torchaudio
```

### 数据集的创建

本次图像分类用的的动物的5分类数据集，包括：猫(cat)、牛(cattle)、狗(dog)、马(house)、猪(pig)

```bash
python train.py 
```

<img src="file:///C:/Users/13890/AppData/Roaming/marktext/images/2024-03-03-18-49-11-image.png" title="" alt="" data-align="center">

训练完成后就会保存网络的模型以及权重

![](C:\Users\13890\AppData\Roaming\marktext\images\2024-03-03-18-54-12-image.png)

### RKNN模型转换

[GitHub - rockchip-linux/rknn-toolkit2](https://github.com/rockchip-linux/rknn-toolkit2/tree/master)

![](C:\Users\13890\AppData\Roaming\marktext\images\2024-03-03-20-52-23-image.png)

模型转换需要一台ubuntu的机器，并克隆rknn-toolkit2仓库,同时新建一个conda 环境

```bash
conda create -n rknn python=3.10
```

```bash
git clone https://github.com/rockchip-linux/rknn-toolkit2.git
```

```bash
# 根据自己的python版本选择txt和wheel
pip install  -r rknn-toolkit2/rknn-toolkit2/packages/requirements_cp310-1.6.0.txt 
cd rknn-toolkit2/rknn-toolkit2/packages/
pip install rknn_toolkit2-1.6.0+81f21f4d-cp310-cp310-linux_x86_64.whl 
```

在仓库里面提供了pt2rknn.py

![](C:\Users\13890\AppData\Roaming\marktext\images\2024-03-03-20-52-00-image.png)

![](C:\Users\13890\AppData\Roaming\marktext\images\2024-03-03-21-27-52-image.png)

pt2rknn代码中，存在一个量化选项，填入  `True`进行整型量化，但实验测试其精度损失严重

```python
ret = rknn.build(do_quantization=False, dataset='./dataset.txt')
```

### 模型在泰山派上部署


