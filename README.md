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

![](https://s2.loli.net/2024/03/04/uoK6R5xdLFPA1fS.png)

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

![](https://s2.loli.net/2024/03/04/QLCpH2sygVEGJFf.png)

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

![](https://s2.loli.net/2024/03/04/nliS39p4VrxJoHC.png)

训练完成后就会保存网络的模型以及权重

![](https://s2.loli.net/2024/03/04/8do1SeEzr2s9Tna.png)

### RKNN模型转换

[GitHub - rockchip-linux/rknn-toolkit2](https://github.com/rockchip-linux/rknn-toolkit2/tree/master)

![](https://s2.loli.net/2024/03/04/8u5sYXoC6Ewmjc4.png)

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

在仓库里面提供了pt2rknn.py,在安装rknn_toolkit2时候，请使用阿里云镜像

[pypi镜像_pypi下载地址_pypi安装教程-阿里巴巴开源镜像站](https://developer.aliyun.com/mirror/pypi?spm=a2c6h.13651102.0.0.2ac31b11vYRRy3)

```bash
python pt2rknn.py
```

![](https://s2.loli.net/2024/03/04/9E6Tn8L3GQobPu2.png)

![](https://s2.loli.net/2024/03/04/XNwQbtRVuZD7ISe.png)

pt2rknn代码中，存在一个量化选项，填入  `True`进行整型量化，但实验测试其精度损失严重

```python
ret = rknn.build(do_quantization=False, dataset='./dataset.txt')
```

### 模型在泰山派上部署

请在在泰山派上下载官方提供的Debian10镜像，Ubuntu系统测试有一定点Bug

然后我喜欢在PC机上远程连接泰山派，使用的软件是 [*MobaXterm*](http://www.baidu.com/link?url=KIi6MGunswJXl6_aI59nxuXANCpEwI7EnC8cqee5IRSPaQ0vEBG48W6oVDGsBWZ4)

![](https://s2.loli.net/2024/03/04/fqDJgbxoI3lSLAc.png)

    sudo apt update
    sudo apt upgrade

安装miniconda

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zs
source .bashrc
```

同时建议换源[anaconda | 镜像站使用帮助 | 清华大学开源软件镜像站 | Tsinghua Open Source Mirror](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)

下载更新NPU驱动https://github.com/rockchip-linux/rknpu2

```bash
git clone https://github.com/rockchip-linux/rknpu2.git #建议PC下载后利用Mobaxterm开发板上
```

```bash
cp rknpu2/runtime/RK356X/Linux/rknn_server/aarch64/usr/bin/rknn_server /usr/bin/rknn_server
cp rknpu2/runtime/RK356X/Linux/librknn_api/aarch64/librknnrt.so /usr/lib/librknnrt.so
cp rknpu2/runtime/RK356X/Linux/librknn_api/aarch64/librknn_api.so /usr/lib/librknn_api.so
```

```bash
 bash rknpu2/runtime/RK356X/Linux/rknn_server/aarch64/usr/bin/start_rknn.sh
                                                                                                                 bin/start_rknn.sh
```

创建Python环境

```bash
conda create -n rknn python=3.10
conda activate rknn
sudo apt update
sudo apt upgrade
sudo apt install gcc
```

https://github.com/rockchip-linux/rknn-toolkit2/tree/master/rknn_toolkit_lite2/packages

选取合适的包下载并安装安装

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install rknn_toolkit_lite2-1.6.0-cp310-cp310-linux_aarch64.whl
pip install opencv-python
```

到此板卡环境就安装好了

运行我们的测试代码

    python test.py
