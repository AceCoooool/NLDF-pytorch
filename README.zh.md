# NLFD
[English](./README.md)

基于Pytorch的非官方版本实现： [Non-Local Deep Features for Salient Object Detection](https://sites.google.com/view/zhimingluo/nldf).（未完）

<p align="center"><img width="100%" src="png/example.png" /></p>

官方Tensorflow版本链接: [NLFD](https://github.com/zhimingluo/NLDF)

此实现的几点改动:

1. 数据集（个人没找到MSRA-B的图片）
2. 网络结构上的一些不同：此处采用最后输出为单个概率图，官方版本中是两个互异的概率图

## 依赖库

- [Python 3](https://www.continuum.io/downloads)
- [Pytorch 0.3.0](http://pytorch.org/)
- [torchvision](http://pytorch.org/)
- [visdom](https://github.com/facebookresearch/visdom) (optional for visualization)

## 使用说明

### 1. 复制仓库到本地

```shell
git clone git@github.com:AceCoooool/NLFD-Pytorch.git
cd NLFD-Pytorch/
```

### 2. 从网上下载数据集

注：原始论文中采用更多的数据集

可从下面链接下载数据集： [ECSSD](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html)  

```shell
bash download.sh
```

### 3. 提取预先训练好的VGG

```bash
cd tools/
python extract_vgg.py
cd ..
```

注：此处个人直接采用torchvision里面训练好的VGG

### 4. 训练

```shell
python main.py --mode='train' --train_path='you_data' --label_path='you_label' --batch_size=8 --visdom=True
```

注：

1. `--val=True`：训练阶段开启validation. 你可以将部分训练集作为验证集。同时提供验证集的路径
2. `you_data, you_label` ：关于第2步中数据集的路径

### 5. 示例

```shell
python demo.py --demo_img='your_picture' --trained_model='pre_trained pth' --cuda=True
```

注：这里采用的`pre_trained.pth`来自第4步训练好的模型

### 6. 测试

```shell
python main.py --mode='test', --test_path='you_data' --test_label='your_label' --batch_size=1 --model='your_trained_model'
```

注：目前提供的版本不是标准的测试方式（采用了缩放后的图片和标签），以及max f-measure存在问题

## 训练好的模型

下述仅给出一个迭代了较多轮的结果：[pretrained_model](https://drive.google.com/file/d/17ZpXi9YKTgPeNepvohNyPfnALYhXsC2d/view?usp=sharing)