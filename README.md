# NLDF
[中文说明](./README.zh.md)

An unofficial implementation of [Non-Local Deep Features for Salient Object Detection](https://sites.google.com/view/zhimingluo/nldf).

<p align="center"><img width="100%" src="png/example.png" /></p>

The official Tensorflow version: [NLDF](https://github.com/zhimingluo/NLDF)

Some thing difference:

1. ~~dataset~~
2. score with one channel, rather than two channels
3. Dice IOU: boundary version and area version

## Prerequisites

- [Python 3](https://www.continuum.io/downloads)
- [Pytorch 1.0](http://pytorch.org/)
- [torchvision](http://pytorch.org/)
- [visdom](https://github.com/facebookresearch/visdom) (optional for visualization)

## Results

The information of Loss:

![](./png/loss.png)

Performance:

| Dataset | max F(paper) | MAE(paper) | max F(here) | MAE(here) |
| :-----: | :----------: | :--------: | :---------: | :-------: |
| MSRA-B  |    0.911     |   0.048    |   0.9006    |  0.0592   |

Note: 

1. only training 200 epoch, larger epoch may nearly the original paper
2. This reproduction use area IOU, and original paper use boundary IOU
3. ~~it's unfairness to this compare. (Different training data, I can not find the dataset use in original paper )~~

## Usage

### 1. Clone the repository

```shell
git clone git@github.com:AceCoooool/NLDF-pytorch.git
cd NLDF-pytorch/
```

### 2. Download the dataset

Note: the original paper use other datasets.

Download the [ECSSD](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html) dataset.  

```shell
bash download.sh
```

### 3. Get pre-trained vgg

```bash
cd tools/
python extract_vgg.py
cd ..
```

### 4. Demo

```shell
python demo.py --demo_img='your_picture' --trained_model='pre_trained pth' --cuda=True
```

Note: 

1. default choose: download and copy the [pretrained model](https://drive.google.com/file/d/10cnWpqABT6MRdTO0p17hcHornMs6ggQL/view?usp=sharing) to `weights` directory.
2. a demo picture is in `png/demo.jpg`

### 5. Train

```shell
python main.py --mode='train' --train_path='you_data' --label_path='you_label' --batch_size=8 --visdom=True --area=True
```

Note:

1. `--area=True, --boundary=True` area and boundary Dice IOU (default: `--area=True --boundary=False`)
2. `--val=True` add the validation (but your need to add the `--val_path` and `--val_label`)
3. `you_data, you_label` means your training data root. (connect to the step 2)

### 6. Test

```shell
python main.py --mode='test', --test_path='you_data' --test_label='your_label' --batch_size=1 --model='your_trained_model'
```

Note:

1. use the same evaluation (this is a reproduction from original achievement)

## Bug

1. The boundary Dice IOU may cause `inf`，it is better to use area Dice IOU.

Maybe, it is better to add Batch Normalization. 