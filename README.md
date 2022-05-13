# Paddle-FSL-Baseline

## 一、简介
论文：《A Closer Look at Few-shot Classification》[论文链接](https://arxiv.org/abs/1904.04232)

小样本分类旨在通过有限标记样例学习一个分类器来识别未知类，虽然近些年取得了一些重要的研究进展，但各方法网络结构、元学习算法的复杂性以及实现细节的差异为比较当前进展提出挑战。
论文提出：

1. 对几种代表性的小样本分类算法进行一致性比较分析，结果证明随着特征提取神经网络模型能力的提升，在给定领域差异的数据集上，各方法的性能差异显著缩小；

2. 提出了一个基准模型，该模型在mini-ImageNet等数据集上的性能可以媲美几种SOTA方法；

3. 提出了一种新的用于评估小样本分类算法跨领域泛化能力的实验设定，结果发现当特征提取神经网络能力较弱时，减少类内差异是提升模型性能的一个重要因素， 当特征提取神经网络能力较强时，类内差异不再关键。

在一个实际的跨领域设定中，作者发现基准模型+微调的方式可以得到比SOTA更好的性能表现。

[参考项目地址链接](https://github.com/wyharveychen/CloserLookFewShot)
## 二、复现精度
代码在miniImageNet数据集下训练和测试。

5-way Acc：

| |1-shot|5-shot|
| :---: | :---: | :---: |
|论文|48.2% |66.4%|
|复现|48.5% |66.6%|

## 三、数据集
2016年google DeepMind团队从Imagnet数据集中抽取的一小部分（大小约3GB）制作了Mini-Imagenet数据集，共有100个类别，每个类别都有600张图片，共60000张（都是.jpg结尾的文件）。

Mini-Imagenet数据集中还包含了train.csv、val.csv以及test.csv三个文件。

* train.csv包含38400张图片，共64个类别。
* val.csv包含9600张图片，共16个类别。
* test.csv包含12000张图片，共20个类别。

每个csv文件之间的图像以及类别都是相互独立的，即共60000张图片，100个类。


## 四、环境依赖
paddlepaddle-gpu==2.2.2

## 五、快速开始

本项目5-way分类可设1-shot和5-shot。如果用5-shot可设置--n_shot 5，用1-shot可设置--n_shot 1。下面以5-shot为例。

### step1: 加载数据集
下载MiniImagenet数据集文件放在本repo的./filelists下

可以在这里下载[MiniImagenet数据集](https://aistudio.baidu.com/aistudio/datasetdetail/138415)


### step2: 训练

```bash
python3 train.py --n_shot 5
```

训练的模型保存在本repo的./record目录下

训练的日志保存在本repo的./logs目录下

### step3: 保存特征

将提取的特征保存在分类层之前，以提高测试速度。加载./record目录下模型进行特征保存

```bash
python3 save_features.py --n_shot 5
```

### step4: 测试

```bash
python3 test.py --n_shot 5
```

测试时程序会加载本repo的./record下保存的训练模型文件。

可以[下载训练好的模型数据](https://aistudio.baidu.com/aistudio/datasetdetail/140016) checkpoint_clfs.zip，放到本repo的./record下。 然后直接执行第step3保存特征和第step4测试命令。

## 六、代码结构与参数说明

### 6.1 代码结构

```
├─data                            # 数据处理包
├─filelists                       # 数据文件
├─methods                         # 模型方法
├─logs                            # 训练日志
├─record                          # 训练保存文件 
│  backbone.py                    # 特征提取网络
│  configs.py                     # 配置文件  
│  io_utils.py                    # 配置文件
│  README.md                      # readme
│  save_features.py               # 保存特征
│  train.py                       # 训练
│  test.py                        # 测试
```
### 6.2 参数说明

可以在io_utils.py文件中查看设置训练与测试相关参数

## 七、模型信息

关于模型的其他信息，可以参考下表：

| 信息 | 说明 |
| --- | --- |
| 发布者 | Lieber |
| 时间 | 2022.04 |
| 框架版本 | Paddle 2.2.2 |
| 应用场景 | 小样本 |
| 支持硬件 | GPU、CPU |
| 下载链接 | [最优模型](https://aistudio.baidu.com/aistudio/datasetdetail/140016)|
| 在线运行 | [notebook](https://aistudio.baidu.com/aistudio/projectdetail/3793411)|