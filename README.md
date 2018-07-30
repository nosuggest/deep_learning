# 介绍
非专业的nlp、图像工程师，tensorflow调包工程师，注重快速实现，那么，让我们一起做点有趣的事情吧！
这个工程的目的是将自己在工程和学术研究中，一些应用到深度学习的思路和方法整理汇总出来：

- 帮助深度学习入门同学快速上手
- 提供一些在现有的机器学习方向上的新的方向和思路的整理
- 拓宽思维的禁锢

关于深度学习相关的浅入浅出的介绍，可以快速入门一下[浅入浅出深度学习理论实践](http://shataowei.com/2018/02/07/浅入浅出深度学习理论实践/)。

关于深度学习相关的点击预估的介绍，可以快速入门一下[yoho注册概率预估](http://shataowei.com/2018/03/04/yoho注册概率预估/)。

关于深度学习相关的图像识别的介绍，可以快速入门一下[基于SSD下的图像内容识别（一）](http://shataowei.com/2017/12/01/基于SSD下的图像内容识别（一）/)，[基于SSD下的图像内容识别（二）](http://shataowei.com/2017/12/01/基于SSD下的图像内容识别（二）/)。

关于深度学习相关的目标向量化的介绍，可以快速入门一下[深度学习下的电商商品推荐](http://shataowei.com/2017/08/19/深度学习下的电商商品推荐/)。

关于深度学习相关的多层感知机的介绍，可以快速入门一下[基于Tensorflow实现多层感知机网络MLPs](http://shataowei.com/2018/07/25/基于Tensorflow实现多层感知机网络MLPs/)。

关于深度学习相关的deepfm的介绍，可以快速入门一下[基于Tensorflow实现DeepFM](http://shataowei.com/2018/07/30/基于Tensorflow实现DeepFM/)。

关于深度学习相关的Deep Neural Networks for YouTube Recommendations的介绍，可以快速入门一下[利用DNN做推荐的实现过程总结](https://zhuanlan.zhihu.com/p/38638747)。

# 项目
## RNN_applied_classification
利用GRU，提取用户的行为时序稀疏特征，并产出stack初始层的思路

![](http://upload-images.jianshu.io/upload_images/1129359-d5b28a58edc73240.jpeg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## CNN_applied_classification
利用全连接+CNN，提取稀疏特征，并产出stack初始层的思路

**网络结构**
![](http://upload-images.jianshu.io/upload_images/1129359-59c552e6a61b37e7.jpeg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## Wide & Deep
Google 推荐算法的代码修正，原始代码来源于网络但是不能执行及流程不完整，修复代码demo，现可以直接复制后使用

**网络结构**

![](http://upload-images.jianshu.io/upload_images/1129359-e90396f9e07c4af7.jpeg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

**注意**

在Linux环境下，tensorflow==1.0.0会有如下的报错，而MacBook环境下，tensorflow=1.0.0就不会报错：

```python
double free or corruption (!prev): 0x0000000001f03dd0 ***
```
解决方法是更新版本到1.6.0（其他版本我没试），官方之前有人提过[issue](https://github.com/tensorflow/tensorflow/issues/15848)，大家注意一哈！

## SSD_object_recognition
利用ssd直接实现物体区域识别

**图片版效果：**

![](http://upload-images.jianshu.io/upload_images/1129359-6d4fd382feeb6239.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

**视频版本地址：**

[行人识别场景](https://v.qq.com/x/page/b0548k7ffel.html)

[车辆识别场景](https://v.qq.com/x/page/a0567wd27jz.html)

## Word2vec_recommend
利用样本频率+Huffman树路径最大概率的方法，实现特征向量化的思路
![](http://upload-images.jianshu.io/upload_images/1129359-612db0b5dc8c9041.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## MLPs
最简单的入门级神经网络算法
![多层感知机网络](https://upload-images.jianshu.io/upload_images/1129359-967dcdad03d8ff41.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## DeepFm
入门级别的CTR Prediction的神经网络算法
![DeepFM的网络结构图](https://upload-images.jianshu.io/upload_images/1129359-9e634bcced58d53f.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## Deep Neural Networks for YouTube Recommendations
最近在利用来自google的YouTube团队发表在16年9月的RecSys会议的论文Deep Neural Networks for YouTube Recommendations做用户个性化商品推荐，看到不少论文上的理论总结分析，都很精彩，我手动实现了一遍，总结了一些实际工程中的体会，给大家也给自己一个总结交代。
![Deep Neural Networks for YouTube Recommendations](https://upload-images.jianshu.io/upload_images/1129359-67a74922f9908400.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

normal_version：按照论文未修改的basemodel
attention_version：在basemodel的基础上，加了attention机制（线性attention/rnnattention）
record_dataformat_version：在basemodel的基础上，利用record机制存储数据，加快训练速度


# 工具
- python 3.6
- tensorflow 1.0.0
- nltk 3.2.4
- jieba 0.39
- data_preprocessing 0.0.2

# 其他
鄙人才疏学浅，不免有错误的地方，如果你发现了，麻烦通过以下的方式告知：
- WeChat:sharalion
- Issue
- E-mail:stw386@sina.com
- [Message Board in my bolg](http://shataowei.com)
- 公众号：ml_trip

![](https://upload-images.jianshu.io/upload_images/1129359-654dc61c581d94e1.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
