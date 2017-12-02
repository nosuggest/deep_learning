# 这边只概述理论，详细参见我的个人博客：
[基于SSD下的图像内容识别](http://shataowei.com/2017/12/01/基于SSD下的图像内容识别（一）/)

# 理论概述
产生candidates_boxs，candidates_boxs通过基本属性的初筛，candidates_boxs根据IOU原则下的NMS进行复选，再将复选出来的box根据你已经训练好的分类模型确定到底是啥

![](http://upload-images.jianshu.io/upload_images/1129359-3eb665a372aa4089.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
