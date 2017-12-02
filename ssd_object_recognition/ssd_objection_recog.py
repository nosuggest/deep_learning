#pip install moviepy
#git clone https://github.com/balancap/SSD-Tensorflow
#unzip ./SSD-Tensorflow/checkpoints/ssd_300_vgg.ckpt.zip

import os
import math
import random
import sys
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as mpcm
sys.path.append('./SSD-Tensorflow/')
from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing

# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)

slim = tf.contrib.slim

#%matplotlib inline

l_VOC_CLASS = [
                'aeroplane',   'bicycle', 'bird',  'boat',      'bottle',
                'bus',         'car',     'cat',   'chair',     'cow',
                'diningTable', 'dog',     'horse', 'motorbike', 'person',
                'pottedPlant', 'sheep',   'sofa',  'train',     'TV'
]

# 定义数据格式
net_shape = (300, 300)
data_format = 'NHWC'  # [Number, height, width, color]，Tensorflow backend 的格式

# 预处理，以 Tensorflow backend, 将输入图片大小改成 300x300，作为下一步输入
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, 
    None, 
    None, 
    net_shape, 
    data_format, 
    resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE
)
image_4d = tf.expand_dims(image_pre, 0)

# 定义 SSD 模型结构
reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_300.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

# 导入官方给出的 SSD 模型参数
ckpt_filename = '/Users/slade/SSD-Tensorflow/checkpoints/ssd_300_vgg.ckpt'
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

ssd_anchors = ssd_net.anchors(net_shape)

def colors_subselect(colors, num_classes=21):
    dt = len(colors) // num_classes
    sub_colors = []
    for i in range(num_classes):
        color = colors[i*dt]
        if isinstance(color[0], float):
            sub_colors.append([int(c * 255) for c in color])
        else:
            sub_colors.append([c for c in color])
    return sub_colors


def bboxes_draw_on_img(img, classes, scores, bboxes, colors, thickness=5):
    shape = img.shape
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i]
        color = colors[classes[i]]
        # Draw bounding box...
        p1 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
        p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
        cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)
        # Draw text...
        s = '%s:%.3f' % ( l_VOC_CLASS[int(classes[i])-1], scores[i])
        p1 = (p1[0]-5, p1[1])
        cv2.putText(img, s, p1[::-1], cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

colors_plasma = colors_subselect(mpcm.plasma.colors, num_classes=21)

def process_image(img, select_threshold=0.3, nms_threshold=.8, net_shape=(300, 300)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})

    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, ssd_anchors,
            select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)

    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    bboxes_draw_on_img(img, rclasses, rscores, rbboxes, colors_plasma, thickness=2)
    return img


img = cv2.imread("/Users/slade/Desktop/yoho/picture_recognize/test7.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(process_image(img))
plt.show()




import imageio
imageio.plugins.ffmpeg.download()
from moviepy.editor import VideoFileClip

def process_video (input_path, output_path):
    clip = VideoFileClip (input_path)
    result = clip.fl_image(process_image)
    %time result.write_videofile(output_path, audio=False)

process_video("/Volumes/slade/yoho/picture_recognize/sight_test6.mp4", "/Volumes/slade/yoho/picture_recognize/sight_test6_res.mp4")
