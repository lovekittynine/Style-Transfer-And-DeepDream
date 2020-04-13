#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 22:46:56 2020

@author: wsw
"""

import tensorflow as tf
import numpy as np
import os
import skimage.io as io
from skimage import transform
from model import VggNet
from utils import decode_image_deepdream, show_image
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 屏蔽通知信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

parser = argparse.ArgumentParser('Style Transfer')
parser.add_argument('--content-image', default='./style.jpg', type=str, 
                    help='Input content image')
parser.add_argument('--num-steps', type=int, default=1000, help='Training Steps')
parser.add_argument('--lr', type=float, default=10.0, help='learning rate')
parser.add_argument('--interval', type=int, default=100, help='Interval to display image')
args = parser.parse_args()

"""
Note: DeepDream needs to normalize input image into [0-1], and the network will quickly converge
"""

def train():
    num_steps = args.num_steps
    content_path = args.content_image
    # load image
    content_img = io.imread(content_path)
    
    # preprocessing
    content_img = preprocessing(content_img)
    # convert target image to variable
    content_img = tf.Variable(content_img)
    
    # create model
    model = VggNet(model_path='vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    model.trainable = False
    # create optimizer
    # optimizer = tf.optimizers.Adam(args.lr)
    # train process
    
    for step in range(num_steps):
        content_loss = train_one_step(content_img, model)
        if (step+1)%args.interval == 0:
            # show image
            # 注意, tf2.0中tensor和numpy不共享内存
            result = content_img.numpy()
            result = decode_image_deepdream(result)
            # show_image(result)
        
        if step%10 == 0:
            print('Step:[%d/%d]-Content_loss:%.3f'%\
                (step, num_steps, content_loss),
                end='\r', flush=True)
        
    print()
    

def preprocessing(img):
    """
    convert to tensor and normalize to -1-1
    """
    # what a fuck keng!!!!
    # note transform.resize will normalize image to [0-1]
    img = transform.rescale(img, 0.5, multichannel=True)
    """
    img = img*255
    img = tf.keras.applications.vgg16.preprocess_input(img)
    """
    img = tf.convert_to_tensor(img)
    img = tf.dtypes.cast(img, dtype=tf.float32)
    img = tf.expand_dims(img, axis=0)
    return img


def activation_loss(feature):
    """
    计算activation loss
    feature1: content image feature map - 1xHxWxC
    feature2: style image feature map - 1xHxWxC
    """
    return tf.reduce_mean(feature)
  

@tf.function
def train_one_step(content_img, model):
    with tf.GradientTape() as tape:
    
        content_outputs = model(content_img)
        content_losses = 0.0
        for output in content_outputs[:5]:
            content_losses += activation_loss(output)
        
    grads = tape.gradient(content_losses, content_img)
    content_img.assign_add(args.lr*grads)
    return content_losses


if __name__ == "__main__":
    train()
