import tensorflow as tf
import numpy as np
import os
import skimage.io as io
from skimage import transform
from model import VggNet
from utils import decode_image_transfer, show_image
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 屏蔽通知信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

parser = argparse.ArgumentParser('Style Transfer')
parser.add_argument('--content-image', default='./yoyo.jpeg', type=str, 
                    help='Input content image')
parser.add_argument('--style-image', default='./xiangrikui.jpg', type=str, 
                    help='Input style image')
parser.add_argument('--num-steps', type=int, default=1000, help='Training Steps')
parser.add_argument('--lr', type=float, default=2.0, help='learning rate')
parser.add_argument('--interval', type=int, default=100, help='Interval to display image')
args = parser.parse_args()


def train():
    num_steps = args.num_steps
    content_path = args.content_image
    style_path = args.style_image
    # load image
    content_img = io.imread(content_path)
    style_img = io.imread(style_path)
    content_img = transform.resize(content_img, (300, 300))*255.
    style_img = transform.resize(style_img, (300, 300))*255.
    # 生成的图片
    # target_img = np.copy(content_img)
    target_img = (content_img+style_img)/2
    
    # preprocessing
    content_img = preprocessing(content_img)
    style_img = preprocessing(style_img)
    target_img = preprocessing(target_img)
    # convert target image to variable
    target_img = tf.Variable(target_img)

    # create model
    model = VggNet(model_path='vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    model.trainable = False
    # create optimizer
    optimizer = tf.optimizers.Adam(args.lr)
    # train process
    for step in range(num_steps):
        style_loss, content_loss, loss = train_one_step(style_img, content_img, target_img, optimizer, model)
        if (step+1)%args.interval == 0:
            # show image
            # 注意, tf2.0中tensor和numpy不共享内存
            result = target_img.numpy()
            decode_image_transfer(result)
            # show_image(result)
        
        if step%10 == 0:
            print('Step:[%d/%d]-Style_loss:%.3f-Content_loss:%.3f-Total_loss:%.3f'%\
                (step, num_steps, style_loss, content_loss, loss),
                end='\r', flush=True)
    print()


def preprocessing(img):
    """
    convert to tensor and normalize to -1-1
    """
    img = tf.keras.applications.vgg16.preprocess_input(img)
    img = tf.convert_to_tensor(img)
    # img /= 255.0
    img = tf.dtypes.cast(img, dtype=tf.float32)
    img = tf.expand_dims(img, axis=0)
    return img


def content_loss(feature1, feature2):
    """
    计算生成图像和内容图像之间的内容损失
    feature1: content image feature map - 1xHxWxC
    feature2: style image feature map - 1xHxWxC
    """
    loss = tf.reduce_sum(tf.square(feature1-feature2))
    H, W, C = feature1.get_shape()[1:].as_list()
    loss = loss/(W*C*H)
    return loss

def gram_matrix(feature):
    """
    计算feature map的Gram矩阵
    """
    H, W, C = feature.shape[1:].as_list()
    feature = tf.reshape(feature, (-1, C))
    # transpose_a 表示第一个矩阵转置, transpose_b表示第二个矩阵转置
    gram_mtx = tf.matmul(feature, feature, transpose_a=True)
    # 归一化
    gram_mtx /= H*W*C
    return gram_mtx


def style_loss(feature1, feature2):
    """
    计算生成图像和风格图像之间的风格损失
    feature1: content image feature map - 1xHxWxC
    feature2: style image feature map - 1xHxWxC
    """
    gram_mtx1 = gram_matrix(feature1)
    gram_mtx2 = gram_matrix(feature2)
    loss = tf.reduce_mean(tf.square(gram_mtx1 - gram_mtx2))
    return loss


@tf.function
def train_one_step(style_img, content_img, target_img, optimizer, model):
    with tf.GradientTape() as tape:
        style_outputs = model(style_img)
        content_outputs = model(content_img)
        target_outputs = model(target_img)
        # style losses计算所有层的风格损失
        # content losses仅仅计算二层损失
        style_losses, content_losses = 0.0, 0.0
        for idx in range(len(target_outputs)):
            if idx<=4:
                style_losses += style_loss(target_outputs[idx], style_outputs[idx])
            else:
                content_losses += content_loss(target_outputs[idx], content_outputs[idx])
        loss = style_losses + content_losses
        
    
    grads = tape.gradient(loss, [target_img])
    optimizer.apply_gradients(zip(grads, [target_img]))
    
    return style_losses, content_losses, loss


if __name__ == "__main__":
    train()
