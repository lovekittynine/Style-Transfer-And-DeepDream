import matplotlib.pyplot as plt
import numpy as np
import os


def decode_image_transfer(img):
    """
    将输出图片进行解码
    img： numpy array
    """
    
    img = np.squeeze(img)
    # for style transform
    # convert to [0-255]
    img += np.array([[[103.939, 116.779, 123.68]]])
    # convert to RGB
    img = img[...,::-1]
    # for deep dream
    # img = img*255.0
    # convert to uint8
    img = np.clip(img, 0, 255).astype(np.uint8)
    plt.imsave('./yoyo_G.jpg', img)
    
    
def decode_image_deepdream(img):
    """
    将输出图片进行解码
    img： numpy array
    """
    
    img = np.squeeze(img)
    # for deep dream
    img = img*255.0
    # convert to uint8
    img = np.clip(img, 0, 255).astype(np.uint8)
    plt.imsave('./we6.jpg', img)

   
def show_image(img, save=True):
    plt.ion()
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    plt.ioff()
    plt.close()
    if save:
        plt.imsave('./generate.jpg', img)

