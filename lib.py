import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf

import models.scripts as sc
from models.scripts import resolve_single
from keras_preprocessing import image as img


def eraser():
    folders = ['static/input', "static/output/"]
    for f in folders:
        for filename in os.listdir(f):
            os.unlink(os.path.join(f, filename))

def load_weights() :

    gan_weights = "models/weights/gan_generator.h5"
    gan_generator = sc.generator()
    gan_generator.load_weights(gan_weights)

    return gan_generator

def resizeImage(image):
    if image.height > 250 or image.width > 250:
        resizeRatio = min(250/image.width, 250/image.height)

        newHeight = int(resizeRatio * image.height)
        newWidth = int(resizeRatio * image.width) 

        newSize = (newWidth, newHeight)
        return image.resize(newSize, Image.ANTIALIAS)
    return image

def preprocess(image):
    if image.mode == 'RGBA':
        # Create a blank background image
        bg = Image.new('RGB', image.size, (255, 255, 255))
        # Paste image to background image
        bg.paste(image, (0, 0), image)
        return bg
    return image

def load_image(path):
    img = Image.open(path)
    
    img = preprocess(img)
    img = resizeImage(img)

    return np.array(img)

def cal_psnr(img1, img2):
    psnr1 = tf.image.psnr(load_image(img1), load_image(img2), max_val=255)
    print("PSNR Value:", tf.keras.backend.get_value(psnr1))

def save_image(img, path):
    sizes = np.shape(img)
    height = float(sizes[0])
    width = float(sizes[1])

    fig = plt.figure()
    fig.set_size_inches(width/height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(img)
    plt.savefig(path, dpi = height) 
    plt.close()

def plotter(path, name) :
    
    lr = load_image(path)
    
    gan_generator = load_weights()
    gan_sr = resolve_single(gan_generator, lr)
    
    save_image(gan_sr, name)
