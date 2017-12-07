import os
#using GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
config = tf.ConfigProto()
#use 80% of the GPU memory
config.gpu_options.per_process_gpu_memory_fraction = 1
session = tf.Session(config=config)

from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Input
from keras.layers.convolutional import Conv2D
from keras.layers.merge import concatenate
from keras import backend as K
from keras import optimizers

import matplotlib.pyplot as plt
import cv2
import numpy
import math
import h5py

def psnr(y_true, y_pred):
    mse = K.mean(K.square(y_true[:,:,0] - y_pred[:,:,0]), axis=(-3, -2))
    mse=K.mean(mse)
    return 20 * K.log(1. / K.sqrt(mse)) / numpy.log(10)#Attention,if you normalization pixels ,you use 1 not 255

def ssim(y_true, y_pred):
    K1 = 0.04
    K2 = 0.06
    mu_x = K.mean(y_pred)
    mu_y = K.mean(y_true)
    sig_x = K.std(y_pred)
    sig_y = K.std(y_true)
    sig_xy = K.mean(y_pred * y_true) - mu_x * mu_y
    L =  255
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    ssim = (2 * mu_x * mu_y + C1) * (2 * sig_xy * C2) * 1.0 / ((mu_x ** 2 + mu_y ** 2 + C1) * (sig_x ** 2 + sig_y ** 2 + C2))
    return ssim

def create_model(img_height,img_width,img_channel):
    ip = Input(shape=(img_height, img_width,img_channel))
    L1 = Conv2D(32, (11, 11), padding='same', activation='relu', kernel_initializer='glorot_uniform')(ip)
    L2 = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='glorot_uniform')(L1)
    L3 = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='glorot_uniform')(L2)
    L4 = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='glorot_uniform')(L3)
    L4=concatenate([L4,L1],axis=-1)
    #L4 = BatchNormalization(momentum=bn_num)(L4)
    L5 = Conv2D(64, (1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform')(L4)
    L6 = Conv2D(64, (5, 5), padding='same', activation='relu', kernel_initializer='glorot_uniform')(L5)
    L6=concatenate([L6,L1],axis=-1)
    L7 = Conv2D(128, (1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform')(L6)
    L8 = Conv2D(img_channel, (5, 5), padding='same', activation='relu', kernel_initializer='glorot_uniform')(L7)
    deblocking =Model(inputs=ip,outputs= L8)
    return deblocking

'''
from keras.utils import plot_model
plot_model(deblocking, to_file='model1.png', show_shapes=True, show_layer_names=True)
'''
def main():
    im = plt.imread('test/input/433.png')
    h,w = im.shape
    c = 1
    im = im.reshape(1,h,w,c)
    print(im.shape)
    l8 = create_model(h,w,c)
    l8.load_weights('models/198-0.0008910.hdf5')
    out = l8.predict(im)
    print(out.shape)
    out = out.reshape(h,w)
    out[out > 1] = 1
    out[out < 0] = 0
    print(out.shape)
    #plt.imsave('test/resutl/result.bmp',out)
    cv2.imwrite('test/result/result-y.bmp',out*255.0)

if __name__ == "__main__":
    main()
