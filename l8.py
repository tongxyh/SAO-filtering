import os
#using GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
config = tf.ConfigProto()
#use 80% of the GPU memory
config.gpu_options.per_process_gpu_memory_fraction = 1
session = tf.Session(config=config)

from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Input,BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.merge import concatenate
from keras import backend as K
from keras import optimizers
#from skimage.measure import compare_ssim as ssim

import matplotlib.pyplot as plt
import numpy
import math
import h5py

def psnr(y_true, y_pred):
    mse = K.mean(K.square(y_true[:,:,0] - y_pred[:,:,0]), axis=(-3, -2))
    mse = K.mean(mse)
    return 20 * K.log(1. / K.sqrt(mse)) / numpy.log(10)#Attention,if you normalization pixels ,you use 1 not 255

def ssim(y_true, y_pred):
    K1 = 0.04 #0.01
    K2 = 0.06 #0.03
    mu_x = K.mean(y_pred)
    mu_y = K.mean(y_true)
    sig_x = K.std(y_pred)
    sig_y = K.std(y_true)
    sig_xy = K.mean(y_pred * y_true) - mu_x * mu_y
    L =  1.0
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    ssim = (2 * mu_x * mu_y + C1) * (2 * sig_xy + C2) * 1.0 / ((mu_x ** 2 + mu_y ** 2 + C1) * (sig_x ** 2 + sig_y ** 2 + C2))
    return ssim

def create_model(img_height,img_width,img_channel):
    ip = Input(shape=(img_height, img_width,img_channel))
    L1 = Conv2D(32, (11, 11), padding='same', activation='relu', kernel_initializer='glorot_uniform')(ip)
    L2 = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='glorot_uniform')(L1)
    L3 = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='glorot_uniform')(L2)
    L4 = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='glorot_uniform')(L3)
    L4=concatenate([L4,L1],axis=-1)
    #L4 = BatchNormalization(momentum=0.9)(L4)
    L5 = Conv2D(64, (1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform')(L4)
    L6 = Conv2D(64, (5, 5), padding='same', activation='relu', kernel_initializer='glorot_uniform')(L5)
    L6=concatenate([L6,L1],axis=-1)
    L7 = Conv2D(128, (1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform')(L6)
    L8 = Conv2D(img_channel, (5, 5), padding='same', kernel_initializer='glorot_uniform')(L7)
    deblocking =Model(inputs=ip,outputs= L8)
    optimizer = optimizers.Adam(lr=1e-4)
    deblocking.compile(optimizer=optimizer,loss='mean_squared_error', metrics=[psnr,ssim)
    return deblocking

'''
from keras.utils import plot_model
plot_model(deblocking, to_file='model1.png', show_shapes=True, show_layer_names=True)
'''
def main():
    file_train = h5py.File(r'train64.h5','r')
    train_data = file_train['data'][:].transpose(0,3,2,1)
    train_label = file_train['label'][:].transpose(0,3,2,1)

    file_test = h5py.File(r'test64.h5','r')
    test_data = file_test['data'][:].transpose(0,3,2,1)
    test_label = file_test['label'][:].transpose(0,3,2,1)

    #plt.imsave('test/result/train_in.png',train_data[1,:,:,:])
    #plt.imsave('test/result/train_out.png',train_label[1,:,:,:])
    #plt.imsave('test/result/test_in.png',test_data[1,:,:,:])
    #plt.imsave('test/result/test_out.png',test_label[1,:,:,:])
    #ssim_noise = ssim(img, img_noise,data_range=img_noise.max() - img_noise.min())

    fvc_model = create_model(64,64,1)
    checkpoint=ModelCheckpoint(filepath=r'models/1/{epoch:02d}-{val_loss:.7f}.hdf5',period=1)
    fvc_model.fit(train_data, train_label,
                    epochs=200,
                    batch_size=64,
                    shuffle=True,
                    validation_data=(test_data, test_label),callbacks=[checkpoint])

if __name__ == "__main__":
    main()
