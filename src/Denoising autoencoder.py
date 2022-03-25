'''
1 import libs
'''
import cv2
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, BatchNormalization, ReLU, Conv2DTranspose, Concatenate, Activation
from keras.models import Model, Sequential
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.datasets import mnist, cifar100, cifar10
import numpy as np

'''
2 import train datasets
'''
# (x_train, y_train), (x_test, y_test) = mnist.load_data()  #从mnist导入图片数据准备训练autoencoder
(x_train, _), (x_test, _) = cifar10.load_data()  #从cifar导入图片数据准备训练autoencoder

# 训练前先将数据调整格式
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 32, 32, 3))  # mnist是28,28,1
x_test = np.reshape(x_test, (len(x_test), 32, 32, 3))  # mnist是28,28,1

# np.save('x.train')

# 训练需要将噪声人为的加入，以增加抗噪能力
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

'''
3 train denoising autoencoder 16-8-8
Accuracy: 4th, Speed: 1st
'''
def train_model_16():
    input_img = Input(shape=(32, 32, 3))  # mnist是28,28,1
    # 因为mnist数据库只有黑白两种颜色，所以第三个通道为1。而正常彩色的图片第三个通道为3,。而且在导入到autoencoder之前，需要将分辨率统一
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same', name='encoder')(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x) #mnist是1

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    '''
    fit
    '''
    autoencoder.fit(x_train_noisy, x_train,
                    epochs=5,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(x_test_noisy, x_test),
                    callbacks=[TensorBoard(log_dir='/tmp/tb', histogram_freq=0, write_graph=False)])

    autoencoder.save('autoencoder16.h5')

train_model_16()

'''
4 train denoising autoencoder 128-64-32
Accuracy: 5th, Speed: 4th
'''
# def train_model_128up():
#     input_img = Input(shape=(32, 32, 3))
#     x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
#     x = MaxPooling2D((2, 2), padding='same')(x)
#     x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
#     x = MaxPooling2D((2, 2), padding='same')(x)
#     x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
#     encoded = MaxPooling2D((2, 2), padding='same', name='encoder')(x)
#
#     # at this point the representation is (4, 4, 8) i.e. 128-dimensional
#
#     x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
#     x = UpSampling2D((2, 2))(x)
#     x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
#     x = UpSampling2D((2, 2))(x)
#     x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
#     x = UpSampling2D((2, 2))(x)
#     decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x) #mnist是1
#
#     autoencoder = Model(input_img, decoded)
#     autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
#
#     '''
#     fit
#     '''
#     autoencoder.fit(x_train_noisy, x_train,
#                     epochs=5,
#                     batch_size=128,
#                     shuffle=True,
#                     validation_data=(x_test_noisy, x_test),
#                     callbacks=[TensorBoard(log_dir='/tmp/tb', histogram_freq=0, write_graph=False)])
#
#     autoencoder.save('autoencoder128up.h5')
#
# train_model_128up()

'''
5 train denoising autoencoder 32-16-8
Accuracy: 3nd, Speed: 3rd
'''
# def train_model_32():
#     input_img = Input(shape=(32, 32, 3))  # mnist是28,28,1
#     # 因为mnist数据库只有黑白两种颜色，所以第三个通道为1。而正常彩色的图片第三个通道为3,。而且在导入到autoencoder之前，需要将分辨率统一
#     x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)  #根据图片尺寸修改的话，(3,3)和(2,2)都不需要修改，就是根据图片尺寸改一下16,8,1这些数就行
#     x = MaxPooling2D((2, 2), padding='same')(x)
#     x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
#     x = MaxPooling2D((2, 2), padding='same')(x)
#     x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#     encoded = MaxPooling2D((2, 2), padding='same', name='encoder')(x)
#
#     # at this point the representation is (4, 4, 8) i.e. 128-dimensional
#
#     x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
#     x = UpSampling2D((2, 2))(x)
#     x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
#     x = UpSampling2D((2, 2))(x)
#     x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
#     x = UpSampling2D((2, 2))(x)
#     decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x) #mnist是1
#
#     autoencoder = Model(input_img, decoded)
#     autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
#
#     '''
#     fit
#     '''
#     autoencoder.fit(x_train_noisy, x_train,
#                     epochs=5,
#                     batch_size=128,
#                     shuffle=True,
#                     validation_data=(x_test_noisy, x_test),
#                     callbacks=[TensorBoard(log_dir='/tmp/tb', histogram_freq=0, write_graph=False)])
#
#     autoencoder.save('autoencoder32.h5')
#
# train_model_32()

'''
6 train denoising autoencoder asym
Accuracy: 2nd, Speed: 3rd
'''
# def train_model_asym():
#     model = Sequential()
#
#     model.add(Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=(32, 32, 3)))
#     model.add(BatchNormalization())
#     model.add(Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu'))
#     model.add(Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu'))
#     model.add(BatchNormalization())
#     model.add(UpSampling2D())
#     model.add(Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu'))
#     model.add(BatchNormalization())
#     model.add(Conv2D(3, kernel_size=1, strides=1, padding='same', activation='sigmoid'))
#
#     model.compile(optimizer='adam', metrics=['accuracy'], loss='mean_squared_error')
#     autoencoder=model
#     autoencoder.summary()
#
#     autoencoder.fit(x_train_noisy, x_train,
#                     epochs=5,
#                     batch_size=128,
#                     shuffle=True,
#                     validation_data=(x_test_noisy, x_test),
#                     callbacks=[TensorBoard(log_dir='/tmp/tb', histogram_freq=0, write_graph=False)])
#
#     autoencoder.save('autoencoderasym.h5')
#
# train_model_asym()

'''
7 train denoising autoencoder 256up
Accuracy: 1st, Speed: 5th
'''
def conv_block(x, filters, kernel_size, strides=2):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
               padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

def deconv_block(x, filters, kernel_size, strides=2):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
               padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

def deconv_block(x, filters, kernel_size):
    x = Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=2,
                        padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

def train_model_256up():
    input = Input(shape=(32, 32, 3), name='input')
    conv_block1 = conv_block(input, 32, 3)
    conv_block2 = conv_block(conv_block1, 64, 3)
    conv_block3 = conv_block(conv_block2, 128, 3)
    conv_block4 = conv_block(conv_block3, 256, 3)
    conv_block5 = conv_block(conv_block4, 256, 3, 1)

    deconv_block1 = deconv_block(conv_block5, 256, 3)
    merge1 = Concatenate()([deconv_block1, conv_block3])
    deconv_block2 = deconv_block(merge1, 128, 3)
    merge2 = Concatenate()([deconv_block2, conv_block2])
    deconv_block3 = deconv_block(merge2, 64, 3)
    merge3 = Concatenate()([deconv_block3, conv_block1])
    deconv_block4 = deconv_block(merge3, 32, 3)

    final_deconv = Conv2DTranspose(filters=3, kernel_size=3, padding='same')(deconv_block4)
    output = Activation('sigmoid', name='output')(final_deconv)

    return Model(input, output, name='dae')

train_model_256up()

autoencoder = train_model_256up()
autoencoder.compile(loss='mse', optimizer='adam')

checkpoint = ModelCheckpoint('autoencoder256up.h5', verbose=1, save_weights_only=True)
autoencoder.fit(x_train_noisy, x_train, validation_data=(x_test_noisy, x_test),
                epochs=5, batch_size=128, callbacks=[checkpoint])

'''
8 denoising images
'''
# autoencoder.load_weights('autoencoder256up.h5')
# test_data_denoised = autoencoder.predict(x_test_noisy)
