from tensorflow.python.keras.layers import Convolution2D, \
    BatchNormalization, \
    Activation, \
    AveragePooling2D, \
    concatenate, \
    MaxPooling2D, \
    Input, \
    Dense, \
    Dropout, \
    Flatten
from tensorflow.python.keras import regularizers, initializers
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.utils import multi_gpu_model
import tensorflow as tf
import cv2
import os
import platform


def parse_tfrecord(example):
    img_features = tf.parse_single_example(
        example,
        features={'Label': tf.FixedLenFeature([], tf.int64),
                  'Raw_Image': tf.FixedLenFeature([], tf.string), })
    image = tf.decode_raw(img_features['Raw_Image'], tf.uint8)
    image = tf.reshape(image, [299, 299, 3])
    image = tf.cast(image, dtype=tf.float32)
    image = tf.divide(image, 255)
    label = tf.cast(img_features['Label'], tf.int32)
    label = tf.one_hot(label, depth=7)
    return image, label


def tfdata_generator(x, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(x)
    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(1))
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    return dataset


def conv2d_bn(x, n_filter, n_row, n_col, padding='same', stride=(1, 1), use_bias=False):
    x = Convolution2D(n_filter, (n_row, n_col), strides=stride, padding=padding, use_bias=use_bias,
                      kernel_regularizer=regularizers.l2(0.00004),
                      kernel_initializer=initializers.VarianceScaling(scale=2.0, mode='fan_in',
                                                                      distribution='normal',
                                                                      seed=None))(x)
    x = BatchNormalization(momentum=0.9997, scale=False)(x)
    x = Activation('relu')(x)
    return x


def inception_block_a(inputs):
    branch_1 = conv2d_bn(inputs, 64, 1, 1)
    branch_1 = conv2d_bn(branch_1, 96, 3, 3)
    branch_1 = conv2d_bn(branch_1, 96, 3, 3)

    branch_2 = conv2d_bn(inputs, 64, 1, 1)
    branch_2 = conv2d_bn(branch_2, 96, 3, 3)

    branch_3 = conv2d_bn(inputs, 96, 1, 1)
    branch_4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
    branch_4 = conv2d_bn(branch_4, 96, 1, 1)

    x = concatenate([branch_1, branch_2, branch_3, branch_4])
    return x


def reduction_block_a(inputs):
    branch_1 = conv2d_bn(inputs, 192, 1, 1)
    branch_1 = conv2d_bn(branch_1, 224, 3, 3)
    branch_1 = conv2d_bn(branch_1, 256, 3, 3, stride=(2, 2), padding='valid')

    branch_2 = conv2d_bn(inputs, 384, 3, 3, stride=(2, 2), padding='valid')

    branch_3 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(inputs)

    x = concatenate([branch_1, branch_2, branch_3])
    return x


def inception_block_b(inputs):
    branch_1 = conv2d_bn(inputs, 192, 1, 1)
    branch_1 = conv2d_bn(branch_1, 192, 1, 7)
    branch_1 = conv2d_bn(branch_1, 224, 7, 1)
    branch_1 = conv2d_bn(branch_1, 224, 1, 7)
    branch_1 = conv2d_bn(branch_1, 256, 7, 1)

    branch_2 = conv2d_bn(inputs, 192, 1, 1)
    branch_2 = conv2d_bn(branch_2, 224, 1, 7)
    branch_2 = conv2d_bn(branch_2, 256, 1, 7)

    branch_3 = conv2d_bn(inputs, 384, 1, 1)

    branch_4 = AveragePooling2D((2, 2), strides=(1, 1), padding='same')(inputs)

    branch_4 = conv2d_bn(branch_4, 128, 1, 1)

    x = concatenate([branch_1, branch_2, branch_3, branch_4])
    return x


def reduction_block_b(inputs):
    branch_1 = conv2d_bn(inputs, 256, 1, 1)
    branch_1 = conv2d_bn(branch_1, 256, 1, 7)
    branch_1 = conv2d_bn(branch_1, 320, 7, 1)
    branch_1 = conv2d_bn(branch_1, 320, 3, 3, stride=(2, 2), padding='valid')

    branch_2 = conv2d_bn(inputs, 192, 1, 1)
    branch_2 = conv2d_bn(branch_2, 192, 3, 3, stride=(2, 2), padding='valid')

    branch_3 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(inputs)

    x = concatenate([branch_1, branch_2, branch_3])
    return x


def inception_block_c(inputs):
    branch_1 = conv2d_bn(inputs, 384, 1, 1)
    branch_1 = conv2d_bn(branch_1, 448, 1, 3)
    branch_1 = conv2d_bn(branch_1, 512, 3, 1)
    branch_1_1 = conv2d_bn(branch_1, 256, 1, 3)
    branch_1_2 = conv2d_bn(branch_1, 256, 3, 1)

    branch_2 = conv2d_bn(inputs, 384, 1, 1)
    branch_2_1 = conv2d_bn(branch_2, 256, 3, 1)
    branch_2_2 = conv2d_bn(branch_2, 256, 1, 3)

    branch_3 = conv2d_bn(inputs, 256, 1, 1)

    branch_4 = AveragePooling2D((2, 2), strides=(1, 1), padding='same')(inputs)
    branch_4 = conv2d_bn(branch_4, 256, 1, 1)

    x = concatenate([branch_1_1, branch_1_2, branch_2_1, branch_2_2, branch_3, branch_4])
    return x


def inception_stem(inputs):
    net = conv2d_bn(inputs, 32, 3, 3, stride=(2, 2), padding='valid')
    net = conv2d_bn(net, 32, 3, 3, padding='valid')
    net = conv2d_bn(net, 64, 3, 3)

    branch_1 = conv2d_bn(net, 96, 3, 3, stride=(2, 2), padding='valid')
    branch_2 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(net)

    net = concatenate([branch_1, branch_2])

    branch_1 = conv2d_bn(net, 64, 1, 1)
    branch_1 = conv2d_bn(branch_1, 64, 7, 1)
    branch_1 = conv2d_bn(branch_1, 64, 1, 7)
    branch_1 = conv2d_bn(branch_1, 96, 3, 3, padding='valid')

    branch_2 = conv2d_bn(net, 64, 1, 1)
    branch_2 = conv2d_bn(branch_2, 96, 3, 3, padding='valid')

    net = concatenate([branch_1, branch_2])

    branch_1 = MaxPooling2D((2, 2), strides=(2, 2), padding='valid')(net)
    branch_2 = conv2d_bn(net, 192, 3, 3, stride=(2, 2), padding='valid')

    net = concatenate([branch_1, branch_2])

    return net


def keras_model():
    inputs = Input(shape=(299, 299, 3))
    net = inception_stem(inputs)
    for i in range(4):
        net = inception_block_a(net)
    net = reduction_block_a(net)
    for i in range(7):
        net = inception_block_b(net)
    net = reduction_block_b(net)
    for i in range(3):
        net = inception_block_c(net)
    net = AveragePooling2D(8, 8, padding='valid')(net)
    net = Dropout(0.5)(net)
    net = Flatten()(net)
    outputs = Dense(units=7, activation='softmax')(net)

    model = Model(inputs, outputs, name='inception_v4')
    return model


def predict():


    model = keras_model()
    print(model.summary())
    model.load_weights('weights-improvement-410-1556.7754-0.91.hdf5', by_name=True)
    data = tfdata_generator(img, 1)
    print(np.argmax(model.predict(x=img)))


if __name__ == '__main__':
    predict()
