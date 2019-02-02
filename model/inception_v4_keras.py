from keras.preprocessing.image import ImageDataGenerator
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
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.utils import multi_gpu_model
from tensorflow.python.keras.optimizers import RMSprop
import tensorflow as tf
import os


def parse_tfrecord(example):
    img_features = tf.parse_single_example(
        example,
        features={'Label': tf.FixedLenFeature([], tf.int64),
                  'Raw_Image': tf.FixedLenFeature([], tf.string), })
    image = tf.decode_raw(img_features['Raw_Image'], tf.uint8)
    image = tf.reshape(image, [299, 299, 3])
    image = tf.cast(image, dtype=tf.float32)
    label = tf.cast(img_features['Label'], tf.int32)
    label = tf.one_hot(label, depth=7)
    return image, label


def tfdata_generator(filename, batch_size):
    dataset = tf.data.TFRecordDataset(filenames=[filename])
    dataset = dataset.map(parse_tfrecord)
    dataset = dataset.shuffle(7316)
    dataset = dataset.repeat()
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


def inception_block_a(input):
    branch_1 = conv2d_bn(input, 64, 1, 1)
    branch_1 = conv2d_bn(branch_1, 96, 3, 3)
    branch_1 = conv2d_bn(branch_1, 96, 3, 3)

    branch_2 = conv2d_bn(input, 64, 1, 1)
    branch_2 = conv2d_bn(branch_2, 96, 3, 3)

    branch_3 = conv2d_bn(input, 96, 1, 1)
    branch_4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
    branch_4 = conv2d_bn(branch_4, 96, 1, 1)

    x = concatenate([branch_1, branch_2, branch_3, branch_4])
    return x


def reduction_block_a(input):
    branch_1 = conv2d_bn(input, 192, 1, 1)
    branch_1 = conv2d_bn(branch_1, 224, 3, 3)
    branch_1 = conv2d_bn(branch_1, 256, 3, 3, stride=(2, 2), padding='valid')

    branch_2 = conv2d_bn(input, 384, 3, 3, stride=(2, 2), padding='valid')

    branch_3 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(input)

    x = concatenate([branch_1, branch_2, branch_3])
    return x


def inception_block_b(input):
    branch_1 = conv2d_bn(input, 192, 1, 1)
    branch_1 = conv2d_bn(branch_1, 192, 1, 7)
    branch_1 = conv2d_bn(branch_1, 224, 7, 1)
    branch_1 = conv2d_bn(branch_1, 224, 1, 7)
    branch_1 = conv2d_bn(branch_1, 256, 7, 1)

    branch_2 = conv2d_bn(input, 192, 1, 1)
    branch_2 = conv2d_bn(branch_2, 224, 1, 7)
    branch_2 = conv2d_bn(branch_2, 256, 1, 7)

    branch_3 = conv2d_bn(input, 384, 1, 1)

    branch_4 = AveragePooling2D((2, 2), strides=(1, 1), padding='same')(input)

    branch_4 = conv2d_bn(branch_4, 128, 1, 1)

    x = concatenate([branch_1, branch_2, branch_3, branch_4])
    return x


def reduction_block_b(input):
    branch_1 = conv2d_bn(input, 256, 1, 1)
    branch_1 = conv2d_bn(branch_1, 256, 1, 7)
    branch_1 = conv2d_bn(branch_1, 320, 7, 1)
    branch_1 = conv2d_bn(branch_1, 320, 3, 3, stride=(2, 2), padding='valid')

    branch_2 = conv2d_bn(input, 192, 1, 1)
    branch_2 = conv2d_bn(branch_2, 192, 3, 3, stride=(2, 2), padding='valid')

    branch_3 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(input)

    x = concatenate([branch_1, branch_2, branch_3])
    return x


def inception_block_c(input):
    branch_1 = conv2d_bn(input, 384, 1, 1)
    branch_1 = conv2d_bn(branch_1, 448, 1, 3)
    branch_1 = conv2d_bn(branch_1, 512, 3, 1)
    branch_1_1 = conv2d_bn(branch_1, 256, 1, 3)
    branch_1_2 = conv2d_bn(branch_1, 256, 3, 1)

    branch_2 = conv2d_bn(input, 384, 1, 1)
    branch_2_1 = conv2d_bn(branch_2, 256, 3, 1)
    branch_2_2 = conv2d_bn(branch_2, 256, 1, 3)

    branch_3 = conv2d_bn(input, 256, 1, 1)

    branch_4 = AveragePooling2D((2, 2), strides=(1, 1), padding='same')(input)
    branch_4 = conv2d_bn(branch_4, 256, 1, 1)

    x = concatenate([branch_1_1, branch_1_2, branch_2_1, branch_2_2, branch_3, branch_4])
    return x


def inception_stem(input):
    net = conv2d_bn(input, 32, 3, 3, stride=(2, 2), padding='valid')
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


def keras_model(num_classes=7, keep_prob=0.5, include_top=True):
    inputs = Input(shape=(299, 299, 3))
    x = inception_stem(inputs)
    for i in range(4):
        x = inception_block_a(x)
    x = reduction_block_a(x)
    for i in range(7):
        x = inception_block_b(x)
    x = reduction_block_b(x)
    for i in range(3):
        x = inception_block_c(x)
    if include_top:
        x = AveragePooling2D(8, 8, padding='valid')(x)
        x = Dropout(keep_prob)(x)
        x = Flatten()(x)
        x = Dense(units=num_classes, activation='softmax')(x)

    model = Model(inputs, x, name='inception_v4')

    return model


def train(ckpt=None, batch_size=32):
    tf.enable_eager_execution()
    if os.path.exists(os.path.join('.', 'datasets')):
        tfrecord = os.path.join('.', 'datasets', 'snake299.training.tfrecord')
        training_set = tfdata_generator(filename=tfrecord, batch_size=batch_size)
    else:
        tfrecord = os.path.join('..', 'datasets', 'snake299.training.tfrecord')
        training_set = tfdata_generator(filename=tfrecord, batch_size=batch_size)
    image,label = training_set.make_one_shot_iterator()
    model = keras_model()
    if ckpt != None:
        print('loading weights')
        model.load_weights(ckpt)
    try:
        parallel_model = multi_gpu_model(model, gpus=2, cpu_merge=False)
        print('Training using multiple GPUs')
    except:
        parallel_model = model
        print('Training using single GPU')
    optimizer = RMSprop(lr=0.045, rho=0.9, epsilon=1, decay=0.94)

    aug = ImageDataGenerator(featurewise_center=True,
                             featurewise_std_normalization=True,
                             rotation_range=180,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             brightness_range=(0.2, 0.8),
                             shear_range=0.2,
                             zoom_range=0.5,
                             channel_shift_range=0.5,
                             horizontal_flip=True)

    parallel_model.compile(optimizer=optimizer, loss='categorical_crossentropy'
                           , metrics=['accuracy'])
    if os.path.exists(os.path.join('.', 'saved_model')):
        save_path = os.path.join('.', 'saved_model', 'weights-improvement-{epoch:02d}-{loss:.4f}.hdf5')
    else:
        save_path = os.path.join('..', 'saved_model', 'weights-improvement-{epoch:02d}-{loss:.4f}.hdf5')
    ckpt = ModelCheckpoint(filepath=save_path, monitor='loss', save_best_only=True,
                           save_weights_only=True, mode='min', period=10)
    callbacks_list = [ckpt]
    parallel_model.fit_generator(aug.flow(x=image, y=label,batch_size=12), steps_per_epoch=229, epochs=30000,
                                     callbacks=callbacks_list)


if __name__ == '__main__':
    train(batch_size=8)
