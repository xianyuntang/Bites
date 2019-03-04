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
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import regularizers, initializers
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.utils import multi_gpu_model
import tensorflow as tf
import os
import platform
import random


def tfdata_generator(filename, batch_size, aug=False):
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
        if aug is True:
            operation = []
            for i in range(6):
                operation.append(random.randint(0, 1))
            print(operation[0] is True)
            if operation[0]:
                image = tf.image.random_brightness(image, max_delta=0.2)
            if operation[1]:
                image = tf.image.random_crop(image, [200, 200, 3])
                image = tf.image.resize_images(image, (299, 299), method=random.randint(0, 3))
            if operation[2]:
                image = tf.image.random_contrast(image, 0.5, 1.5)
            if operation[3]:
                image = tf.contrib.image.rotate(image, 30)
            if operation[4]:
                image = tf.image.random_hue(image, max_delta=0.05)
            if operation[5]:
                image = tf.image.random_saturation(image, 0, 2)
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)
        return image, label

    dataset = tf.data.TFRecordDataset(filenames=[filename])
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=40)
    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(7316))
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    return dataset


def conv2d_bn(x, n_filter, n_row, n_col, padding='same', stride=(1, 1), use_bias=False):
    x = Convolution2D(n_filter, (n_row, n_col), strides=stride, padding=padding, use_bias=use_bias,
                      kernel_regularizer=regularizers.l2(0.00004),
                      kernel_initializer=initializers.TruncatedNormal(stddev=0.1))(x)
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
    net = Dropout(0.2)(net)
    net = Flatten()(net)
    outputs = Dense(units=7, activation='softmax')(net)

    model = Model(inputs, outputs, name='inception_v4_snake')

    return model


def get_next_batch(dataset_iterator):
    image, label = dataset_iterator.get_next()
    return image.numpy(), label.numpy()


def main(unused_argv):
    if platform.system() == 'Windows':
        print('Running on Windows')
        base_dir = os.path.join('E:\\', 'Program', 'Bite')
        save_path = os.path.join(base_dir, 'ckpt',
                                 'weights-improvement-{epoch:02d}-{loss:.4f}-{acc:.2f}.hdf5')


    elif platform.system() == 'Linux':
        print('Running on Linux')
        base_dir = os.path.join('/media', 'md0', 'xt1800i', 'Bite')
        save_path = os.path.join(base_dir, 'ckpt',
                                 'weights-improvement-{epoch:02d}-{loss:.4f}-{acc:.2f}.hdf5')

    else:
        print('Running on unsupported system')
        return


    tfrecord = os.path.join(base_dir, 'datasets', 'tfrecord', 'snake_all.tfrecord')
    with tf.device('/cpu:0'):
        model = keras_model()
        training_set = tfdata_generator(filename=tfrecord, batch_size=FLAGS.batch_size, aug=True)
        validation_set = tfdata_generator(filename=tfrecord, batch_size=FLAGS.batch_size)
    if FLAGS.ckpt is not None:
        print('loading weights')
        model.load_weights(FLAGS.ckpt)
    try:
        parallel_model = multi_gpu_model(model, gpus=2)
        print('Training on multiple GPUs')
    except:
        parallel_model = model
        print('Training on single GPU')
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=0.001, global_step=global_step,
                                               staircase=True, decay_steps=int(18288 / FLAGS.batch_size), decay_rate=0.96)
    # learning_rate = tf.keras.optimizers.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    print(model.summary())
    parallel_model.compile(optimizer=optimizer, loss='categorical_crossentropy'
                           , metrics=['accuracy'])

    ckpt = ModelCheckpoint(filepath=save_path, monitor='val_loss', save_best_only=False,
                           save_weights_only=True, mode='auto', period=10)
    callbacks_list = [ckpt]

    parallel_model.fit(x=training_set.make_one_shot_iterator(),
                       steps_per_epoch=int(18288 / FLAGS.batch_size),
                       batch_size=FLAGS.batch_size,
                       epochs=500000,
                       validation_data=validation_set.make_one_shot_iterator(),
                       validation_steps=int(18288 / FLAGS.batch_size),
                       callbacks=callbacks_list)


if __name__ == '__main__':
    tf.enable_eager_execution()
    FLAGS = tf.flags.FLAGS
    tf.flags.DEFINE_string('ckpt', None, 'filename for checkpoint')
    tf.flags.DEFINE_integer('batch_size', 32, 'batch_size')

    tf.app.run()
