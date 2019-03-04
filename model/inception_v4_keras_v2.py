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
from tensorflow.python.keras.models import Model
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
    dataset = dataset.shuffle(FLAGS.num_image)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    return dataset


def conv2d_bn(x, n_filter, n_row, n_col, padding='same', stride=(1, 1), use_bias=False):
    x = Convolution2D(n_filter, (n_row, n_col), strides=stride, padding=padding, use_bias=use_bias,
                      kernel_regularizer=regularizers.l2(0.0004),
                      kernel_initializer=initializers.VarianceScaling(scale=1, mode='fan_in',
                                                                      distribution='normal',
                                                                       seed=None)
                                                                      )(x)
    x = BatchNormalization()(x)
    x = tf.nn.elu(x)
    return x

# TODO add batch_normalization
def concatenate_bn(x):
    x = concatenate(x)
    x = BatchNormalization()(x)
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

    x = concatenate_bn([branch_1, branch_2, branch_3, branch_4])
    return x


def reduction_block_a(inputs):
    branch_1 = conv2d_bn(inputs, 192, 1, 1)
    branch_1 = conv2d_bn(branch_1, 224, 3, 3)
    branch_1 = conv2d_bn(branch_1, 256, 3, 3, stride=(2, 2), padding='valid')

    branch_2 = conv2d_bn(inputs, 384, 3, 3, stride=(2, 2), padding='valid')

    branch_3 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(inputs)

    x = concatenate_bn([branch_1, branch_2, branch_3])
    return x


def inception_block_b(inputs):
    branch_1 = conv2d_bn(inputs, 192, 1, 1)
    branch_1 = conv2d_bn(branch_1, 192, 7, 1)
    branch_1 = conv2d_bn(branch_1, 224, 1, 7)
    branch_1 = conv2d_bn(branch_1, 224, 7, 1)
    branch_1 = conv2d_bn(branch_1, 256, 1, 7)

    branch_2 = conv2d_bn(inputs, 192, 1, 1)
    branch_2 = conv2d_bn(branch_2, 224, 1, 7)
    branch_2 = conv2d_bn(branch_2, 256, 1, 7)

    branch_3 = conv2d_bn(inputs, 384, 1, 1)

    branch_4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(inputs)

    branch_4 = conv2d_bn(branch_4, 128, 1, 1)

    x = concatenate_bn([branch_1, branch_2, branch_3, branch_4])
    return x


def reduction_block_b(inputs):
    branch_1 = conv2d_bn(inputs, 256, 1, 1)
    branch_1 = conv2d_bn(branch_1, 256, 1, 7)
    branch_1 = conv2d_bn(branch_1, 320, 7, 1)
    branch_1 = conv2d_bn(branch_1, 320, 3, 3, stride=(2, 2), padding='valid')

    branch_2 = conv2d_bn(inputs, 192, 1, 1)
    branch_2 = conv2d_bn(branch_2, 192, 3, 3, stride=(2, 2), padding='valid')

    branch_3 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(inputs)

    x = concatenate_bn([branch_1, branch_2, branch_3])
    return x


def inception_block_c(inputs):
    branch_1 = conv2d_bn(inputs, 384, 1, 1)
    branch_1 = conv2d_bn(branch_1, 448, 3, 1)
    branch_1 = conv2d_bn(branch_1, 512, 1, 3)
    branch_1_1 = conv2d_bn(branch_1, 256, 1, 3)
    branch_1_2 = conv2d_bn(branch_1, 256, 3, 1)
    branch_1 = concatenate_bn([branch_1_1, branch_1_2])

    branch_2 = conv2d_bn(inputs, 384, 1, 1)
    branch_2_1 = conv2d_bn(branch_2, 256, 3, 1)
    branch_2_2 = conv2d_bn(branch_2, 256, 1, 3)
    branch_2 = concatenate_bn([branch_2_1, branch_2_2])

    branch_3 = conv2d_bn(inputs, 256, 1, 1)

    branch_4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
    branch_4 = conv2d_bn(branch_4, 256, 1, 1)

    x = concatenate_bn([branch_1, branch_2, branch_3, branch_4])
    return x


def inception_stem(inputs):
    net = conv2d_bn(inputs, 32, 3, 3, stride=(2, 2), padding='valid')
    net = conv2d_bn(net, 32, 3, 3, padding='valid')
    net = conv2d_bn(net, 64, 3, 3)

    branch_1 = conv2d_bn(net, 96, 3, 3, stride=(2, 2), padding='valid')
    branch_2 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(net)

    net = concatenate_bn([branch_1, branch_2])

    branch_1 = conv2d_bn(net, 64, 1, 1)
    branch_1 = conv2d_bn(branch_1, 64, 1, 7)
    branch_1 = conv2d_bn(branch_1, 64, 7, 1)
    branch_1 = conv2d_bn(branch_1, 96, 3, 3, padding='valid')

    branch_2 = conv2d_bn(net, 64, 1, 1)
    branch_2 = conv2d_bn(branch_2, 96, 3, 3, padding='valid')

    net = concatenate_bn([branch_1, branch_2])

    branch_1 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(net)
    branch_2 = conv2d_bn(net, 192, 3, 3, stride=(2, 2), padding='valid')

    net = concatenate_bn([branch_1, branch_2])

    return net


def keras_model(inputs):
    inputs = Input(tensor=inputs)
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
    return outputs

# TODO write code here
def main(unused_argv):
    beta = 0.01
    if platform.system() == 'Windows':
        print('Running on Windows')
        base_dir = os.path.join('E:\\', 'Program', 'Bite')
    elif platform.system() == 'Linux':
        print('Running on Linux')
        base_dir = os.path.join('/media', 'md0', 'xt1800i', 'Bite')
    else:
        print('Running on unsupported system')
        return

    tfrecord = os.path.join(base_dir, 'datasets', 'tfrecord', f'{FLAGS.training_file}.tfrecord')
    ckpt_dir = os.path.join(base_dir, 'ckpt')

    training_set = tfdata_generator(filename=tfrecord, batch_size=FLAGS.batch_size, aug=True).make_one_shot_iterator()
    validation_set = tfdata_generator(filename=tfrecord, batch_size=FLAGS.batch_size).make_one_shot_iterator()

    x_train = tf.placeholder(dtype=tf.float32, shape=[None, 299, 299, 3])
    y_label = tf.placeholder(dtype=tf.float32, shape=[None, 7])
    outputs = keras_model(x_train)
    with tf.name_scope('loss'):
        cross_entropy = -y_label * tf.log(tf.add(outputs, 1e-7))
        regularize = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables() if 'conv2d' in var.name])
        loss = tf.reduce_mean(cross_entropy+beta*regularize)
    tf.summary.scalar('loss', loss)
    with tf.name_scope('accuracy'):
        correct_pred = tf.equal(tf.argmax(outputs, 1), tf.argmax(y_label, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=0.045, global_step=global_step,
                                               staircase=True, decay_steps=int(FLAGS.num_image / FLAGS.batch_size),
                                               decay_rate=0.96)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,momentum=0.5)
    train_op = optimizer.minimize(loss=loss, global_step=global_step)
    variables = tf.trainable_variables()
    #gradients = optimizer.compute_gradients(loss, variables)

    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        merge = tf.summary.merge_all()
        writer = tf.summary.FileWriter(os.path.join(base_dir, 'logs'), sess.graph)
        if FLAGS.ckpt is not None:
            print("restore ckpt . . .")
            saver.restore(sess, os.path.join(ckpt_dir, FLAGS.ckpt))
        else:
            print("new trainer . . .")
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
        next_element = training_set.get_next()
        import numpy as np
        while True:
            batch_image, batch_label = sess.run(next_element)
            _, l, i, o, a, v = sess.run(
                [train_op, loss, global_step, outputs, accuracy, variables],
                feed_dict={x_train: batch_image, y_label: batch_label})
            print(f'iterator= {i}, Loss = {l}, Acc ={a}')
            #print(v[0][0][0][0][0:3])
            #print(g[0][0][0][0][0:3])
            print(np.argmax(batch_label, 1))
            print(np.argmax(o, 1))
            if i % 50 == 0:
                rs = sess.run(merge,
                              feed_dict={x_train: batch_image, y_label: batch_label})
                writer.add_summary(rs, i)
            if i % 500 == 0:
                saver.save(sess, os.path.join(ckpt_dir, f'model-{i}.ckpt'))


if __name__ == '__main__':
    FLAGS = tf.flags.FLAGS
    tf.flags.DEFINE_string('ckpt', None, 'name of ckpt')
    tf.flags.DEFINE_integer('batch_size', 32, 'batch_size')
    tf.flags.DEFINE_integer('num_image', 7316, 'number of image')
    tf.flags.DEFINE_string('training_file', 'snake', 'name of training file')
    tf.app.run()
