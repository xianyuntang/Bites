import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def parse_tfrecord(example):
    img_features = tf.parse_single_example(
        example,
        features={'Label': tf.FixedLenFeature([], tf.int64),
                  'Raw_Image': tf.FixedLenFeature([], tf.string), })
    image = tf.decode_raw(img_features['Raw_Image'], tf.uint8)
    image = tf.reshape(image, [299, 299, 3])
    image = tf.cast(image, dtype=tf.float32)
    image = tf.divide(image, 255)
    label = tf.cast(img_features['Label'], tf.int64)
    label = tf.one_hot(label, depth=7)
    return image, label


# non-eager mode
def tfdata_generator(filename, batch_size):
    dataset = tf.data.TFRecordDataset(filenames=[filename])
    dataset = dataset.map(parse_tfrecord)
    dataset = dataset.shuffle(7316)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    return dataset


def conv2d_bn(inputs, filters, kernel_size, strides, padding):
    x = Conv2D(inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0004),
                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                                                           mode='FAN_IN',
                                                                                           uniform=False,
                                                                                           seed=None))

    x = tf.layers.batch_normalization(x, momentum=0.9997, scale=False)
    x = tf.nn.relu(x)
    return x


def inception_stem(inputs):
    with tf.name_scope('inception_stem'):
        with tf.name_scope('stem'):
            net = conv2d_bn(inputs, filters=32, kernel_size=(3, 3), strides=2, padding='valid')
            net = conv2d_bn(net, filters=32, kernel_size=(3, 3), strides=1, padding='valid')
            net = conv2d_bn(net, filters=64, kernel_size=(3, 3), strides=1, padding='same')
            branch_1 = conv2d_bn(net, filters=96, kernel_size=(3, 3), strides=2, padding='valid')
            branch_2 = tf.layers.max_pooling2d(net, pool_size=(3, 3), strides=2, padding='valid')

        net = tf.concat(axis=3, values=[branch_1, branch_2])
        with tf.name_scope('diversion_1'):
            branch_1 = conv2d_bn(net, filters=64, kernel_size=(1, 1), strides=1, padding='same')
            branch_1 = conv2d_bn(branch_1, filters=64, kernel_size=(7, 1), strides=1, padding='same')
            branch_1 = conv2d_bn(branch_1, filters=64, kernel_size=(1, 7), strides=1, padding='same')
            branch_1 = conv2d_bn(branch_1, filters=96, kernel_size=(3, 3), strides=1, padding='valid')
            branch_2 = conv2d_bn(net, filters=64, kernel_size=(1, 1), strides=1, padding='same')
            branch_2 = conv2d_bn(branch_2, filters=96, kernel_size=(3, 3), strides=1, padding='valid')

        net = tf.concat(axis=3, values=[branch_1, branch_2])
        with tf.name_scope('division_2'):
            branch_1 = tf.layers.max_pooling2d(net, pool_size=(3, 3), strides=2, padding='valid')
            branch_2 = conv2d_bn(net, filters=192, kernel_size=(3, 3), strides=2, padding='valid')

    return tf.concat(axis=3, values=[branch_1, branch_2])


def inception_block_a(inputs):  # input size 35x35x384 output size 35x35x384
    with tf.name_scope('inception_block_a'):
        branch_1 = conv2d_bn(inputs, filters=64, kernel_size=(1, 1), strides=1, padding='same')
        branch_1 = conv2d_bn(branch_1, filters=96, kernel_size=(1, 1), strides=1, padding='same')
        branch_1 = conv2d_bn(branch_1, filters=96, kernel_size=(3, 3), strides=1, padding='same')
        branch_2 = conv2d_bn(inputs, filters=64, kernel_size=(1, 1), strides=1, padding='same')
        branch_2 = conv2d_bn(branch_2, filters=96, kernel_size=(3, 3), strides=1, padding='same')
        branch_3 = conv2d_bn(inputs, filters=96, kernel_size=(1, 1), strides=1, padding='same')
        branch_4 = tf.layers.average_pooling2d(inputs, pool_size=(3, 3), strides=1, padding='same')
        branch_4 = conv2d_bn(branch_4, filters=96, kernel_size=(1, 1), strides=1, padding='same')
    return tf.concat(axis=3, values=[branch_1, branch_2, branch_3, branch_4])


def reduction_block_a(inputs):
    with tf.name_scope('reduction_block_a'):
        branch_1 = conv2d_bn(inputs, filters=192, kernel_size=(1, 1), strides=1, padding='same')
        branch_1 = conv2d_bn(branch_1, filters=224, kernel_size=(3, 3), strides=1, padding='same')
        branch_1 = conv2d_bn(branch_1, filters=256, kernel_size=(3, 3), strides=2, padding='valid')
        branch_2 = conv2d_bn(inputs, filters=384, kernel_size=(3, 3), strides=2, padding='valid')
        branch_3 = tf.layers.max_pooling2d(inputs, pool_size=(3, 3), strides=2, padding='valid')
    return tf.concat(axis=3, values=[branch_1, branch_2, branch_3])


def inception_block_b(inputs):  # input 17x17x1024 output 17x17x1024
    with tf.name_scope('inception_block_b'):
        with tf.name_scope('branch_1'):
            branch_1 = conv2d_bn(inputs, filters=192, kernel_size=(1, 1), strides=1, padding='same')
            branch_1 = conv2d_bn(branch_1, filters=192, kernel_size=(1, 7), strides=1, padding='same')
            branch_1 = conv2d_bn(branch_1, filters=224, kernel_size=(7, 1), strides=1, padding='same')
            branch_1 = conv2d_bn(branch_1, filters=192, kernel_size=(1, 7), strides=1, padding='same')
            branch_1 = conv2d_bn(branch_1, filters=256, kernel_size=(7, 1), strides=1, padding='same')

        with tf.name_scope('branch_2'):
            branch_2 = conv2d_bn(inputs, filters=192, kernel_size=(1, 1), strides=1, padding='same')
            branch_2 = conv2d_bn(branch_2, filters=224, kernel_size=(1, 1), strides=1, padding='same')
            branch_2 = conv2d_bn(branch_2, filters=256, kernel_size=(1, 1), strides=1, padding='same')
        with tf.name_scope('branch_3'):
            branch_3 = conv2d_bn(inputs, filters=384, kernel_size=(1, 1), strides=1, padding='same')
        with tf.name_scope('branch_4'):
            branch_4 = tf.layers.average_pooling2d(inputs, pool_size=(2, 2), strides=1, padding='same')
            branch_4 = conv2d_bn(branch_4, filters=128, kernel_size=(1, 1), strides=1, padding='same')

    return tf.concat(axis=3, values=[branch_1, branch_2, branch_3, branch_4])


def reduction_block_b(inputs):  # input 17x17x1024 output 8x8x1536
    with tf.name_scope('inception_block_c'):
        with tf.name_scope('branch_1'):
            branch_1 = conv2d_bn(inputs, filters=256, kernel_size=(1, 1), strides=1, padding='same')
            branch_1 = conv2d_bn(branch_1, filters=256, kernel_size=(1, 7), strides=1, padding='same')
            branch_1 = conv2d_bn(branch_1, filters=320, kernel_size=(7, 1), strides=1, padding='same')
            branch_1 = conv2d_bn(branch_1, filters=320, kernel_size=(3, 3), strides=2, padding='valid')

        with tf.name_scope('branch_2'):
            branch_2 = conv2d_bn(inputs, filters=192, kernel_size=(1, 1), strides=1, padding='same')
            branch_2 = conv2d_bn(branch_2, filters=192, kernel_size=(3, 3), strides=2, padding='valid')
        with tf.name_scope('branch_3'):
            branch_3 = tf.layers.max_pooling2d(inputs, pool_size=(3, 3), strides=2, padding='valid')

        return tf.concat(axis=3, values=[branch_1, branch_2, branch_3])


def inception_block_c(inputs):
    with tf.name_scope('inception_block_c'):
        with tf.name_scope('branch_1'):
            branch_1 = conv2d_bn(inputs, filters=384, kernel_size=(1, 1), strides=1, padding='same')
            branch_1 = conv2d_bn(branch_1, filters=448, kernel_size=(1, 3), strides=1, padding='same')
            branch_1 = conv2d_bn(branch_1, filters=512, kernel_size=(3, 1), strides=1, padding='same')

            with tf.name_scope('branch_1_1'):
                branch_1_1 = conv2d_bn(branch_1, filters=256, kernel_size=(1, 3), strides=1, padding='same')
            with tf.name_scope('branch_1_2'):
                branch_1_2 = conv2d_bn(branch_1, filters=256, kernel_size=(3, 1), strides=1, padding='same')

        with tf.name_scope('branch_2'):
            branch_2 = conv2d_bn(inputs, filters=384, kernel_size=(1, 1), strides=1, padding='same')
            with tf.name_scope('branch_2_1'):
                branch_2_1 = conv2d_bn(branch_2, filters=256, kernel_size=(3, 1), strides=1, padding='same')

            with tf.name_scope('branch_2_2'):
                branch_2_2 = conv2d_bn(branch_2, filters=256, kernel_size=(1, 3), strides=1, padding='same')

        with tf.name_scope('branch_3'):
            branch_3 = conv2d_bn(branch_2, filters=256, kernel_size=(1, 1), strides=1, padding='same')

        with tf.name_scope('branch_4'):
            branch_4 = tf.layers.average_pooling2d(inputs, pool_size=(2, 2), strides=1, padding='same')
            branch_4 = conv2d_bn(branch_4, filters=256, kernel_size=(1, 1), strides=1, padding='same')

    return tf.concat(axis=3,
                     values=[branch_1_1, branch_1_2, branch_2_1, branch_2_2, branch_3, branch_4])


def build_model(x_train, y_label):
    net = inception_stem(x_train)
    # 4x inception-A
    for i in range(4):
        net = inception_block_a(net)
    # reduction-A
    net = reduction_block_a(net)
    # 7 x inception-B
    for i in range(7):
        net = inception_block_b(net)
    # reduction-B
    net = reduction_block_b(net)
    # 3 x inception-C
    for i in range(3):
        net = inception_block_c(net)
    net = tf.layers.average_pooling2d(net, pool_size=(2, 2), strides=1, padding='same')
    net = tf.layers.dropout(net)
    net = tf.layers.flatten(net)
    output = tf.layers.dense(net, 7, activation='softmax')
    output += 1e-10

    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return train_step, cross_entropy, accuracy


def train():
    file = '../datasets/snake299.training.tfrecord'
    dataset = tfdata_generator(filename=file, batch_size=32)
    iterator = dataset.make_one_shot_iterator()
    image, label = iterator.get_next()
    x_train = tf.placeholder(dtype=tf.float32, shape=[None, 299, 299, 3])
    y_label = tf.placeholder(dtype=tf.float32, shape=[None, 7])
    train_step, cross_entropy, accuracy = build_model(x_train, y_label)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        if os.path.isfile(os.path.join('.', 'saved_model', 'model.ckpt.index')):
            print("restore ckpt . . .")
            saver.restore(sess, os.path.join('.', 'saved_model', 'model.ckpt'))
        else:
            print("new trainer . . .")
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
        epoch = 0
        start = time.time()
        while True:
            image_batch, label_batch = sess.run([image, label])
            step, loss, acc = sess.run([train_step, cross_entropy, accuracy],
                                       feed_dict={x_train: image_batch, y_label: label_batch})
            if epoch % 10 == 0:
                print('epoch= {} Loss= {} acc ={}'.format(epoch, loss, acc))
            if (epoch + 1) % 500 == 0:
                print('saving checkpoint . . .')
                saver.save(sess, os.path.join('.', 'saved_model', 'model.ckpt'))
            epoch += 1


if __name__ == '__main__':
    DEBUG = False

    train()
