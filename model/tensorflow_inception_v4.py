import tensorflow as tf
import os
import platform
import random
import time

os.environ['CUDA_VISIBLE_DEVICES'] = "1"


# non-eager mode
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
                image = tf.image.resize_images(image, (299, 299))
            if operation[2]:
                pass
                # image = tf.image.random_contrast(image, 0.5, 1.5)
            if operation[3]:
                image = tf.contrib.image.rotate(image, 15)
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


def conv2d_bn(inputs, filters, strides, padding, use_bias=False):
    tf.add_to_collection("losses", tf.nn.l2_loss(inputs))
    x = tf.nn.conv2d(inputs, filters, strides=strides, padding=padding)
    x = batch_norm(x)
    x = tf.nn.relu(x)
    # print(x.shape)
    return x


def batch_norm(inputs, is_training=True, decay=0.999):
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False, dtype=tf.float32)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False, dtype=tf.float32)

    epsilon = 0.001
    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                                             batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs,
                                         pop_mean, pop_var, beta, scale, epsilon)


def inception_base(inputs):
    weights = {
        'net_w1': tf.Variable(tf.truncated_normal([3, 3, 3, 32], stddev=1)),
        'net_w2': tf.Variable(tf.truncated_normal([3, 3, 32, 32], stddev=1)),
        'net_w3': tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=1)),
        'branch_0_w1': tf.Variable(tf.truncated_normal([3, 3, 64, 96], stddev=1)),
        'branch_0_w2': tf.Variable(tf.truncated_normal([1, 1, 160, 64], stddev=1)),
        'branch_0_w3': tf.Variable(tf.truncated_normal([3, 3, 64, 96], stddev=1)),
        'branch_0_w4': tf.Variable(tf.truncated_normal([3, 3, 192, 192], stddev=1)),
        'branch_1_w1': tf.Variable(tf.truncated_normal([1, 1, 160, 64], stddev=1)),
        'branch_1_w2': tf.Variable(tf.truncated_normal([1, 7, 64, 64], stddev=1)),
        'branch_1_w3': tf.Variable(tf.truncated_normal([7, 1, 64, 64], stddev=1)),
        'branch_1_w4': tf.Variable(tf.truncated_normal([3, 3, 64, 96], stddev=1)),

    }

    with tf.variable_scope('inception_base', values=[inputs]):
        with tf.name_scope('stem'):
            net = conv2d_bn(inputs, weights['net_w1'], strides=[1, 2, 2, 1], padding='VALID')
            net = conv2d_bn(net, weights['net_w2'], strides=[1, 1, 1, 1], padding='VALID')
            net = conv2d_bn(net, weights['net_w3'], strides=[1, 1, 1, 1], padding='SAME')
            branch_0 = conv2d_bn(net, weights['branch_0_w1'], strides=[1, 2, 2, 1], padding='VALID')
            branch_1 = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
        net = tf.concat(axis=3, values=[branch_0, branch_1])
        with tf.variable_scope('block_1'):
            branch_0 = conv2d_bn(net, weights['branch_0_w2'], strides=[1, 1, 1, 1], padding='SAME')
            branch_0 = conv2d_bn(branch_0, weights['branch_0_w3'], strides=[1, 1, 1, 1], padding='VALID')
            branch_1 = conv2d_bn(net, weights['branch_1_w1'], strides=[1, 1, 1, 1], padding='SAME')
            branch_1 = conv2d_bn(branch_1, weights['branch_1_w2'], strides=[1, 1, 1, 1], padding='SAME')
            branch_1 = conv2d_bn(branch_1, weights['branch_1_w3'], strides=[1, 1, 1, 1], padding='SAME')
            branch_1 = conv2d_bn(branch_1, weights['branch_1_w4'], strides=[1, 1, 1, 1], padding='VALID')
        net = tf.concat(axis=3, values=[branch_0, branch_1])

        with tf.variable_scope('block_2'):
            branch_0 = conv2d_bn(net, weights['branch_0_w4'], strides=[1, 2, 2, 1], padding='VALID')
            branch_1 = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
    net = tf.concat(axis=3, values=[branch_0, branch_1])
    return batch_norm(net)


def inception_block_a(inputs):  # input size 35x35x384 output size 35x35x384
    with tf.variable_scope('inception_a', [inputs], reuse=tf.AUTO_REUSE):
        # weights = {
        #
        #     'branch_0_w1': tf.Variable(tf.truncated_normal([1, 1, 384, 96], stddev=1)),
        #     'branch_1_w1': tf.Variable(tf.truncated_normal([1, 1, 384, 64], stddev=1)),
        #     'branch_1_w2': tf.Variable(tf.truncated_normal([3, 3, 64, 96], stddev=1)),
        #     'branch_2_w1': tf.Variable(tf.truncated_normal([1, 1, 384, 64], stddev=1)),
        #     'branch_2_w2': tf.Variable(tf.truncated_normal([3, 3, 64, 96], stddev=1)),
        #     'branch_2_w3': tf.Variable(tf.truncated_normal([3, 3, 96, 96], stddev=1)),
        #     'branch_3_w1': tf.Variable(tf.truncated_normal([1, 1, 384, 96], stddev=1))
        # }
        weights = {

            'branch_0_w1': tf.get_variable(name='inception_a_branch_0_w1', shape=[1, 1, 384, 96]),
            'branch_1_w1': tf.get_variable(name='inception_a_branch_1_w1', shape=[1, 1, 384, 64]),
            'branch_1_w2': tf.get_variable(name='inception_a_branch_1_w2', shape=[3, 3, 64, 96]),
            'branch_2_w1': tf.get_variable(name='inception_a_branch_2_w1', shape=[1, 1, 384, 64]),
            'branch_2_w2': tf.get_variable(name='inception_a_branch_2_w2', shape=[3, 3, 64, 96]),
            'branch_2_w3': tf.get_variable(name='inception_a_branch_2_w3', shape=[3, 3, 96, 96]),
            'branch_3_w1': tf.get_variable(name='inception_a_branch_3_w1', shape=[1, 1, 384, 96])
        }

        with tf.name_scope('inception_block_a'):
            branch_0 = conv2d_bn(inputs, weights['branch_3_w1'], strides=[1, 1, 1, 1], padding='SAME')
            branch_1 = conv2d_bn(inputs, weights['branch_1_w1'], strides=[1, 1, 1, 1], padding='SAME')
            branch_1 = conv2d_bn(branch_1, weights['branch_1_w2'], strides=[1, 1, 1, 1], padding='SAME')
            branch_2 = conv2d_bn(inputs, weights['branch_2_w1'], strides=[1, 1, 1, 1], padding='SAME')
            branch_2_ = conv2d_bn(branch_2, weights['branch_2_w2'], strides=[1, 1, 1, 1], padding='SAME')
            branch_2 = conv2d_bn(branch_2_, weights['branch_2_w3'], strides=[1, 1, 1, 1], padding='SAME')

            branch_3 = tf.nn.avg_pool(inputs, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
            branch_3 = conv2d_bn(branch_3, weights['branch_3_w1'], strides=[1, 1, 1, 1], padding='SAME')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
    return batch_norm(net)


def reduction_block_a(inputs):
    with tf.variable_scope('block_reduction_a', [inputs], reuse=None):
        weights = {

            'branch_0_w1': tf.Variable(tf.truncated_normal([3, 3, 384, 384], stddev=1)),
            'branch_1_w1': tf.Variable(tf.truncated_normal([1, 1, 384, 192], stddev=1)),
            'branch_1_w2': tf.Variable(tf.truncated_normal([3, 3, 192, 224], stddev=1)),
            'branch_1_w3': tf.Variable(tf.truncated_normal([3, 3, 224, 256], stddev=1))

        }

        with tf.name_scope('reduction_block_a'):
            branch_0 = conv2d_bn(inputs, weights['branch_0_w1'], strides=[1, 2, 2, 1], padding='VALID')
            branch_1 = conv2d_bn(inputs, weights['branch_1_w1'], strides=[1, 1, 1, 1], padding='SAME')
            branch_1 = conv2d_bn(branch_1, weights['branch_1_w2'], strides=[1, 1, 1, 1], padding='SAME')
            branch_1 = conv2d_bn(branch_1, weights['branch_1_w3'], strides=[1, 2, 2, 1], padding='VALID')
            branch_2 = tf.nn.max_pool(inputs, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2])
    return batch_norm(net)


def inception_block_b(inputs):  # input 17x17x1024 output 17x17x1024
    with tf.variable_scope('block_inception_b', [inputs], reuse=tf.AUTO_REUSE):
        # weights = {
        #     'branch_0_w1': tf.Variable(tf.truncated_normal([1, 1, 1024, 384], stddev=1)),
        #     'branch_1_w1': tf.Variable(tf.truncated_normal([1, 1, 1024, 192], stddev=1)),
        #     'branch_1_w2': tf.Variable(tf.truncated_normal([1, 7, 192, 224], stddev=1)),
        #     'branch_1_w3': tf.Variable(tf.truncated_normal([7, 1, 224, 256], stddev=1)),
        #     'branch_2_w1': tf.Variable(tf.truncated_normal([1, 1, 1024, 192], stddev=1)),
        #     'branch_2_w2': tf.Variable(tf.truncated_normal([7, 1, 192, 192], stddev=1)),
        #     'branch_2_w3': tf.Variable(tf.truncated_normal([1, 7, 192, 224], stddev=1)),
        #     'branch_2_w4': tf.Variable(tf.truncated_normal([7, 1, 224, 224], stddev=1)),
        #     'branch_2_w5': tf.Variable(tf.truncated_normal([1, 7, 224, 256], stddev=1)),
        #     'branch_3_w1': tf.Variable(tf.truncated_normal([1, 1, 1024, 128], stddev=1))
        #
        # }
        weights = {
            'branch_0_w1': tf.get_variable(name='inception_b_branch_0_w1', shape=[1, 1, 1024, 384]),
            'branch_1_w1': tf.get_variable(name='inception_b_branch_1_w1', shape=[1, 1, 1024, 192]),
            'branch_1_w2': tf.get_variable(name='inception_b_branch_1_w2', shape=[1, 7, 192, 224]),
            'branch_1_w3': tf.get_variable(name='inception_b_branch_1_w3', shape=[7, 1, 224, 256]),
            'branch_2_w1': tf.get_variable(name='inception_b_branch_2_w1', shape=[1, 1, 1024, 192]),
            'branch_2_w2': tf.get_variable(name='inception_b_branch_2_w2', shape=[7, 1, 192, 192]),
            'branch_2_w3': tf.get_variable(name='inception_b_branch_2_w3', shape=[1, 7, 192, 224]),
            'branch_2_w4': tf.get_variable(name='inception_b_branch_2_w4', shape=[7, 1, 224, 224]),
            'branch_2_w5': tf.get_variable(name='inception_b_branch_2_w5', shape=[1, 7, 224, 256]),
            'branch_3_w1': tf.get_variable(name='inception_b_branch_3_w1', shape=[1, 1, 1024, 128]),

        }

        branch_0 = conv2d_bn(inputs, weights['branch_0_w1'], strides=[1, 1, 1, 1], padding='SAME')
        branch_1 = conv2d_bn(inputs, weights['branch_1_w1'], strides=[1, 1, 1, 1], padding='SAME')
        branch_1 = conv2d_bn(branch_1, weights['branch_1_w2'], strides=[1, 1, 1, 1], padding='SAME')
        branch_1 = conv2d_bn(branch_1, weights['branch_1_w3'], strides=[1, 1, 1, 1], padding='SAME')
        branch_2 = conv2d_bn(inputs, weights['branch_2_w1'], strides=[1, 1, 1, 1], padding='SAME')
        branch_2 = conv2d_bn(branch_2, weights['branch_2_w2'], strides=[1, 1, 1, 1], padding='SAME')
        branch_2 = conv2d_bn(branch_2, weights['branch_2_w3'], strides=[1, 1, 1, 1], padding='SAME')
        branch_2 = conv2d_bn(branch_2, weights['branch_2_w4'], strides=[1, 1, 1, 1], padding='SAME')
        branch_2 = conv2d_bn(branch_2, weights['branch_2_w5'], strides=[1, 1, 1, 1], padding='SAME')
        branch_3 = tf.nn.avg_pool(inputs, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
        branch_3 = conv2d_bn(branch_3, weights['branch_3_w1'], strides=[1, 1, 1, 1], padding='SAME')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
    return batch_norm(net)


def reduction_block_b(inputs):  # input 17x17x1024 output 8x8x1536
    with tf.variable_scope('block_reduction_block_b', [inputs], reuse=None):
        weights = {
            'branch_0_w1': tf.Variable(tf.truncated_normal([1, 1, 1024, 192], stddev=1)),
            'branch_0_w2': tf.Variable(tf.truncated_normal([3, 3, 192, 192], stddev=1)),
            'branch_1_w1': tf.Variable(tf.truncated_normal([1, 1, 1024, 256], stddev=1)),
            'branch_1_w2': tf.Variable(tf.truncated_normal([1, 7, 256, 256], stddev=1)),
            'branch_1_w3': tf.Variable(tf.truncated_normal([7, 1, 256, 320], stddev=1)),
            'branch_1_w4': tf.Variable(tf.truncated_normal([3, 3, 320, 320], stddev=1)),

        }
        # weights = {
        #     'branch_0_w1': tf.get_variable(name='inception_c_branch_0_w1', shape=[1, 1, 1024, 192]),
        #     'branch_0_w2': tf.get_variable(name='inception_c_branch_0_w2', shape=[3, 3, 192, 192]),
        #     'branch_1_w1': tf.get_variable(name='inception_c_branch_1_w1', shape=[1, 1, 1024, 256]),
        #     'branch_1_w2': tf.get_variable(name='inception_c_branch_1_w2', shape=[1, 7, 256, 256]),
        #     'branch_1_w3': tf.get_variable(name='inception_c_branch_1_w3', shape=[7, 1, 256, 320]),
        #     'branch_1_w4': tf.get_variable(name='inception_c_branch_1_w4', shape=[3, 3, 320, 320]),
        #
        # }

        with tf.name_scope('reduction_block_b'):
            branch_0 = conv2d_bn(inputs, weights['branch_0_w1'], strides=[1, 1, 1, 1], padding='SAME')
            branch_0 = conv2d_bn(branch_0, weights['branch_0_w2'], strides=[1, 2, 2, 1], padding='VALID')
            branch_1 = conv2d_bn(inputs, weights['branch_1_w1'], strides=[1, 1, 1, 1], padding='SAME')
            branch_1 = conv2d_bn(branch_1, weights['branch_1_w2'], strides=[1, 1, 1, 1], padding='SAME')
            branch_1 = conv2d_bn(branch_1, weights['branch_1_w3'], strides=[1, 1, 1, 1], padding='SAME')
            branch_1 = conv2d_bn(branch_1, weights['branch_1_w4'], strides=[1, 2, 2, 1], padding='VALID')

            branch_2 = tf.nn.max_pool(inputs, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2])
    return batch_norm(net)


def inception_block_c(inputs):
    with tf.variable_scope('block_inception_c', [inputs], reuse=tf.AUTO_REUSE):
        # weights = {
        #     'branch_0_w1': tf.Variable(tf.truncated_normal([1, 1, 1536, 256], stddev=1)),
        #     'branch_1_w1': tf.Variable(tf.truncated_normal([1, 1, 1536, 384], stddev=1)),
        #     'branch_1_1_w1': tf.Variable(tf.truncated_normal([3, 1, 384, 256], stddev=1)),
        #     'branch_1_2_w1': tf.Variable(tf.truncated_normal([1, 3, 384, 256], stddev=1)),
        #     'branch_2_w1': tf.Variable(tf.truncated_normal([1, 1, 1536, 384], stddev=1)),
        #     'branch_2_w2': tf.Variable(tf.truncated_normal([1, 3, 384, 448], stddev=1)),
        #     'branch_2_w3': tf.Variable(tf.truncated_normal([3, 1, 448, 512], stddev=1)),
        #     'branch_2_1_w1': tf.Variable(tf.truncated_normal([1, 3, 512, 256], stddev=1)),
        #     'branch_2_2_w1': tf.Variable(tf.truncated_normal([3, 1, 512, 256], stddev=1)),
        #     'branch_3_w1': tf.Variable(tf.truncated_normal([1, 1, 1536, 256], stddev=1))
        # }
        weights = {
            'branch_0_w1': tf.get_variable(name='inception_c_branch_0_w1', shape=[1, 1, 1536, 256]),
            'branch_1_w1': tf.get_variable(name='inception_c_branch_1_w1', shape=[1, 1, 1536, 384]),
            'branch_1_1_w1': tf.get_variable(name='inception_c_branch_1_1_w1', shape=[3, 1, 384, 256]),
            'branch_1_2_w1': tf.get_variable(name='inception_c_branch_1_2_w1', shape=[1, 3, 384, 256]),
            'branch_2_w1': tf.get_variable(name='inception_c_branch_2_w1', shape=[1, 1, 1536, 384]),
            'branch_2_w2': tf.get_variable(name='inception_c_branch_2_w2', shape=[1, 3, 384, 448]),
            'branch_2_w3': tf.get_variable(name='inception_c_branch_2_w3', shape=[3, 1, 448, 512]),
            'branch_2_1_w1': tf.get_variable(name='inception_c_branch_2_1_w1', shape=[1, 3, 512, 256]),
            'branch_2_2_w1': tf.get_variable(name='inception_c_branch_2_2_w1', shape=[3, 1, 512, 256]),
            'branch_3_w1': tf.get_variable(name='inception_c_branch_3_w1', shape=[1, 1, 1536, 256])
        }

        branch_0 = conv2d_bn(inputs, weights['branch_0_w1'], strides=[1, 1, 1, 1], padding='SAME')
        branch_1 = conv2d_bn(inputs, weights['branch_1_w1'], strides=[1, 1, 1, 1], padding='SAME')

        branch_1_1 = conv2d_bn(branch_1, weights['branch_1_1_w1'], strides=[1, 1, 1, 1], padding='SAME')

        branch_1_2 = conv2d_bn(branch_1, weights['branch_1_2_w1'], strides=[1, 1, 1, 1], padding='SAME')

        branch_1 = tf.concat(axis=3, values=[branch_1_1, branch_1_2])
        branch_2 = conv2d_bn(inputs, weights['branch_2_w1'], strides=[1, 1, 1, 1], padding='SAME')

        branch_2 = conv2d_bn(branch_2, weights['branch_2_w2'], strides=[1, 1, 1, 1], padding='SAME')
        branch_2 = conv2d_bn(branch_2, weights['branch_2_w3'], strides=[1, 1, 1, 1], padding='SAME')

        branch_2_1 = conv2d_bn(branch_2, weights['branch_2_1_w1'], strides=[1, 1, 1, 1], padding='SAME')

        branch_2_2 = conv2d_bn(branch_2, weights['branch_2_2_w1'], strides=[1, 1, 1, 1], padding='SAME')
        branch_2 = tf.concat(axis=3, values=[branch_2_1, branch_2_2])

        branch_3 = tf.nn.avg_pool(inputs, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
        branch_3 = conv2d_bn(branch_3, weights['branch_3_w1'], strides=[1, 1, 1, 1], padding='SAME')

        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
    return batch_norm(net)


def model_function(inputs):
    # weights = {
    #     'net_w1': tf.Variable(tf.truncated_normal([1, 1, 384, 128], stddev=1)),
    #     'net_w2': tf.Variable(tf.truncated_normal([1, 1, 128, 768], stddev=1)),
    #
    #
    #
    # }
    net = inception_base(inputs)
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
    net = tf.nn.avg_pool(net, ksize=[1, 5, 5, 1], strides=[1, 3, 3, 1], padding='SAME')
    # net = conv2d_bn(net,weights['net_w1'],strides=[1,1,1,1],padding='SAME')
    # net = conv2d_bn(net, weights['net_w2'], strides=[1, 1, 1, 1], padding='VALID')
    net = tf.nn.dropout(net, 0.8)
    layer_dropout_size = net.get_shape().as_list()[1] * net.get_shape().as_list()[2] * net.get_shape().as_list()[3]
    layer_flatten = tf.reshape(net,
                               [-1, layer_dropout_size])
    W = tf.Variable(tf.truncated_normal([layer_dropout_size, 7], stddev=1))
    tf.add_to_collection('losses', tf.nn.l2_loss(W))
    logits = tf.nn.softmax(tf.matmul(layer_flatten, W))
    return logits


def main(unused_argv):
    with tf.Graph().as_default():
        beta = 1e-5
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

        training_set = tfdata_generator(filename=tfrecord, batch_size=FLAGS.batch_size,
                                        aug=True).make_one_shot_iterator()
        validation_set = tfdata_generator(filename=tfrecord, batch_size=FLAGS.batch_size).make_one_shot_iterator()

        x_train = tf.placeholder(dtype=tf.float32, shape=[None, 299, 299, 3])
        y_label = tf.placeholder(dtype=tf.int32, shape=[None, 7])
        outputs = model_function(x_train)
        with tf.name_scope('loss'):
            cross_entropy = tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs, labels=tf.argmax(y_label, 1)))
            regularize = tf.add_n(tf.get_collection("losses"))
            # for var in tf.trainable_variables():
            #     print(var.name)
            # loss = tf.reduce_mean(cross_entropy + beta * regularize)
            loss = tf.add(cross_entropy, tf.multiply(beta, regularize))
        tf.summary.scalar('loss', loss)
        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(tf.argmax(outputs, 1), tf.argmax(y_label, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(learning_rate=0.045, global_step=global_step,
                                                   staircase=True, decay_steps=int(FLAGS.num_image / FLAGS.batch_size),
                                                   decay_rate=0.96)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss=loss, global_step=global_step)

        variables = tf.trainable_variables()
        # gradients = tf.gradients(loss,variables)
        # print(gradients)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            merge = tf.summary.merge_all()
            writer = tf.summary.FileWriter(os.path.join(base_dir, 'logs'), sess.graph)
            if FLAGS.ckpt is not None:
                print("restore ckpt . . .")
                saver.restore(sess, os.path.join(ckpt_dir, FLAGS.ckpt))
            else:
                print("new trainer . . .")
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
            train_next_element = training_set.get_next()
            val_next_element = validation_set.get_next()
            import numpy as np
            train_acc = 0
            while True:
                start = time.time()
                batch_image, batch_label = sess.run(train_next_element)
                _, i, a,l = sess.run([train_op, global_step, accuracy,loss],
                                    feed_dict={x_train: batch_image, y_label: batch_label})

                train_acc += a
                print('--------------------------------------')
                # print(L2)
                # print(v[0][0][0][0][0:3])
                # print(g[0][0][0][0][0:3])
                if i % 10 == 0:
                    print(f'time = {time.time() - start}, iterator = {i}, Loss = {l}, Acc = {train_acc/10}')
                    train_acc =0
                if i % 50 == 0:
                    rs = sess.run(merge,
                                  feed_dict={x_train: batch_image, y_label: batch_label})
                    writer.add_summary(rs, i)
                if i % 100 == 0:
                    val_batch_image, val_batch_label = sess.run(val_next_element)
                    val_loss, val_acc = sess.run(
                        [loss, accuracy],
                        feed_dict={x_train: val_batch_image, y_label: val_batch_label})
                    print(f'iterator= {i}, val_Loss = {val_loss}, val_Acc ={val_acc}')
                if i % 500 == 0:
                    saver.save(sess, os.path.join(ckpt_dir, f'model-{i}.ckpt'))


if __name__ == '__main__':
    FLAGS = tf.flags.FLAGS
    tf.flags.DEFINE_string('ckpt', None, 'name of ckpt')
    tf.flags.DEFINE_integer('batch_size', 32, 'batch_size')
    tf.flags.DEFINE_integer('num_image', 7316, 'number of image')
    tf.flags.DEFINE_string('training_file', 'snake', 'name of training file')
    tf.app.run()
