import tensorflow as tf
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


def conv2d_bn(inputs, filters, strides, padding):
    x = tf.nn.conv2d(inputs, filters, strides=strides, padding=padding)
    # print(x.shape)
    mean, var = tf.nn.moments(x, axes=[0, 1, 2])
    scale = tf.Variable(tf.ones(filters.shape[3]))
    shift = tf.Variable(tf.zeros(filters.shape[3]))
    epsilon = 0.001
    x = tf.nn.batch_normalization(x, mean, var, shift, scale, epsilon)

    return x


def inception_stem(inputs):
    weights = {
        'stem_w1': tf.Variable(tf.truncated_normal([3, 3, 3, 32], stddev=0.1)),
        'stem_w2': tf.Variable(tf.truncated_normal([3, 3, 32, 32])),
        'stem_w3': tf.Variable(tf.truncated_normal([3, 3, 32, 64])),
        'stem_w4': tf.Variable(tf.truncated_normal([3, 3, 64, 96])),
        'block1_branch1_w1': tf.Variable(tf.truncated_normal([1, 1, 160, 64])),
        'block1_branch1_w2': tf.Variable(tf.truncated_normal([7, 1, 64, 64])),
        'block1_branch1_w3': tf.Variable(tf.truncated_normal([1, 7, 64, 64])),
        'block1_branch1_w4': tf.Variable(tf.truncated_normal([3, 3, 64, 96])),
        'block1_branch2_w1': tf.Variable(tf.truncated_normal([1, 1, 160, 64])),
        'block1_branch2_w2': tf.Variable(tf.truncated_normal([3, 3, 64, 96])),
        'block2_branch1_w1': tf.Variable(tf.truncated_normal([3, 3, 192, 192]))

    }
    biases = {
        'stem_b1': tf.Variable(tf.truncated_normal(shape=[32], stddev=0.1)),
        'stem_b2': tf.Variable(tf.truncated_normal(shape=[32], stddev=0.1)),
        'stem_b3': tf.Variable(tf.truncated_normal(shape=[64], stddev=0.1)),
        'stem_b4': tf.Variable(tf.truncated_normal(shape=[96], stddev=0.1)),
        'block1_branch1_b1': tf.Variable(tf.truncated_normal(shape=[64], stddev=0.1)),
        'block1_branch1_b2': tf.Variable(tf.truncated_normal(shape=[64], stddev=0.1)),
        'block1_branch1_b3': tf.Variable(tf.truncated_normal(shape=[64], stddev=0.1)),
        'block1_branch1_b4': tf.Variable(tf.truncated_normal(shape=[96], stddev=0.1)),
        'block1_branch2_b1': tf.Variable(tf.truncated_normal(shape=[64], stddev=0.1)),
        'block1_branch2_b2': tf.Variable(tf.truncated_normal(shape=[96], stddev=0.1)),
        'block2_branch1_b1': tf.Variable(tf.truncated_normal(shape=[192], stddev=0.1))

    }
    with tf.name_scope('inception_stem'):
        with tf.name_scope('stem'):
            stem_c1 = conv2d_bn(inputs, weights['stem_w1'], strides=[1, 2, 2, 1], padding='VALID')
            stem_a1 = tf.nn.relu(stem_c1 + biases['stem_b1'])
            stem_c2 = conv2d_bn(stem_a1, weights['stem_w2'], strides=[1, 1, 1, 1], padding='VALID')
            stem_a2 = tf.nn.relu(stem_c2 + biases['stem_b2'])
            stem_c3 = conv2d_bn(stem_a2, weights['stem_w3'], strides=[1, 1, 1, 1], padding='SAME')
            stem_a3 = tf.nn.relu(stem_c3 + biases['stem_b3'])
            stem_c4 = conv2d_bn(stem_a3, weights['stem_w4'], strides=[1, 2, 2, 1], padding='VALID')
            stem_a4 = tf.nn.relu(stem_c4 + biases['stem_b4'])
            branch_1_m4 = tf.nn.max_pool(stem_a3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
        net = tf.concat(axis=3, values=[stem_a4, branch_1_m4])
        with tf.name_scope('block_1'):
            branch_1_c1 = conv2d_bn(net, weights['block1_branch1_w1'], strides=[1, 1, 1, 1], padding='SAME')
            branch_1_a1 = tf.nn.relu(branch_1_c1 + biases['block1_branch1_b1'])
            branch_1_c2 = conv2d_bn(branch_1_a1, weights['block1_branch1_w2'], strides=[1, 1, 1, 1], padding='SAME')
            branch_1_a2 = tf.nn.relu(branch_1_c2 + biases['block1_branch1_b2'])
            branch_1_c3 = conv2d_bn(branch_1_a2, weights['block1_branch1_w3'], strides=[1, 1, 1, 1], padding='SAME')
            branch_1_a3 = tf.nn.relu(branch_1_c3 + biases['block1_branch1_b3'])
            branch_1_c4 = conv2d_bn(branch_1_a3, weights['block1_branch1_w4'], strides=[1, 1, 1, 1], padding='VALID')
            branch_1_a4 = tf.nn.relu(branch_1_c4 + biases['block1_branch1_b4'])
            branch_2_c1 = conv2d_bn(net, weights['block1_branch2_w1'], strides=[1, 1, 1, 1], padding='SAME')
            branch_2_a1 = tf.nn.relu(branch_2_c1 + biases['block1_branch2_b1'])
            branch_2_c2 = conv2d_bn(branch_2_a1, weights['block1_branch2_w2'], strides=[1, 1, 1, 1], padding='VALID')
            branch_2_a2 = tf.nn.relu(branch_2_c2 + biases['block1_branch2_b2'])
        net = tf.concat(axis=3, values=[branch_1_a4, branch_2_a2])
        with tf.name_scope('block_2'):
            branch_1_m1 = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
            branch_2_c1 = conv2d_bn(net, weights['block2_branch1_w1'], strides=[1, 2, 2, 1], padding='VALID')
            branch_2_a1 = tf.nn.relu(branch_2_c1 + biases['block2_branch1_b1'])

    return tf.concat(axis=3, values=[branch_1_m1, branch_2_a1])


def inception_block_a(inputs):  # input size 35x35x384 output size 35x35x384
    weights = {
        'branch_1_w1': tf.Variable(tf.truncated_normal([1, 1, 384, 64], stddev=0.1)),
        'branch_1_w2': tf.Variable(tf.truncated_normal([3, 3, 64, 96], stddev=0.1)),
        'branch_1_w3': tf.Variable(tf.truncated_normal([3, 3, 96, 96], stddev=0.1)),
        'branch_2_w1': tf.Variable(tf.truncated_normal([1, 1, 384, 64], stddev=0.1)),
        'branch_2_w2': tf.Variable(tf.truncated_normal([3, 3, 64, 96], stddev=0.1)),
        'branch_3_w1': tf.Variable(tf.truncated_normal([1, 1, 384, 96], stddev=0.1)),
        'branch_4_w1': tf.Variable(tf.truncated_normal([1, 1, 384, 96], stddev=0.1))

    }
    biases = {
        'branch_1_b1': tf.Variable(tf.truncated_normal(shape=[64], stddev=0.1)),
        'branch_1_b2': tf.Variable(tf.truncated_normal(shape=[96], stddev=0.1)),
        'branch_1_b3': tf.Variable(tf.truncated_normal(shape=[96], stddev=0.1)),
        'branch_2_b1': tf.Variable(tf.truncated_normal(shape=[64], stddev=0.1)),
        'branch_2_b2': tf.Variable(tf.truncated_normal(shape=[96], stddev=0.1)),
        'branch_3_b1': tf.Variable(tf.truncated_normal(shape=[96], stddev=0.1)),
        'branch_4_b1': tf.Variable(tf.truncated_normal(shape=[96], stddev=0.1))

    }
    with tf.name_scope('inception_block_a'):
        branch_1_c1 = conv2d_bn(inputs, weights['branch_1_w1'], strides=[1, 1, 1, 1], padding='SAME')
        branch_1_a1 = tf.nn.relu(branch_1_c1 + biases['branch_1_b1'])
        branch_1_c2 = conv2d_bn(branch_1_a1, weights['branch_1_w2'], strides=[1, 1, 1, 1], padding='SAME')
        branch_1_a2 = tf.nn.relu(branch_1_c2 + biases['branch_1_b2'])
        branch_1_c3 = conv2d_bn(branch_1_a2, weights['branch_1_w3'], strides=[1, 1, 1, 1], padding='SAME')
        branch_1_a3 = tf.nn.relu(branch_1_c3 + biases['branch_1_b3'])
        branch_2_c1 = conv2d_bn(inputs, weights['branch_2_w1'], strides=[1, 1, 1, 1], padding='SAME')
        branch_2_a1 = tf.nn.relu(branch_2_c1 + biases['branch_2_b1'])
        branch_2_c2 = conv2d_bn(branch_2_a1, weights['branch_2_w2'], strides=[1, 1, 1, 1], padding='SAME')
        branch_2_a2 = tf.nn.relu(branch_2_c2 + biases['branch_2_b2'])
        branch_3_c1 = conv2d_bn(inputs, weights['branch_3_w1'], strides=[1, 1, 1, 1], padding='SAME')
        branch_3_a1 = tf.nn.relu(branch_3_c1 + biases['branch_3_b1'])
        branch_4_p1 = tf.nn.avg_pool(inputs, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
        branch_4_c1 = conv2d_bn(branch_4_p1, weights['branch_4_w1'], strides=[1, 1, 1, 1], padding='SAME')
        branch_4_a1 = tf.nn.relu(branch_4_c1 + biases['branch_4_b1'])
    return tf.concat(axis=3, values=[branch_1_a3, branch_2_a2, branch_3_a1, branch_4_a1])


def reduction_block_a(inputs):
    weights = {
        'branch_1_w1': tf.Variable(tf.truncated_normal([1, 1, 384, 192], stddev=0.1)),
        'branch_1_w2': tf.Variable(tf.truncated_normal([3, 3, 192, 224], stddev=0.1)),
        'branch_1_w3': tf.Variable(tf.truncated_normal([3, 3, 224, 256], stddev=0.1)),
        'branch_2_w1': tf.Variable(tf.truncated_normal([3, 3, 384, 384], stddev=0.1))

    }
    biases = {
        'branch_1_b1': tf.Variable(tf.truncated_normal(shape=[192], stddev=0.1)),
        'branch_1_b2': tf.Variable(tf.truncated_normal(shape=[224], stddev=0.1)),
        'branch_1_b3': tf.Variable(tf.truncated_normal(shape=[256], stddev=0.1)),
        'branch_2_b1': tf.Variable(tf.truncated_normal(shape=[384], stddev=0.1))

    }
    with tf.name_scope('reduction_block_a'):
        branch_1_c1 = conv2d_bn(inputs, weights['branch_1_w1'], strides=[1, 1, 1, 1], padding='SAME')
        branch_1_a1 = tf.nn.relu(branch_1_c1 + biases['branch_1_b1'])
        branch_1_c2 = conv2d_bn(branch_1_a1, weights['branch_1_w2'], strides=[1, 1, 1, 1], padding='SAME')
        branch_1_a2 = tf.nn.relu(branch_1_c2 + biases['branch_1_b2'])
        branch_1_c3 = conv2d_bn(branch_1_a2, weights['branch_1_w3'], strides=[1, 2, 2, 1], padding='VALID')
        branch_1_a3 = tf.nn.relu(branch_1_c3 + biases['branch_1_b3'])
        branch_2_c1 = conv2d_bn(inputs, weights['branch_2_w1'], strides=[1, 2, 2, 1], padding='VALID')
        branch_2_a1 = tf.nn.relu(branch_2_c1 + biases['branch_2_b1'])
        branch_3_m1 = tf.nn.max_pool(inputs, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

    return tf.concat(axis=3, values=[branch_1_a3, branch_2_a1, branch_3_m1])


def inception_block_b(inputs):  # input 17x17x1024 output 17x17x1024
    weights = {
        'branch_1_w1': tf.Variable(tf.truncated_normal([1, 1, 1024, 192], stddev=0.1)),
        'branch_1_w2': tf.Variable(tf.truncated_normal([1, 7, 192, 192], stddev=0.1)),
        'branch_1_w3': tf.Variable(tf.truncated_normal([7, 1, 192, 224], stddev=0.1)),
        'branch_1_w4': tf.Variable(tf.truncated_normal([1, 7, 224, 224], stddev=0.1)),
        'branch_1_w5': tf.Variable(tf.truncated_normal([7, 1, 224, 256], stddev=0.1)),
        'branch_2_w1': tf.Variable(tf.truncated_normal([1, 1, 1024, 192], stddev=0.1)),
        'branch_3_w1': tf.Variable(tf.truncated_normal([1, 1, 1024, 384], stddev=0.1)),
        'branch_2_w2': tf.Variable(tf.truncated_normal([1, 7, 192, 224], stddev=0.1)),
        'branch_2_w3': tf.Variable(tf.truncated_normal([1, 7, 224, 256], stddev=0.1)),
        'branch_4_w2': tf.Variable(tf.truncated_normal([1, 1, 1024, 128], stddev=0.1))

    }
    biases = {
        'branch_1_b1': tf.Variable(tf.truncated_normal(shape=[192], stddev=0.1)),
        'branch_1_b2': tf.Variable(tf.truncated_normal(shape=[192], stddev=0.1)),
        'branch_1_b3': tf.Variable(tf.truncated_normal(shape=[224], stddev=0.1)),
        'branch_1_b4': tf.Variable(tf.truncated_normal(shape=[224], stddev=0.1)),
        'branch_1_b5': tf.Variable(tf.truncated_normal(shape=[256], stddev=0.1)),
        'branch_2_b1': tf.Variable(tf.truncated_normal(shape=[192], stddev=0.1)),
        'branch_2_b2': tf.Variable(tf.truncated_normal(shape=[224], stddev=0.1)),
        'branch_2_b3': tf.Variable(tf.truncated_normal(shape=[256], stddev=0.1)),
        'branch_3_b1': tf.Variable(tf.truncated_normal(shape=[384], stddev=0.1)),
        'branch_4_b2': tf.Variable(tf.truncated_normal(shape=[128], stddev=0.1))

    }
    with tf.name_scope('inception_block_b'):
        with tf.name_scope('branch_1'):
            branch_1_c1 = conv2d_bn(inputs, weights['branch_1_w1'], strides=[1, 1, 1, 1], padding='SAME')
            branch_1_a1 = tf.nn.relu(branch_1_c1 + biases['branch_1_b1'])
            branch_1_c2 = conv2d_bn(branch_1_a1, weights['branch_1_w2'], strides=[1, 1, 1, 1], padding='SAME')
            branch_1_a2 = tf.nn.relu(branch_1_c2 + biases['branch_1_b2'])
            branch_1_c3 = conv2d_bn(branch_1_a2, weights['branch_1_w3'], strides=[1, 1, 1, 1], padding='SAME')
            branch_1_a3 = tf.nn.relu(branch_1_c3 + biases['branch_1_b3'])
            branch_1_c4 = conv2d_bn(branch_1_a3, weights['branch_1_w4'], strides=[1, 1, 1, 1], padding='SAME')
            branch_1_a4 = tf.nn.relu(branch_1_c4 + biases['branch_1_b4'])
            branch_1_c5 = conv2d_bn(branch_1_a4, weights['branch_1_w5'], strides=[1, 1, 1, 1], padding='SAME')
            branch_1_a5 = tf.nn.relu(branch_1_c5 + biases['branch_1_b5'])
        with tf.name_scope('branch_2'):
            branch_2_c1 = conv2d_bn(inputs, weights['branch_2_w1'], strides=[1, 1, 1, 1], padding='SAME')
            branch_2_a1 = tf.nn.relu(branch_2_c1 + biases['branch_2_b1'])
            branch_2_c2 = conv2d_bn(branch_2_a1, weights['branch_2_w2'], strides=[1, 1, 1, 1], padding='SAME')
            branch_2_a2 = tf.nn.relu(branch_2_c2 + biases['branch_2_b2'])
            branch_2_c3 = conv2d_bn(branch_2_a2, weights['branch_2_w3'], strides=[1, 1, 1, 1], padding='SAME')
            branch_2_a3 = tf.nn.relu(branch_2_c3 + biases['branch_2_b3'])
        with tf.name_scope('branch_3'):
            branch_3_c1 = conv2d_bn(inputs, weights['branch_3_w1'], strides=[1, 1, 1, 1], padding='SAME')
            branch_3_a1 = tf.nn.relu(branch_3_c1 + biases['branch_3_b1'])
        with tf.name_scope('branch_4'):
            branch_4_p1 = tf.nn.avg_pool(inputs, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
            branch_4_c2 = conv2d_bn(branch_4_p1, weights['branch_4_w2'], strides=[1, 1, 1, 1], padding='SAME')
            branch_4_a2 = tf.nn.relu(branch_4_c2 + biases['branch_4_b2'])
    return tf.concat(axis=3, values=[branch_1_a5, branch_2_a3, branch_3_a1, branch_4_a2])


def reduction_block_b(inputs):  # input 17x17x1024 output 8x8x1536
    weights = {
        'branch_1_w1': tf.Variable(tf.truncated_normal([1, 1, 1024, 256], stddev=0.1)),
        'branch_1_w2': tf.Variable(tf.truncated_normal([1, 7, 256, 256], stddev=0.1)),
        'branch_1_w3': tf.Variable(tf.truncated_normal([7, 1, 256, 320], stddev=0.1)),
        'branch_1_w4': tf.Variable(tf.truncated_normal([3, 3, 320, 320], stddev=0.1)),
        'branch_2_w1': tf.Variable(tf.truncated_normal([1, 1, 1024, 192], stddev=0.1)),
        'branch_2_w2': tf.Variable(tf.truncated_normal([3, 3, 192, 192], stddev=0.1))

    }
    biases = {
        'branch_1_b1': tf.Variable(tf.truncated_normal(shape=[256], stddev=0.1)),
        'branch_1_b2': tf.Variable(tf.truncated_normal(shape=[256], stddev=0.1)),
        'branch_1_b3': tf.Variable(tf.truncated_normal(shape=[320], stddev=0.1)),
        'branch_1_b4': tf.Variable(tf.truncated_normal(shape=[320], stddev=0.1)),
        'branch_2_b1': tf.Variable(tf.truncated_normal(shape=[192], stddev=0.1)),
        'branch_2_b2': tf.Variable(tf.truncated_normal(shape=[192], stddev=0.1))

    }
    with tf.name_scope('inception_block_c'):
        with tf.name_scope('branch_1'):
            branch_1_c1 = conv2d_bn(inputs, weights['branch_1_w1'], strides=[1, 1, 1, 1], padding='SAME')
            branch_1_a1 = tf.nn.relu(branch_1_c1 + biases['branch_1_b1'])
            branch_1_c2 = conv2d_bn(branch_1_a1, weights['branch_1_w2'], strides=[1, 1, 1, 1], padding='SAME')
            branch_1_a2 = tf.nn.relu(branch_1_c2 + biases['branch_1_b2'])
            branch_1_c3 = conv2d_bn(branch_1_a2, weights['branch_1_w3'], strides=[1, 1, 1, 1], padding='SAME')
            branch_1_a3 = tf.nn.relu(branch_1_c3 + biases['branch_1_b3'])
            branch_1_c4 = conv2d_bn(branch_1_a3, weights['branch_1_w4'], strides=[1, 2, 2, 1], padding='VALID')
            branch_1_a4 = tf.nn.relu(branch_1_c4 + biases['branch_1_b4'])
        with tf.name_scope('branch_2'):
            branch_2_c1 = conv2d_bn(inputs, weights['branch_2_w1'], strides=[1, 1, 1, 1], padding='SAME')
            branch_2_a1 = tf.nn.relu(branch_2_c1 + biases['branch_2_b1'])
            branch_2_c2 = conv2d_bn(branch_2_a1, weights['branch_2_w2'], strides=[1, 2, 2, 1], padding='VALID')
            branch_2_a2 = tf.nn.relu(branch_2_c2 + biases['branch_2_b2'])
        with tf.name_scope('branch_3'):
            branch_3_m1 = tf.nn.max_pool(inputs, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
        return tf.concat(axis=3, values=[branch_3_m1, branch_2_a2, branch_1_a4])


def inception_block_c(inputs):
    weights = {
        'branch_1_w1': tf.Variable(tf.truncated_normal([1, 1, 1536, 384], stddev=0.1)),
        'branch_1_w2': tf.Variable(tf.truncated_normal([1, 3, 384, 448], stddev=0.1)),
        'branch_1_w3': tf.Variable(tf.truncated_normal([3, 1, 448, 512], stddev=0.1)),
        'branch_1_1_w1': tf.Variable(tf.truncated_normal([1, 3, 512, 256], stddev=0.1)),
        'branch_1_2_w1': tf.Variable(tf.truncated_normal([3, 1, 512, 256], stddev=0.1)),
        'branch_2_w1': tf.Variable(tf.truncated_normal([1, 1, 1536, 384], stddev=0.1)),
        'branch_2_1_w1': tf.Variable(tf.truncated_normal([3, 1, 384, 256], stddev=0.1)),
        'branch_2_2_w1': tf.Variable(tf.truncated_normal([1, 3, 384, 256], stddev=0.1)),
        'branch_3_w1': tf.Variable(tf.truncated_normal([1, 1, 1536, 256], stddev=0.1)),
        'branch_4_w1': tf.Variable(tf.truncated_normal([1, 1, 1536, 256], stddev=0.1))

    }
    biases = {
        'branch_1_b1': tf.Variable(tf.truncated_normal(shape=[384], stddev=0.1)),
        'branch_1_b2': tf.Variable(tf.truncated_normal(shape=[448], stddev=0.1)),
        'branch_1_b3': tf.Variable(tf.truncated_normal(shape=[512], stddev=0.1)),
        'branch_1_1_b1': tf.Variable(tf.truncated_normal(shape=[256], stddev=0.1)),
        'branch_1_2_b1': tf.Variable(tf.truncated_normal(shape=[256], stddev=0.1)),
        'branch_2_b1': tf.Variable(tf.truncated_normal(shape=[384], stddev=0.1)),
        'branch_2_1_b1': tf.Variable(tf.truncated_normal(shape=[256], stddev=0.1)),
        'branch_2_2_b1': tf.Variable(tf.truncated_normal(shape=[256], stddev=0.1)),
        'branch_3_b1': tf.Variable(tf.truncated_normal(shape=[256], stddev=0.1)),
        'branch_4_b1': tf.Variable(tf.truncated_normal(shape=[256], stddev=0.1))

    }
    with tf.name_scope('inception_block_c'):
        with tf.name_scope('branch_1'):
            branch_1_c1 = conv2d_bn(inputs, weights['branch_1_w1'], strides=[1, 1, 1, 1], padding='SAME')
            branch_1_a1 = tf.nn.relu(branch_1_c1 + biases['branch_1_b1'])
            branch_1_c2 = conv2d_bn(branch_1_a1, weights['branch_1_w2'], strides=[1, 1, 1, 1], padding='SAME')
            branch_1_a2 = tf.nn.relu(branch_1_c2 + biases['branch_1_b2'])
            branch_1_c3 = conv2d_bn(branch_1_a2, weights['branch_1_w3'], strides=[1, 1, 1, 1], padding='SAME')
            branch_1_a3 = tf.nn.relu(branch_1_c3 + biases['branch_1_b3'])
            with tf.name_scope('branch_1_1'):
                branch_1_1_c1 = conv2d_bn(branch_1_a3, weights['branch_1_1_w1'], strides=[1, 1, 1, 1], padding='SAME')
                branch_1_1_a1 = tf.nn.relu(branch_1_1_c1 + biases['branch_1_1_b1'])
            with tf.name_scope('branch_1_2'):
                branch_1_2_c1 = conv2d_bn(branch_1_a3, weights['branch_1_2_w1'], strides=[1, 1, 1, 1], padding='SAME')
                branch_1_2_a1 = tf.nn.relu(branch_1_2_c1 + biases['branch_1_2_b1'])
        with tf.name_scope('branch_2'):
            branch_2_c1 = conv2d_bn(inputs, weights['branch_2_w1'], strides=[1, 1, 1, 1], padding='SAME')
            branch_2_a1 = tf.nn.relu(branch_2_c1 + biases['branch_2_b1'])
            with tf.name_scope('branch_2_1'):
                branch_2_1_c1 = conv2d_bn(branch_2_a1, weights['branch_2_1_w1'], strides=[1, 1, 1, 1], padding='SAME')
                branch_2_1_a1 = tf.nn.relu(branch_2_1_c1 + biases['branch_2_1_b1'])
            with tf.name_scope('branch_2_2'):
                branch_2_2_c1 = conv2d_bn(branch_2_a1, weights['branch_2_2_w1'], strides=[1, 1, 1, 1], padding='SAME')
                branch_2_2_a1 = tf.nn.relu(branch_2_2_c1 + biases['branch_2_2_b1'])
        with tf.name_scope('branch_3'):
            branch_3_c1 = conv2d_bn(inputs, weights['branch_3_w1'], strides=[1, 1, 1, 1], padding='SAME')
            branch_3_a1 = tf.nn.relu(branch_3_c1 + biases['branch_3_b1'])
        with tf.name_scope('branch_4'):
            branch_4_p1 = tf.nn.avg_pool(inputs, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
            branch_4_c1 = conv2d_bn(branch_4_p1, weights['branch_4_w1'], strides=[1, 1, 1, 1], padding='SAME')
            branch_4_a1 = tf.nn.relu(branch_4_c1 + biases['branch_4_b1'])

    return tf.concat(axis=3,
                     values=[branch_1_1_a1, branch_1_2_a1, branch_2_1_a1, branch_2_2_a1, branch_3_a1, branch_4_a1])


def model_function(x, y):
    net = inception_stem(x)
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
    net = tf.nn.avg_pool(net, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
    net = tf.nn.dropout(net, 0.5)
    layer_dropout_size = net.get_shape().as_list()[1] * net.get_shape().as_list()[2] * net.get_shape().as_list()[3]
    layer_flatten = tf.reshape(net,
                               [-1, layer_dropout_size])
    W = tf.Variable(tf.truncated_normal([layer_dropout_size, 7], stddev=0.1))
    b = tf.Variable(tf.truncated_normal(shape=[7], stddev=0.1))
    logits = tf.nn.softmax(tf.matmul(layer_flatten, W) + b)
    with tf.name_scope('loss'):
        cross_entropy = -tf.reduce_sum(y * tf.log(logits + 1e-10))
    tf.summary.scalar('loss', cross_entropy)
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    return cross_entropy, accuracy


def main():
    if platform.system() == 'Linux':
        base_dir = os.path.join('/media', 'md0', 'xt1800i', 'Bite')

    else:
        base_dir = os.path.join('D:\\', 'Program', 'Bite')

    filename = os.path.join(base_dir, 'datasets', 'snake299.training.tfrecord')
    print(filename)
    ckpt_dir = os.path.join(base_dir, 'ckpt')
    dataset = tfdata_generator(filename, batch_size=8)
    iterator = dataset.make_one_shot_iterator()
    x_image, y_label = iterator.get_next()

    loss, accuracy = model_function(x_image, y_label)
    train_step = tf.train.AdamOptimizer(learning_rate=0.045).minimize(loss)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        merge = tf.summary.merge_all()
        writer = tf.summary.FileWriter(os.path.join(base_dir, 'train'), sess.graph)
        if os.path.isfile(os.path.join(ckpt_dir, 'model.ckpt.index')):
            print("restore ckpt . . .")
            saver.restore(sess, os.path.join(ckpt_dir, 'model.ckpt'))
        else:
            print("new trainer . . .")
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
        epoch = 1
        while True:
            _, l, acc = sess.run([train_step, loss, accuracy])
            print('epoch= {}, Loss = {}, acc= {}'.format(epoch, l, acc))
            epoch += 1
            if epoch % 50 == 0:
                rs = sess.run(merge)
                writer.add_summary(rs, epoch)
            if epoch % 500 == 0:
                saver.save(sess,ckpt_dir)

if __name__ == '__main__':
    DEBUG = False

    main()
