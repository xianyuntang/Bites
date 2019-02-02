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


if __name__ == '__main__':
    tf.enable_eager_execution()
    if os.path.exists(os.path.join('.', 'datasets')):
        tfrecord = os.path.join('.', 'datasets', 'snake299.training.tfrecord')
        training_set = tfdata_generator(filename=tfrecord, batch_size=32)
    else:
        tfrecord = os.path.join('..', 'datasets', 'snake299.training.tfrecord')
        training_set = tfdata_generator(filename=tfrecord, batch_size=32)
    image,label = training_set.make_one_shot_iterator()