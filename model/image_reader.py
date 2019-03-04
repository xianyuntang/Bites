import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import os
import cv2
from matplotlib import pyplot as plt
import random


def tfdata_generator(filename, batch_size, aug=False):
    def parse_tfrecord(example):
        img_features = tf.parse_single_example(
            example,
            features={'Label': tf.FixedLenFeature([], tf.int64),
                      'Raw_Image': tf.FixedLenFeature([], tf.string), })
        image = tf.decode_raw(img_features['Raw_Image'], tf.uint8)
        image = tf.reshape(image, [299, 299, 3])
        image = tf.divide(image, 255)
        label = tf.cast(img_features['Label'], tf.int32)
        label = tf.one_hot(label, depth=7)
        if aug is True:
            operation = []
            for i in range(6):
                operation.append(random.randint(0, 1))
            if operation[0]:
                image = tf.image.random_brightness(image, max_delta=0.1)
            if operation[1]:
                image = tf.image.random_crop(image, [250, 250, 3])
                image = tf.image.resize_images(image, (299, 299))
            # if operation[2]:
            #     image = tf.image.random_contrast(image, 0.5, 1.5)
            if operation[3]:
                image = tf.contrib.image.rotate(image, 30)
            # if operation[4]:
            #     image = tf.image.random_hue(image, max_delta=0.05)
            # if operation[5]:
            #     image = tf.image.random_saturation(image, 0, 2)
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)
        return image, label

    dataset = tf.data.TFRecordDataset(filenames=[filename])
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=12)
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    return dataset


def get_next_batch(dataset_iterator):
    return dataset_iterator.get_next()


if __name__ == '__main__':
    import numpy as np

    tf.enable_eager_execution()
    base_dir = os.path.join('E:\\', 'Program', 'Bite')
    filename = os.path.join(base_dir, 'datasets', 'tfrecord', 'snake_crop.tfrecord')
    dataset = tfdata_generator(filename, 9, aug=True)
    iterator = dataset.make_one_shot_iterator()
    plt.figure()
    while True:
        image, label = iterator.get_next()
        for i in range(9):
            img = cv2.cvtColor(image[i].numpy(), cv2.COLOR_BGR2RGB)
            plt.subplot(3, 3, i + 1)
            plt.title(np.argmax(label[i]))
            plt.imshow(img)
        plt.show()
