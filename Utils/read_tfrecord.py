import tensorflow as tf
import os
import cv2

filename = "D:\Program\Bite\datasets\snake.training.tfrecord"
filename_queue = tf.train.string_input_producer([filename], shuffle=True, num_epochs=1)
reader = tf.TFRecordReader()
key, serialized_example = reader.read(filename_queue)
img_features = tf.parse_single_example(
    serialized_example,
    features={'Label': tf.FixedLenFeature([], tf.int64),
              'Raw_Image': tf.FixedLenFeature([], tf.string), })
image = tf.decode_raw(img_features['Raw_Image'], tf.uint8)
image = tf.reshape(image, [28, 28, 3])
label = tf.cast(img_features['Label'], tf.int64)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    count = 0
    try:
        while not coord.should_stop():
            image_data, label_data = sess.run([image, label])
            # image_data = image_data.reshape((640,640))
            # print(len(image_data))
            cv2.imshow('test', image_data)
            cv2.waitKey(1000)
            count += 1
        print('Done!')
    except tf.errors.OutOfRangeError:
        print('Done!')

    finally:
        coord.request_stop()
    coord.join(threads)
