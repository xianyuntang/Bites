import tensorflow_hub as hub
import cv2
import numpy as  np
import tensorflow as tf
module = hub.Module("https://tfhub.dev/google/imagenet/inception_resnet_v2/classification/1")
height, width = hub.get_expected_image_size(module)
images = cv2.imread(
    'D:\Program\Bite\predict\DSC_0135.JPG')  # A batch of images with shape [batch_size, height, width, 3].
images = cv2.resize(images, (299, 299))
images = np.expand_dims(images, 0)

outputs = module(dict(images=images), signature="image_classification",
                 as_dict=True)
logits = outputs["default"]


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    r = sess.run(logits)
    print(r)
