import cv2
import os
import numpy as np
import tensorflow as tf
import json


def main():
    # os.chdir(os.path.join('D:\\', 'Program', 'Bite'))
    total_image = 0
    acutus = append_image('acutus')
    y_label = append_label(0, acutus.shape[0])
    total_image += acutus.shape[0]
    print(acutus.shape[0])
    mucrosquamatus = append_image('mucrosquamatus')
    x_train = np.append(acutus, mucrosquamatus, axis=0)
    y_label = np.append(y_label, append_label(1, mucrosquamatus.shape[0]))
    total_image += mucrosquamatus.shape[0]
    print(mucrosquamatus.shape[0])
    del mucrosquamatus
    del acutus
    multinctus = append_image('multinctus')
    x_train = np.append(x_train, multinctus, axis=0)
    y_label = np.append(y_label, append_label(2, multinctus.shape[0]))
    total_image += multinctus.shape[0]
    print(multinctus.shape[0])
    del multinctus
    naja = append_image('naja')
    x_train = np.append(x_train, naja, axis=0)
    y_label = np.append(y_label, append_label(3, naja.shape[0]))
    total_image += naja.shape[0]
    print(naja.shape[0])
    del naja
    russelii = append_image('russelii')
    x_train = np.append(x_train, russelii, axis=0)
    y_label = np.append(y_label, append_label(4, russelii.shape[0]))
    total_image += russelii.shape[0]
    print(russelii.shape[0])
    del russelii
    schmidt = append_image('schmidt')
    x_train = np.append(x_train, schmidt, axis=0)
    y_label = np.append(y_label, append_label(5, schmidt.shape[0]))
    total_image += schmidt.shape[0]
    print(schmidt.shape[0])
    del schmidt
    nonvenomous = append_image('nonvenomous')
    x_train = np.append(x_train, nonvenomous, axis=0)
    y_label = np.append(y_label, append_label(6, nonvenomous.shape[0]))
    total_image += nonvenomous.shape[0]
    print(nonvenomous.shape[0])
    del nonvenomous
    print(x_train.shape)
    print(y_label.shape)
    conver2tfrecord(y_label, x_train, x_train.shape[0])


# def json_parser(snake):
#     string = open(os.path.join('/home', 'xt1800i', 'Bite', 'origindata', snake + '.json'), 'r').read()
#     json_data = json.loads(string)
#     filename = []
#     fileinfo = []
#     for i in range(0, len(json_data['frames'])):
#         tempname = 'D:\Program\Bite\origindata\{}\{}'.format(snake, json_data['visitedFrameNames'][i])
#         tempinfo = {}
#         x1 = json_data['frames'][str(i)][0]['x1']
#         x2 = json_data['frames'][str(i)][0]['x2']
#         y1 = json_data['frames'][str(i)][0]['y1']
#         y2 = json_data['frames'][str(i)][0]['y2']
#         id = json_data['frames'][str(i)][0]['id']
#         width = json_data['frames'][str(i)][0]['width']
#         height = json_data['frames'][str(i)][0]['height']
#         tags = json_data['frames'][str(i)][0]['tags']
#
#         tempinfo.update({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'id': id, 'width': width, 'height': height,
#                          'tags': tags})
#         fileinfo.append(tempinfo)
#         filename.append(tempname)
#
#     return filename, fileinfo
#
#
# def append_image(snake):
#     temp = []
#     filename, fileinfo = json_parser(snake)
#     for image in zip(filename, fileinfo):
#         img = cv2.imread(image[0])
#         img_height, img_width = img.shape[0], img.shape[1]
#         img_height_ratio = img_height / image[1]['height']
#         img_width_ratio = img_width / image[1]['width']
#         x1 = int(image[1]['x1'] * img_width_ratio)
#         y1 = int(image[1]['y1'] * img_height_ratio)
#         x2 = int(image[1]['x2'] * img_width_ratio)
#         y2 = int(image[1]['y2'] * img_height_ratio)
#         img = img[y1:y2, x1:x2]
#         img = cv2.resize(img, (299, 299))
#         temp.append(img)
#
#     return np.array(temp)

def append_image(species):
    temp = []
    path = os.path.join('/home', 'xt1800i', 'Bite', 'origindata', species)
    image_list = os.listdir(path)
    for img in image_list:
        img = cv2.imread(os.path.join(path, img))
        img = cv2.resize(img, (299, 299))
        temp.append(img)
    return np.array(temp)


def append_label(label, n_samples):
    temp = []
    for i in range(0, n_samples):
        temp.append(label)
    return np.array(temp)


def bytes_feature(value):
    if isinstance(value, list):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
    else:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int64_feature(value):
    if isinstance(value, list):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    else:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def conver2tfrecord(labels, images, n_samples):
    TFWriter = tf.python_io.TFRecordWriter('snake299.training.tfrecord')
    for i in range(0, n_samples):
        try:
            raw_image = images[i].tostring()
            label = int(labels[i])
            ftrs = tf.train.Features(feature={'Label': int64_feature(label), 'Raw_Image': bytes_feature(raw_image)})
            example = tf.train.Example(features=ftrs)
            TFWriter.write(example.SerializeToString())
        except IOError as e:
            print('Skip!\n')
    TFWriter.close()
    print('Completed')


if __name__ == '__main__':
    main()
