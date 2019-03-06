import cv2
import os
import numpy as np
import tensorflow as tf
import platform
import argparse
import json


def main():
    path_data = os.path.join(path_base, 'resized')
    species_list = os.listdir(path_data)

    classes = 0
    image_list = []
    label_list = []
    for species in species_list:
        if os.path.isdir(os.path.join(path_data, species)):
            append_image_label(species, classes, image_list, label_list)
            print(classes, species)
            classes += 1
    image = np.array(image_list)
    label = np.array(label_list)
    image, label = shuffle(image, label)

    conver2tfrecord(label, image)


def json_parser(species):
    string = open(os.path.join(path_base, 'raw_data', species + '.json'), 'r').read()
    json_data = json.loads(string)
    filename = []
    fileinfo = []
    for i in range(0, len(json_data['frames'])):
        tempname = os.path.join(path_base, 'raw_data', species, json_data['visitedFrameNames'][i])
        tempinfo = {}
        x1 = json_data['frames'][str(i)][0]['x1']
        x2 = json_data['frames'][str(i)][0]['x2']
        y1 = json_data['frames'][str(i)][0]['y1']
        y2 = json_data['frames'][str(i)][0]['y2']
        id = json_data['frames'][str(i)][0]['id']
        width = json_data['frames'][str(i)][0]['width']
        height = json_data['frames'][str(i)][0]['height']
        tags = json_data['frames'][str(i)][0]['tags']

        tempinfo.update({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'id': id, 'width': width, 'height': height,
                         'tags': tags})
        fileinfo.append(tempinfo)
        filename.append(tempname)

    return filename, fileinfo


def append_image_label_with_crop(species, _class, image_temp, label_temp):
    filename, fileinfo = json_parser(species)
    for image in zip(filename, fileinfo):
        img = cv2.imread(image[0])
        img_height, img_width = img.shape[0], img.shape[1]
        img_height_ratio = img_height / image[1]['height']
        img_width_ratio = img_width / image[1]['width']
        x1 = int(image[1]['x1'] * img_width_ratio)
        y1 = int(image[1]['y1'] * img_height_ratio)
        x2 = int(image[1]['x2'] * img_width_ratio)
        y2 = int(image[1]['y2'] * img_height_ratio)
        img = img[y1:y2, x1:x2]
        img = cv2.resize(img, (299, 299))
        image_temp.append(img)
        label_temp.append(_class)


def shuffle(image, label):
    idx = np.random.permutation(len(image))
    image, label = image[idx], label[idx]

    return image, label


def append_image_label(species, _class, image_temp, label_temp):
    path = os.path.join(path_base, 'resized', species)
    image_list = os.listdir(path)
    for img in image_list:
        img = cv2.imread(os.path.join(path, img))
        # img = cv2.resize(img, (299, 299))
        image_temp.append(img)
        label_temp.append(_class)


def conver2tfrecord(labels, images):
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

    path_tf = os.path.join(path_base, 'tfrecord', f'{args.output}.tfrecord')
    TFWriter = tf.python_io.TFRecordWriter(path_tf)

    for i in range(0, images.shape[0]):
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


def to_one_shot(x, depth):
    zeros = np.zeros((x.shape[0], depth))
    zeros[np.arange(x.shape[0]), depth] = 1
    return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', default='snake_resized')
    args = parser.parse_args()
    if platform.system() == 'Linux':
        path_base = os.path.join('/media', 'md0', 'xt1800i', 'Bite', 'datasets')

    elif platform.system() == 'Windows':
        path_base = os.path.join('D:\\', 'Program', 'Bite', 'datasets')
    main()
