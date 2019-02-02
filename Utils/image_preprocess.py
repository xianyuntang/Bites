import numpy as np
import tensorflow as tf
from skimage.transform import rotate
import os
import json
import cv2

import multiprocessing

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def json_parser(species):
    string = open(os.path.join('/home', 'xt1800i', 'Bite', 'origindata', species + '.json'), 'r').read()
    json_data = json.loads(string)
    filename = []
    fileinfo = []
    for i in range(0, len(json_data['frames'])):
        tempname = '/home/xt1800i/Bite/origindata/{}/{}'.format(species, json_data['visitedFrameNames'][i])
        tempinfo = {}
        x1 = json_data['frames'][str(i)][0]['x1']
        x2 = json_data['frames'][str(i)][0]['x2']
        y1 = json_data['frames'][str(i)][0]['y1']
        y2 = json_data['frames'][str(i)][0]['y2']
        name = json_data['visitedFrameNames'][i]
        width = json_data['frames'][str(i)][0]['width']
        height = json_data['frames'][str(i)][0]['height']
        tags = json_data['frames'][str(i)][0]['tags']

        tempinfo.update({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'name': name, 'width': width, 'height': height,
                         'tags': tags})
        fileinfo.append(tempinfo)
        filename.append(tempname)

    return filename, fileinfo

# 隨機旋轉
def random_rotate_image(image_file, image_info, species, num=20):
    image = cv2.imread(image_file)
    img_height, img_width = image.shape[0], image.shape[1]
    img_height_ratio = img_height / image_info['height']
    img_width_ratio = img_width / image_info['width']
    x1 = int(image_info['x1'] * img_width_ratio)
    y1 = int(image_info['y1'] * img_height_ratio)
    x2 = int(image_info['x2'] * img_width_ratio)
    y2 = int(image_info['y2'] * img_height_ratio)
    name = image_info['name']
    image_cut = image[y1:y2, x1:x2]
    image_rotate_list = []

    def random_rotate_image_func(image):
        angle = np.random.uniform(low=-60, high=60)
        return rotate(image, angle)

    for i in range(num):
        image_rotate = random_rotate_image_func(image_cut)
        image_bright = tf.image.random_brightness(image_rotate, max_delta=0.3)
        image_rotate_list.append(image_bright)

    for idx, img in enumerate(image_rotate_list):
        img *= 255
        img = cv2.resize(np.array(img), (299, 299))
        cv2.imwrite('/home/xt1800i/Bite/processeddata/{}/{}_rotate_{}.jpg'.format(species, name[:-4], idx), img)

# 隨機亮度
def random_brightness_image(image_file, image_info, species, num=20):
    image = cv2.imread(image_file)
    img_height, img_width = image.shape[0], image.shape[1]
    img_height_ratio = img_height / image_info['height']
    img_width_ratio = img_width / image_info['width']
    x1 = int(image_info['x1'] * img_width_ratio)
    y1 = int(image_info['y1'] * img_height_ratio)
    x2 = int(image_info['x2'] * img_width_ratio)
    y2 = int(image_info['y2'] * img_height_ratio)
    name = image_info['name']
    image_cut = image[y1:y2, x1:x2]
    image_list = []

    for i in range(num):
        image_bright = tf.image.random_brightness(image_cut, max_delta=0.3)
        image_list.append(image_bright)

    for idx, img in enumerate(image_list):
        img = np.array(img)
        img = cv2.resize(img, (299, 299))
        cv2.imwrite('/home/xt1800i/Bite/processeddata/{}/{}_brightness_{}.jpg'.format(species, name[:-4], idx), img)

# 隨機裁切
def random_crop_image(image_file, image_info, species, num=20):
    image = cv2.imread(image_file)
    img_height, img_width = image.shape[0], image.shape[1]
    img_height_ratio = img_height / image_info['height']
    img_width_ratio = img_width / image_info['width']
    x1 = int(image_info['x1'] * img_width_ratio)
    y1 = int(image_info['y1'] * img_height_ratio)
    x2 = int(image_info['x2'] * img_width_ratio)
    y2 = int(image_info['y2'] * img_height_ratio)
    name = image_info['name']
    image_cut = image[y1:y2, x1:x2]
    image_list = []

    for i in range(num):
        image_crop = tf.image.random_crop(image_cut, [int(image_cut.shape[0] * 0.8), int(image_cut.shape[1] * 0.8), 3])
        image_bright = tf.image.random_brightness(image_crop, max_delta=0.3)
        image_list.append(image_bright)

    for idx, img in enumerate(image_list):
        img = np.array(img)
        img = cv2.resize(img, (299, 299))
        cv2.imwrite('/home/xt1800i/Bite/processeddata/{}/{}_crop_{}.jpg'.format(species, name[:-4], idx), img)

# 全部隨機
def random_all(image_file, image_info, species, num=100):
    image = cv2.imread(image_file)
    img_height, img_width = image.shape[0], image.shape[1]
    img_height_ratio = img_height / image_info['height']
    img_width_ratio = img_width / image_info['width']
    x1 = int(image_info['x1'] * img_width_ratio)
    y1 = int(image_info['y1'] * img_height_ratio)
    x2 = int(image_info['x2'] * img_width_ratio)
    y2 = int(image_info['y2'] * img_height_ratio)
    name = image_info['name']
    image_cut = image[y1:y2, x1:x2]
    image_list = []

    for i in range(num):
        image_v = tf.image.random_crop(image_cut, [int(image_cut.shape[0] * 0.8), int(image_cut.shape[1] * 0.8), 3])
        image_list.append(image_v)

    for idx, img in enumerate(image_list):
        img = np.array(img)
        img = cv2.resize(img, (299, 299))
        cv2.imwrite('/home/xt1800i/Bite/processeddata/{}/{}_random_{}.jpg'.format(species, name[:-4], idx), img)


def main(species):
    filename, fileinfo = json_parser(species)
    # 產生資料夾
    try:
        os.mkdir('/home/xt1800i/Bite/processeddata/{}'.format(species))
    except:
        pass
    for file, info in zip(filename, fileinfo):
        random_brightness_image(file, info, species, num=5)
        random_rotate_image(file, info, species, num=5)
        random_crop_image(file, info, species, num=5)


if __name__ == '__main__':
    tf.enable_eager_execution()

    p1 = multiprocessing.Process(target=main, args=('acutus',))
    p2 = multiprocessing.Process(target=main, args=('mucrosquamatus',))
    p3 = multiprocessing.Process(target=main, args=('multinctus',))
    p4 = multiprocessing.Process(target=main, args=('naja',))
    p5 = multiprocessing.Process(target=main, args=('nonvenomous',))
    p6 = multiprocessing.Process(target=main, args=('russelii',))
    p7 = multiprocessing.Process(target=main, args=('schmidt',))

    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()
    p7.start()
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()
    p7.join()
