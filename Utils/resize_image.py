import cv2
import glob
import os
from tqdm import tqdm
import sys
import json

path_base = os.path.join('D:\Program\Bite\datasets')


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


def resize_image_crop():
    for species in ['acutus', 'mucrosquamatus', 'multinctus', 'naja', 'nonvenomous', 'russelii', 'stejnegeri']:
        filename, fileinfo = json_parser(species)
        for path, info in zip(filename, fileinfo):
            print(path)
            img = cv2.imread(path)
            height, width, _ = img.shape
            # print(width, height)
            # print(info['width'], info['height'])
            x1 = int((width * info['x1']) / info['width'])
            x2 = int((width * info['x2']) / info['width'])
            y1 = int((height * info['y1']) / info['height'])
            y2 = int((height * info['y2']) / info['height'])
            # img = cv2.resize(img, (info['width'], info['height']))
            img = img[y1:y2, x1:x2]
            print(img.shape)
            img = cv2.resize(img, (299, 299), cv2.INTER_AREA)
            # cv2.imshow('test', img)
            # cv2.waitKey(0)

            cv2.imwrite(os.path.join(path_base, 'resized2', species, os.path.basename(path)), img)


def resize_image():
    for species in ['notsnake']:
        print(species)
        image_list = glob.glob(os.path.join('D:\Program\Bite\datasets\\raw_all\\', species, '*'))
        # print(image_list)
        for image_path in tqdm(image_list):
            print(image_path)
            img = cv2.imread(image_path)
            img = cv2.resize(img, (299, 299), cv2.INTER_AREA)
            new_path = os.path.join('D:\Program\Bite\datasets\\resized', species, os.path.basename(image_path))
            cv2.imwrite(new_path, img)


if __name__ == '__main__':
    resize_image()
