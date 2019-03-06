import cv2
import os
import json


def json_parser(snake):
    base_dir = os.path.join('..')
    string = open(os.path.join(base_dir, 'origindata', '{}.json'.format(snake)), 'r').read()
    json_data = json.loads(string)
    filename = []
    fileinfo = []
    for i in range(0, len(json_data['frames'])):
        tempname = json_data['visitedFrameNames'][i]
        tempinfo = {}
        print(tempname)
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


def draw_border(filename, fileinfo, species='naja'):
    base_dir = os.path.join('..', 'origindata')
    print(filename)
    for itemname, iteminfo in zip(filename, fileinfo):
        img = cv2.imread(os.path.join(base_dir, species, itemname))

        # 高 * 寬
        img_height, img_width = img.shape[0], img.shape[1]
        img_height_ratio = img_height / iteminfo['height']
        img_width_ratio = img_width / iteminfo['width']
        x1 = int(iteminfo['x1'] * img_width_ratio)
        y1 = int(iteminfo['y1'] * img_height_ratio)
        x2 = int(iteminfo['x2'] * img_width_ratio)
        y2 = int(iteminfo['y2'] * img_height_ratio)
        img = cv2.rectangle(img, (x1, y1), (x2, y2),
                            (0, 255, 0), 2, 2)
        cv2.imshow('img', img)
        cv2.waitKey(1000)


def main():
    filename, fileinfo = json_parser('nonvenomous')
    #draw_border(filename, fileinfo)


if __name__ == '__main__':
    main()
