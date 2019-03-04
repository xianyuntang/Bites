from tensorflow.python import keras
import os
from tensorflow.python.keras.applications import inception_resnet_v2
import cv2
import numpy as np
from tensorflow.python.keras.utils import multi_gpu_model
import platform
import h5py

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
result = {
    0: 'acutus',
    1: 'mucrosquamatus',
    2: 'multinctus',
    3: 'naja',
    4: 'nonvenomous',
    5: 'russelii',
    6: 'schmidt',
}

inputs = inception_resnet_v2.InceptionResNetV2(include_top=False, weights=None)
x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(inputs.output)
outputs = keras.layers.Dense(7, activation='softmax', name='predictions')(x)

model = keras.models.Model(inputs.input, outputs)
model.load_weights("/media/md0/xt1800i/Bite/ckpt/weights-epoch-05-val_loss-0.5329-val_acc-0.82.hdf5", by_name=True)

img = cv2.imread("/home/xt1800i/Bite/predict/6.jpg")
img = cv2.resize(img, (299, 299))
img = np.divide(img, 255)
img = np.expand_dims(img, 0)
r = model.predict(x=img)
print(r)
print(np.argmax(r))
