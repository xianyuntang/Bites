from model.net.inception_v4_keras import create_model
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input

weights = 'D:\\Program\\Bite\\model\\ckpt\\inception-v4_weights_tf_dim_ordering_tf_kernels_notop.h5'

inception_v4 = create_model(num_classes=7,include_top=True)

print(inception_v4.summary())
