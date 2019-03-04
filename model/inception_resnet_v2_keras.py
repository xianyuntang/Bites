from tensorflow.python.keras.applications import inception_resnet_v2
from tensorflow.python import keras
from tensorflow.python.keras.utils import multi_gpu_model
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.regularizers import L1L2
import tensorflow as tf
import os
import random
import platform

class ParallelModelCheckpoint(ModelCheckpoint):
    def __init__(self,
                 model,
                 filepath,
                 monitor='val_loss',
                 verbose=0,
                 save_best_only=False,
                 save_weights_only=False,
                 mode='auto',
                 period=1):
        self.single_model = model
        super(ParallelModelCheckpoint, self).__init__(filepath, monitor, verbose, save_best_only, save_weights_only,
                                                      mode, period)

    def set_model(self, model):
        super(ParallelModelCheckpoint, self).set_model(self.single_model)


def tfdata_generator(filename, batch_size, aug=False):
    def parse_tfrecord(example):
        img_features = tf.parse_single_example(
            example,
            features={'Label': tf.FixedLenFeature([], tf.int64),
                      'Raw_Image': tf.FixedLenFeature([], tf.string), })
        image = tf.decode_raw(img_features['Raw_Image'], tf.uint8)
        image = tf.reshape(image, [299, 299, 3])

        image = tf.cast(image, dtype=tf.float32)
        image = tf.divide(image, 255)
        label = tf.cast(img_features['Label'], tf.int32)
        label = tf.one_hot(label, depth=7)
        if aug is True:
            operation = []
            for i in range(6):
                operation.append(random.randint(0, 1))
            print(operation[0] is True)
            if operation[0]:
                image = tf.image.random_brightness(image, max_delta=0.1)
            if operation[1]:
                image = tf.image.random_crop(image, [250, 250, 3])
                image = tf.image.resize_images(image, (299, 299))
            if operation[2]:
                pass
                # image = tf.image.random_contrast(image, 0.5, 1.5)
            if operation[3]:
                image = tf.contrib.image.rotate(image, 30)
            if operation[4]:
                pass
                # image = tf.image.random_hue(image, max_delta=0.05)
            if operation[5]:
                pass
                # image = tf.image.random_saturation(image, 0, 2)
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)
        return image, label

    dataset = tf.data.TFRecordDataset(filenames=[filename])
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=40)
    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(7316))
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    return dataset

if platform.system() == 'Windows':
    base_dir = os.path.join('D:\\', 'Program', 'Bite')

else:
    base_dir = os.path.join('/media', 'md0', 'xt1800i', 'Bite')

tfrecord = os.path.join(base_dir, 'datasets', 'tfrecord', 'snake_all.tfrecord')
training_set = tfdata_generator(filename=tfrecord, batch_size=32, aug=True)
validation_set = tfdata_generator(filename=tfrecord, batch_size=32)

inputs = inception_resnet_v2.InceptionResNetV2(include_top=False)
for layer in inputs.layers:
    layer.trainable = False
x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(inputs.output)
x = keras.layers.Dropout(0.2)(x)
# x = keras.layers.Dense(units=4096, activation='relu', name='final_dense', kernel_regularizer=L1L2(l2=0.001))(x)
# x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(7, activation='softmax', name='predictions', kernel_regularizer=L1L2(l2=0.001))(x)

model = keras.models.Model(inputs.input, outputs)

# parallel_model = multi_gpu_model(model, gpus=2)
parallel_model = model
# learning_rate = tf.keras.optimizers.
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

parallel_model.compile(optimizer=optimizer, loss='categorical_crossentropy'
                       , metrics=['accuracy'])
save_path = os.path.join(base_dir, 'ckpt',
                         'weights-epoch-{epoch:02d}-val_loss-{val_loss:.4f}-val_acc-{val_acc:.2f}.hdf5')
ckpt = ParallelModelCheckpoint(model, filepath=save_path, monitor='val_acc', save_best_only=True,
                               save_weights_only=True, mode='auto', period=5)
earlystopping = keras.callbacks.EarlyStopping('val_loss', patience=50)
callbacks_list = [ckpt]
parallel_model.fit(x=training_set.make_one_shot_iterator(),
                   batch_size=8,
                   steps_per_epoch=int(27290 / 8),
                   validation_data=validation_set.make_one_shot_iterator(),
                   validation_steps=int(27290 / 8),
                   epochs=50000, callbacks=callbacks_list)
