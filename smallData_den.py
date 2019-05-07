from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
# from keras.utils.training_utils import multi_gpu_model
# import tensorflow as tf
print("datagen init... ")
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

# img = load_img('data_dentisy/train/missing/IMG_6204.jpg')  # this is a PIL image
img = load_img('data_dentisy/train/normal/IMG_6697.JPG')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

MAX_GEN = 140
print("datagen in MAX ",MAX_GEN)
# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(x, batch_size=10,
                          # save_to_dir='data_dentisy/train/missing/gen', save_prefix='den', save_format='jpg'):
                            save_to_dir='data_dentisy/train/normal/gen', save_prefix='den', save_format='jpg'):
    i += 1
    if i > MAX_GEN:
        break  # otherwise the generator would loop indefinitely
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
K.set_image_dim_ordering('th')

# config = tf.ConfigProto( device_count = {'CPU': 1 }, log_device_placement=True ) 
# sess = tf.Session(config=config) 
# K.set_session(sess)

# G = 1

# we'll store a copy of the model on *every* GPU and then combine
# the results from the gradient updates on the CPU
# with tf.device("/gpu:0"):
    # initialize the model
    # model = MiniGoogLeNet.build(width=32, height=32, depth=3,
    #     classes=10)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(3, 150, 150)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
# model.add(ZeroPadding2D((1, 1), input_shape=(img_rows, img_cols, channel)))

# the model so far outputs 3D feature maps (height, width, features)
model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# make the model parallel
# model = multi_gpu_model(model, gpus=G)

batch_size = 16

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        'data_dentisy/train/',  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        'data_dentisy/validation',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')

model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=5,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)
model.save_weights('first_try_den.h5')  # always save your weights after training or during training
