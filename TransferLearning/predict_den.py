
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam

from keras.models import model_from_json
from keras.applications.imagenet_utils import decode_predictions
# from imagenet_utils import decode_predictions

import tensorflow as tf

config = tf.ConfigProto(allow_soft_placement=True)

#最多占gpu资源的70%
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

#开始不会给tensorflow全部gpu资源 而是按需增加
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# In[2]:


# base_model=MobileNet(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

# x=base_model.output
# x=GlobalAveragePooling2D()(x)
# x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
# x=Dense(1024,activation='relu')(x) #dense layer 2
# x=Dense(512,activation='relu')(x) #dense layer 3
# preds=Dense(2,activation='softmax')(x) #final layer with softmax activation


# # In[3]:


# model=Model(inputs=base_model.input,outputs=preds)
# #specify the inputs
# #specify the outputs
# #now a model has been created based on our architecture


# # In[4]:


# for layer in model.layers[:20]:
#     layer.trainable=False
# for layer in model.layers[20:]:
#     layer.trainable=True


# # In[5]:


# train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies

# train_generator=train_datagen.flow_from_directory('data_dentisy/train/gen', # this is where you specify the path to the main data folder
#                                                  target_size=(224,224),
#                                                  color_mode='rgb',
#                                                  batch_size=32,
#                                                  class_mode='categorical',
#                                                  shuffle=True)


# # In[33]:


# model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
# # Adam optimizer
# # loss function will be categorical cross entropy
# # evaluation metric will be accuracy

# step_size_train=train_generator.n//train_generator.batch_size
# model.fit_generator(generator=train_generator,
#                    steps_per_epoch=step_size_train,
#                    epochs=10)
# https://machinelearningmastery.com/save-load-keras-deep-learning-models/

# serialize model to JSON
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("model.h5")
# print("Saved model to disk")


# later...


# load json and create model
json_file = open('model_v0527_den.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_v0527_den.h5")
print("Loaded model from disk")

img_path = 'IMG_6204.jpg'
img = image.load_img(img_path, target_size=(150, 150))
unknown = image.img_to_array(img)
# print(unknown)
unknown = np.expand_dims(unknown, axis=0)
# print(unknown)
unknown = preprocess_input(unknown)
print('Input image shape:', unknown.shape)

preds = loaded_model.predict(unknown)
print('Predicted:', preds)
# print('decode_predictions:', decode_predictions(preds))

# evaluate loaded model on test data
# score = loaded_model.evaluate(X, Y, verbose=0)
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
