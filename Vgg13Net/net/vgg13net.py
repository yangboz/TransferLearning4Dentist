# import the necessary packages
from keras import backend as K
# Import necessary components to build LeNet
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
 
class Vgg13Net:
    @staticmethod
    def build(width, height, depth, classes):
       # Initialize model
        model = Sequential()
        img_shape=(height, width, depth)
        # Layer 1
        model.add(Conv2D(64,(3,3),strides=(1,1),input_shape=img_shape,padding='same',activation='relu',kernel_initializer='uniform'))  
        model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
        model.add(MaxPooling2D(pool_size=(2,2)))  
        model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
        model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
        model.add(MaxPooling2D(pool_size=(2,2)))  
        model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
        model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
        model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
        model.add(MaxPooling2D(pool_size=(2,2)))  
        model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
        model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
        model.add(MaxPooling2D(pool_size=(2,2)))  
        model.add(Flatten())
        model.add(Dense(4096,activation='relu'))  
        model.add(Dropout(0.5))  
        model.add(Dense(4096,activation='relu'))  
        model.add(Dropout(0.5))  
        model.add(Dense(1000,activation='softmax'))  
        # model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])  
        # model.summary()
            #model.summary()

        # Compile the model
        #model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=[“accuracy”])

        # return the constructed network architecture
        return model