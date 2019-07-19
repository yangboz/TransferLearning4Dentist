# import the necessary packages
from keras import backend as K
# Import necessary components to build LeNet
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
 
class ZfNet:
    @staticmethod
    def build(width, height, depth, classes):
       # Initialize model
        alexnet = Sequential()
        inputShape = (height, width, depth)
        # if we are using "channels last", update the input shape
        if K.image_data_format() == "channels_first":   #for tensorflow
            inputShape = (depth, height, width)
        model = Sequential()  
        model.add(Conv2D(96,(7,7),strides=(2,2),input_shape=inputShape,padding='valid',activation='relu',kernel_initializer='uniform'))  
        model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))  
        model.add(Conv2D(256,(5,5),strides=(2,2),padding='same',activation='relu',kernel_initializer='uniform'))  
        model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))  
        model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
        model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
        model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
        model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2),dim_ordering="tf"))  
        model.add(Flatten())  
        model.add(Dense(4096,activation='relu'))  
        model.add(Dropout(0.5))  
        model.add(Dense(4096,activation='relu'))  
        model.add(Dropout(0.5))  
        model.add(Dense(1000,activation='softmax'))  
        # model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])  
        # model.summary()

        # Compile the model
        #model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=[“accuracy”])

        # return the constructed network architecture
        return model