# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
 
class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        # 创建模型序列
        model = Sequential()
        # if we are using "channels last", update the input shape
        if K.image_data_format() == "channels_first":   #for tensorflow
            inputShape = (depth, height, width)
        #第一层卷积网络，使用96个卷积核，大小为11x11步长为4， 要求输入的图片为227x227， 3个通道，不加边，激活函数使用relu
        model.add(Conv2D(96, (11, 11), strides=(1, 1), input_shape=(28, 28, 1), padding='same', activation='relu',
                         kernel_initializer='uniform'))
        # 池化层
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        # 第二层加边使用256个5x5的卷积核，加边，激活函数为relu
        model.add(Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        #使用池化层，步长为2
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        # 第三层卷积，大小为3x3的卷积核使用384个
        model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        # 第四层卷积,同第三层
        model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        # 第五层卷积使用的卷积核为256个，其他同上
        model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))
        #model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        #model.summary()
        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model