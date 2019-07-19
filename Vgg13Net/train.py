#https://www.cnblogs.com/skyfsm/p/8051705.html
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
 
# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam,SGD
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
import sys
sys.path.append('..')
from net.vgg13net import Vgg13Net
#https://stackoverflow.com/questions/45662253/can-i-run-keras-model-on-gpu
from tensorflow.python.client import device_lib
from keras import backend as K
import tensorflow as tf
# set GPU memory 
if('tensorflow' == K.backend()):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction=0.7
    sess = tf.Session(config=config)


def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-dtest", "--dataset_test", required=True,
        help="path to input dataset_test")
    ap.add_argument("-dtrain", "--dataset_train", required=True,
        help="path to input dataset_train")
    ap.add_argument("-m", "--model", required=True,
        help="path to output model")
    ap.add_argument("-p", "--plot", type=str, default="plot.png",
        help="path to output accuracy/loss plot")
    ap.add_argument("-epochs", "--epochs", required=True,
        help="number of epochs to train")
    args = vars(ap.parse_args()) 
    return args


# initialize the number of epochs to train for, initial learning rate,
# and batch size
EPOCHS = 10
INIT_LR = 1e-5
BS = 32
CLASS_NUM = 2
norm_size = 32


def load_data(path):
    print("[INFO] loading images...")
    data = []
    labels = []
    # grab the image paths and randomly shuffle them
    imagePaths = sorted(list(paths.list_images(path)))
    random.seed(42)
    random.shuffle(imagePaths)
    # loop over the input images
    for imagePath in imagePaths:
        # print("imagePath:",imagePath)
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (norm_size, norm_size))
        image = img_to_array(image)
        data.append(image)

        # extract the class label from the image path and update the
        # labels list
        # print("imagePath:",imagePath)
        basename_imagePath = os.path.basename(imagePath)
        # print("os.path.basename(imagePath):",basename_imagePath)
        label = int(imagePath.split(os.path.sep)[-2]) 
        # print("label:",label)   
        #label = int(basename_imagePath.split("_")[0])
        labels.append(label)
        # if label not in labels:
        #     labels.append(label)
        # print("labels:",labels)
    
    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)
    print("(np.array)labels:",labels)
    # convert the labels from integers to vectors
    labels = to_categorical(labels, num_classes=CLASS_NUM)
    print("to_categorical_labels:",labels)                         
    return data,labels
    

def train(aug,trainX,trainY,testX,testY,args):
    # initialize the model
    print("[INFO] compiling model...")
    model = Vgg13Net.build(width=norm_size, height=norm_size, depth=3, classes=CLASS_NUM)
    # opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS, momentum=0.9, nesterov=True)
    run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
        metrics=["accuracy"], options = run_opts)#binary cross-entropy,categorical_crossentropy

    # train the network
    print("[INFO] training network...")
    H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
        validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
        epochs=EPOCHS, verbose=1)

    # save the model to disk
    print("[INFO] serializing network...")
    model.save(args["model"])
    
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on den-normal_missing classifier")
    plt.xlabel("Epoch "+str(EPOCHS))
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(args["plot"])
    


#python train.py --dataset_train ../../traffic-sign/train --dataset_test ../../traffic-sign/test --model traffic_sign.model
if __name__=='__main__':
    
    print("list_local_devices:",device_lib.list_local_devices())
    print("tf.test.is_gpu_available():",tf.test.is_gpu_available())
    print("tf.test.gpu_device_name():",tf.test.gpu_device_name())
    print("_get_available_gpus:",K.tensorflow_backend._get_available_gpus())
    print("tf.test.is_built_with_cuda():",tf.test.is_built_with_cuda())



    args = args_parse()
    EPOCHS = int(args["epochs"])
    train_file_path = args["dataset_train"]
    test_file_path = args["dataset_test"]
    trainX,trainY = load_data(train_file_path)
    # print("trainX,trainY:",trainX,trainY)
    # print("(shape)trainX,trainY:",trainX.shape,trainY.shape)
    testX,testY = load_data(test_file_path)
    # print("testX,testY:",testX,testY)
    # construct the image generator for data augmentation
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
        height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
        horizontal_flip=True, fill_mode="nearest")
    train(aug,trainX,trainY,testX,testY,args)