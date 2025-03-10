# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
from keras import backend as K
import tensorflow as tf
# set GPU memory 
if('tensorflow' == K.backend()):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.log_device_placement=True
    # config.gpu_options.allow_soft_placement = True
    config.gpu_options.per_process_gpu_memory_fraction=0.6
    sess = tf.Session(config=config)

norm_size = 200

def args_parse():
# construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True,
        help="path to trained model model")
    ap.add_argument("-i", "--image", required=True,
        help="path to input image")
    ap.add_argument("-s", "--show", action="store_true",
        help="show predict image",default=False)
    args = vars(ap.parse_args())    
    return args

    
def predict(args):
    # load the trained convolutional neural network
    print("[INFO] loading network...")
    model = load_model(args["model"])
    
    #load the image
    image = cv2.imread(args["image"])
    orig = image.copy()
     
    # pre-process the image for classification
    image = cv2.resize(image, (norm_size, norm_size))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
     
    # classify the input image
    result = model.predict(image)[0]
    #print (result.shape)
    proba = np.max(result)
    lbl_names = ["normal","missing"]
    # label = str(np.where(result==proba)[0])
    label = lbl_names[int(np.where(result==proba)[0])]
    label = "{}: {:.2f}%".format(label, proba * 100)
    print(label)
    
    if args['show']:   
        # draw the label on the image
        output = imutils.resize(orig, width=200)
        cv2.putText(output, label, (10, 25),cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2)       
        # show the output image
        cv2.imshow("Output", output)
        cv2.waitKey(0)


#python predict.py --model traffic_sign.model -i ../2.png -s
if __name__ == '__main__':
    args = args_parse()
    predict(args)