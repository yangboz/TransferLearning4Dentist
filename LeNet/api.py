import io

# import the necessary packages
from keras.models import load_model
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify
import tensorflow as tf
import imutils
import cv2
import time

# initialize our Flask application and the Keras model
app = Flask(__name__)
model = None
norm_size = 224


@app.route('/predict', methods=['POST'])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {'success': False}
    print('request')
    # ensure an image was properly uploaded to our endpoint
    if request.method == 'POST':
        if request.files.get('image'):
            # 從 flask request 中讀取圖片（byte str）
            image = request.files['image'].read()
            start_time = time.time()
            print(image,start_time)
             #load the image
    #image = cv2.imread(args["image"])
    #orig = image.copy()
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
           #label = "{}: {:.2f}%".format(label, proba * 100)
            data['label'] = label
            data['proba'] = proba
            data['timeTotal'] = time.time() - start_time
            data['success'] = True
            print(data)

    return jsonify(data)

# 
if __name__ == '__main__':
    print(('* Loading Keras model and Flask starting server...'
        'please wait until server has fully started'))
    if model is None:
    	model = load_model("den_normal_missing.model")
    app.run()