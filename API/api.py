#@see:https://blog.techbridge.cc/2018/11/01/python-flask-keras-image-predict-api/
import io

# import the necessary packages
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify
import tensorflow as tf
import imutils
import cv2
import time
import numpy as np
import os,uuid

# initialize our Flask application and the Keras model
app = Flask(__name__)
model = None
norm_size = 64
#TODO：https://becominghuman.ai/creating-restful-api-to-tensorflow-models-c5c57b692c10

@app.route('/predict', methods=['POST'])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {'success': False}
    #print(request)
    # ensure an image was properly uploaded to our endpoint
    if request.method == 'POST':
        #  flask request（byte str）
        # post request handling
        file = request.files['file']
        extension = os.path.splitext(file.filename)[1]
        f_name = str(uuid.uuid4()) + extension
        upload_f_name = os.path.join("./uploads/", f_name)
        file.save(upload_f_name)
        print("upload saved filename:",upload_f_name)
        # return jsonify(code=200,data={'filename':f_name})
        # print("Python-OcrRecognition accepted request:",images)
        # OcrDetection/Cognition code block here
        cvImage = cv2.imread(upload_f_name)
        print("cvImage:",cvImage)
        (origH, origW) = cvImage.shape[:2]
        print("(origH, origW):",(origH, origW))
        # results = trm.recognition(cvImage,'frozen_east_text_detection.pb',0.05)
        # return jsonify(code=200,results=results)
    # print("text_recognition_module:",results)
    # ###
    # return jsonify(code=200,texts='1+1=2')
    # return Response(json.dumps(js),  mimetype='application/json')
#         image_byte_array = request.files['image'].read()
#         #print("image_byte_array:",image_byte_array)
#         # convert string of image data to uint8
#         image_np_array = np.fromstring(image_byte_array, np.uint8)
#         print("image_np_array:",image_np_array)
# # decode image
#         image = cv2.imdecode(image_np_array, cv2.IMREAD_COLOR)
        start_time = time.time()
        print("got image at: ",start_time)
         #load the image
        #image = cv2.imread(image)
        orig = cvImage.copy()
        # pre-process the image for classification
        image = cv2.resize(cvImage, (norm_size, norm_size))
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
        data['proba'] = str(proba)
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
       print('model is ready...')
#@notice:https://github.com/keras-team/keras/issues/2397
    app.run(host="0.0.0.0", port=5000, threaded=False)