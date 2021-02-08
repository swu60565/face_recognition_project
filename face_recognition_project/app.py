from flask import Flask, request, jsonify, render_template
import joblib

import numpy as np
from numpy import asarray
from numpy import array
from numpy import expand_dims
from numpy import reshape
from numpy import load
from numpy import max

import pandas as pd 
import cv2 
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
from PIL import Image

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsClassifier

import re
import base64
from imageio import imsave, imread

app = Flask(__name__)

# load the facenet model
facenet_model = load_model('model and data/facenet_keras.h5')

# load the model from the file 
knn_model = joblib.load('model and data/knn_model.pkl')  

# load the face dataset
data = np.load('model and data/Final_Dataset.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

dataemd = np.load('model and data/embed_Final_Dataset.npz')
emdTrainX, trainy, emdTestX, testy = dataemd ['arr_0'], dataemd ['arr_1'], dataemd ['arr_2'], dataemd ['arr_3']

# def convertImage(imgData1):
#       print("convertImage")
#   imgreg = re.search(b'base64,(.*)',imgData1)
#   imgstr = imgreg.group(1)
#   imgstr_64 = base64.b64decode(imgstr)
#   with open('output.jpg','wb') as output:
#           output.write(imgstr_64)

#extracting face
def extract_image(image):
  img1 = Image.open(image)
  img1 = img1.convert('RGB')
  pixels = asarray(img1)
  detector = MTCNN()
  f = detector.detect_faces(pixels)
  x1,y1,w,h = f[0]['box']
  x1, y1 = abs(x1), abs(y1)
  x2 = abs(x1+w)
  y2 = abs(y1+h)
  store_face = pixels[y1:y2,x1:x2]
  image1 = Image.fromarray(store_face,'RGB')
  image1 = image1.resize((160,160))
  face_array = asarray(image1)
  return face_array

#extracting embeddings
def extract_embeddings(model,face_pixels):
  face_pixels = face_pixels.astype('float32')
  mean = face_pixels.mean()
  std  = face_pixels.std()
  face_pixels = (face_pixels - mean)/std
  samples = expand_dims(face_pixels,axis=0)
  yhat = model.predict(samples)
  return yhat[0]

@app.route('/')
def predict():      
    #load data and reshape the image
    # imgData = request.get_json().encode()
    # convertImage(imgData)
    # Img = imread('output.png' , pilmode = "L")
    Img = 'test image/20200918_152650.jpg'
    face = extract_image(Img)
    testx = asarray(face)
    testx = testx.reshape(-1,160,160,3)

    #find embeddings
    new_testx = list()
    for test_pixels in testx:
          embeddings = extract_embeddings(facenet_model,test_pixels)
          new_testx.append(embeddings)
          new_testx = asarray(new_testx)  

    #normalize the input data 
    in_encode = Normalizer(norm='l2')
    emdTrainX_norm = in_encode.transform(emdTrainX)
    new_testx = in_encode.transform(new_testx)

    #create a label vector
    new_testy = trainy 
    out_encode = LabelEncoder()
    out_encode.fit(trainy)
    trainy_enc = out_encode.transform(trainy)
    new_testy = out_encode.transform(new_testy)

    #predict 
    predict_train = knn_model.predict(emdTrainX)
    predict_test = knn_model.predict(new_testx)

    #get the confidence score
    probability = knn_model.predict_proba(new_testx)
    confidence = max(probability)
    predict_names = out_encode.inverse_transform(predict_test)

    title = '%s (%.3f)' % (predict_names[0], confidence) 

    return jsonify(title)

if __name__ == "__main__":
    app.debug = True
    # app.run(host = 'ip address', port = 5000)
    app.run()