from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from joblib import load

app = Flask(__name__)
model = load('lung_cancer_classification.joblib')

def func(pred):
    if pred == 0:
        return 'Normal : Not cancerous'
    elif pred == 1:
        return 'Adenocarcinoma'
    elif pred == 2:
        return 'Large Cell Carcinoma'
    else :
        return 'Squamous Cell Carcinoma'

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    #img = Image.open(image_path)
    img = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    img=cv2.resize(img,(128,128))
    img=img.reshape(-1)
    img=img/255
    
    yhat = model.predict([img])
    
    #classification = np.argmax(yhat)
    pred = func(yhat[0])

    response = {'prediction': pred}

    return jsonify(response)

if __name__ == '__main__':
    app.run(port=3000, debug=True)
