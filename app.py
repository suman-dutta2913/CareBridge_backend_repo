# from flask import Flask, render_template, request
# import tensorflow as tf
# import numpy as np
# from PIL import Image

# app = Flask(__name__)
# model = tf.keras.models.load_model('lung_cancer_detection.keras')

# def func(value):
#     if value == 0:
#         return 'It is not a Tumor'
#     else:
#         return 'It is a tumor'

# @app.route('/', methods=['GET'])
# def hello_word():
#     return render_template('index.html')

# @app.route('/', methods=['POST'])
# def predict():
#     imagefile = request.files['imagefile']
#     image_path = "./images/" + imagefile.filename
#     imagefile.save(image_path)


#     img = Image.open(image_path)
#     img = img.convert('RGB')
#     img = img.resize((128, 128))  # Resize image to match model input shape
#     x = np.array(img)
#     x = x.reshape(1, 128, 128, 3)
#     yhat = model.predict(x)
#     classification = np.argmax(yhat)
#     pred = func(classification)

#     # return render_template('index.html', prediction=pred)
#     return "You have a tumor"


# if __name__ == '__main__':
#     app.run(port=3000, debug=True)

from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)
model = tf.keras.models.load_model('lung_cancer_detection.keras')

def func(value):
    if value == 0:
        return 'It is not a Tumor'
    else:
        return 'It is a tumor'

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    img = Image.open(image_path)
    img = img.convert('RGB')
    img = img.resize((128, 128))  # Resize image to match model input shape
    x = np.array(img)
    x = x.reshape(1, 128, 128, 3)
    yhat = model.predict(x)
    classification = np.argmax(yhat)
    pred = func(classification)

    if pred == 'It is a tumor':
        response = {'prediction': pred, 'tumor': True}
    else:
        response = {'prediction': pred, 'tumor': False}

    return jsonify(response)

if __name__ == '__main__':
    app.run(port=3000, debug=True)
