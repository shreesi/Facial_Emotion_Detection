from ctypes import util
from flask import Flask, render_template, request
import tensorflow as tf
import keras
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pyttsx3


app = Flask(__name__)
model = tf.keras.models.load_model('model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
   
    
    objects = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')
    
    img1 = request.files['image']
    img1.save('static/file.jpg')

    im = cv2.imread('static/file.jpg')
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

    faces = cascade.detectMultiScale(gray, 1.1, 3)

    for x,y,w,h in faces:
        cv2.rectangle(im, (x,y), (x+w, y+h), (0,255,0), 2)

        cropped = im[y:y+h, x:x+w]

    cv2.imwrite('static/after.jpg', im)

    try:
        cv2.imwrite('static/cropped.jpg', cropped)

    except:
        pass

    #####################################

    try:
        image1 = cv2.imread('static/cropped.jpg', 0)
    except:
        image1 = cv2.imread('static/file.jpg', 0)

    image1 = cv2.resize(image1,(48,48))
    x = keras.utils.img_to_array(image1)
    x = np.expand_dims(x, axis = 0)

    x /= 255

    custom = model.predict(x)

    x = np.array(x, 'float32')
    x = x.reshape([48, 48]);


    m=0.000000000000000000001
    a=custom[0]
    for i in range(0,len(a)):
        if a[i]>m:
            m=a[i]
            ind=i

    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    rate = engine.getProperty('rate')
    engine.setProperty('voice', voices[1].id) # 0 for male & 1 for female
    engine.setProperty('rate', rate-75) # default rate is 200
    engine.say(objects[ind])
    engine.runAndWait()
            
    return render_template('after.html', prediction_text=objects[ind])

if __name__ == "__main__":
    app.run(debug=True)