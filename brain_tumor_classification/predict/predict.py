import numpy as np
import cv2
from tensorflow.keras.models import load_model

class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

def preprocess_image(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    img = cv2.resize(img, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_image(model_path, image_path):
    model = load_model(model_path)
    image = preprocess_image(image_path)
    prediction = model.predict(image)[0]
    class_index = np.argmax(prediction)
    return class_names[class_index], prediction[class_index]
