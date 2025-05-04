from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2


model = load_model("brain_tumor_classification_model.keras")
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']


def preprocess_image(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)  # denoise
    img = cv2.resize(img, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Convert back to 3 channels
    img = img / 255.0  # normalize
    img = np.expand_dims(img, axis=0)  # add batch dimension
    return img

def predict_image(image_path):
    processed_image = preprocess_image(image_path)
    prediction = model.predict(processed_image)[0]
    predicted_class_index = np.argmax(prediction)
    confidence = prediction[predicted_class_index]
    predicted_class = class_names[predicted_class_index]
    return predicted_class, confidence

# Example usage
image_path = 'test.jpg'  # replace with your image
predicted_class, confidence = predict_image(image_path)
print(f"Prediction: {predicted_class} ({confidence*100:.2f}%)")