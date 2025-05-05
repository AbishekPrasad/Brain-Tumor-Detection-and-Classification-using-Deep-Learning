import cv2
import numpy as np
import matplotlib.pyplot as plt

class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

def preprocess_image(image_path, target_size=(224, 224)):
    original = cv2.imread(image_path)
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    resized = cv2.resize(filtered, target_size)
    colored = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    normalized = colored / 255.0
    return original, np.expand_dims(normalized, axis=0)

def predict_image(model_path, image_path):
    model = load_model(model_path)
    original, input_img = preprocess_image(image_path)
    prediction = model.predict(input_img)[0]
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_names[predicted_class_index]
    confidence = prediction[predicted_class_index]

    if predicted_class != 'no_tumor':
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 130, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) > 100:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(original, (x, y), (x + w, y + h), (255, 255, 255), 2)
                cv2.putText(original, f"{predicted_class} ({confidence*100:.1f}%)",
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


    rgb_image = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 6))
    plt.imshow(rgb_image)
    plt.title("Tumor Detection Result")
    plt.axis('off')
    plt.show()

    return predicted_class, confidence
