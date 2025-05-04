from predict.predict import predict_image
predicted_class, confidence = predict_image('brain_tumor_classification_model.keras', 'test.jpg')
print(f"Predicted: {predicted_class}, Confidence: {confidence*100:.2f}%")
