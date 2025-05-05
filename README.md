# Brain-Tumor-Detection-and-Classification-using-Deep-Learning

This project aims to classify brain MRI images into four categories — **Glioma**, **Meningioma**, **Pituitary Tumor**, or **No Tumor** — using a Convolutional Neural Network (CNN) and visualize tumor regions using OpenCV. The model is trained using TensorFlow/Keras and supports both prediction and visualization.

---

## 📂 Project Structure
brain-tumor-classification/
│
├── data/ # Dataset (Training / Testing splits)
├── model/
│ └── brain_tumor_classification_model.keras # Trained Keras model
├── notebook/
│ └── BrainTumorTraining.ipynb # Jupyter notebook (optional)
├── scripts/
│ ├── train.py # Model training script
│ └── predict.py # Tumor prediction and visualization
├── utils/
│ ├── data_generator.py # Custom data generator for training
│ └── preprocessing.py # Image preprocessing utilities
├── test_images/
│ └── test.jpg # Sample test image
├── main.py # Example script to test model prediction
├── requirements.txt # Python dependencies
└── README.md

---

## 🧪 Model Overview

- Architecture: Convolutional Neural Network (CNN)
- Input: MRI brain image (`224x224x3`)
- Output: One of four classes:
  - `glioma`
  - `meningioma`
  - `pituitary`
  - `no_tumor`

---

## 🖼 Sample Output

The model detects tumors and draws bounding boxes around them using OpenCV:

![sample_output](https://github.com/user-attachments/assets/a7eb0fd7-53d2-4c17-a707-6e9e7249265e)




