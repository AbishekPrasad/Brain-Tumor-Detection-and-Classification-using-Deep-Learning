import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
import kagglehub

#Load dataset
path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
train_folder = os.path.join(path, "Training")
test_folder = os.path.join(path, "Testing")

# Custom data generator to apply preprocessing
class CustomDataGenerator(Sequence):
    def __init__(self, file_paths, labels, batch_size, image_size=(224, 224), num_classes=4, shuffle=True):
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.file_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_paths = self.file_paths[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]
        images = [self.preprocess_image(p) for p in batch_paths]
        return np.array(images), np.array(batch_labels)

    def on_epoch_end(self):
        if self.shuffle:
            zipped = list(zip(self.file_paths, self.labels))
            np.random.shuffle(zipped)
            self.file_paths, self.labels = zip(*zipped)

    def preprocess_image(self, image_path):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
        resized = cv2.resize(denoised, self.image_size)
        normalized = resized / 255.0
        return normalized

# Helper function to get file paths and labels
def get_data_paths_and_labels(directory):
    class_names = sorted(os.listdir(directory))
    file_paths = []
    labels = []
    for idx, class_name in enumerate(class_names):
        class_path = os.path.join(directory, class_name)
        for fname in os.listdir(class_path):
            file_paths.append(os.path.join(class_path, fname))
            one_hot = np.zeros(len(class_names))
            one_hot[idx] = 1
            labels.append(one_hot)
    return file_paths, labels

# Get data and create generators
train_paths, train_labels = get_data_paths_and_labels(train_folder)
test_paths, test_labels = get_data_paths_and_labels(test_folder)

train_generator = CustomDataGenerator(train_paths, train_labels, batch_size=32)
test_generator = CustomDataGenerator(test_paths, test_labels, batch_size=32)

# Load the VGG16 model with pre-trained weights, excluding the top fully connected layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base model

# Build the final model
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4, activation='softmax')  # 4 classes
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)
# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

#Save model
model.save('brain_tumor_classification_model.keras')

# Plot accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()