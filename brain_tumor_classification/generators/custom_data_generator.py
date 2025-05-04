import cv2
import numpy as np
from tensorflow.keras.utils import Sequence

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
        denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
        resized = cv2.resize(denoised, self.image_size)
        img_bgr = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        return img_bgr / 255.0
